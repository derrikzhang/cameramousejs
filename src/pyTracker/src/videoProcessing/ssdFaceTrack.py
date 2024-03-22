from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
from utils.utils import drawScalingBox
from videoProcessing.config import config
from videoProcessing.poseTracker.FSANET_model import *
import dlib
from imutils import face_utils

#SSD initialization
currDirectory = os.path.dirname(os.path.abspath(__file__))
prototxt =  currDirectory + "/deploy.prototxt.txt"
model = currDirectory + "/res10_300x300_ssd_iter_140000.caffemodel"
confidenceThreshold = 0.9 #Minimum confidence for an object to be recognized as a face
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# pose tracker
# load model and weights
if config.DETECT_POSE:
	stage_num = [3,3,3]
	num_capsule = 3
	dim_capsule = 16
	routings = 2
	stage_num = [3,3,3]
	lambda_d = 1
	num_classes = 3
	image_size = 64
	ad = 0.6
	num_primcaps = 7*3
	m_dim = 5
	S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
	pose_model = FSA_net_Capsule(image_size, num_classes, stage_num, lambda_d, S_set)()
	weight_file = currDirectory + "/fsanet_capsule_3_16_2_21_5.h5"
	pose_model.load_weights(weight_file)

# parameters for template matching
template_size = 30
# template_size = 0.04
search_size = 0.5
prev_pos = []
prev_match_template_res = np.array([[]])
template = []
method = cv2.TM_CCOEFF_NORMED
dist_threshold = 0.25

# parameters for tracking using optical flow
# parameters for ShiTomasi corner detection
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
# parameters for Lucas Kanade optical flow
lk_params = dict(
	winSize=(15, 15),
	maxLevel=2,
	criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)
prev_gray = []
p0 = []

# facial landmark detection
# landmark_predictor = dlib.shape_predictor(currDirectory + "/shape_predictor_5_face_landmarks.dat")
landmark_predictor_68 = dlib.shape_predictor(currDirectory + "/shape_predictor_68_face_landmarks.dat")
MOUTH_AR_THRESH = 0.70
EYEBROW_DIST_THRESHOLD=0.65

# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: Create an FaceLandmarker object.
mp_base_options = python.BaseOptions(model_asset_path=currDirectory + '/face_landmarker.task')
mp_options = vision.FaceLandmarkerOptions(base_options=mp_base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=False,
                                       num_faces=1)
mp_detector = vision.FaceLandmarker.create_from_options(mp_options)
EYEBROW_MP_THRESHOLD = 0.4
MOUTH_MP_THRESHOLD = 0.4

op_window_size = None

num_frames = -1
vs = VideoStream(src=0).start()
time.sleep(2.0)

"""
returns the size of the frame on which faces are detected.
Note that config.FRAME_WIDTH and config.FRAME_HEIGHT can't be used directly because the 
final width and height of the frame might change depending on the aspect ratio of the camera

for eg: when config.FRAME_WIDTH and config.FRAME_HEIGHT are both 300 and aspect ratio of camera is 1920/1080,
        eventual height of the frame is 168 (not 300) ,after resizing
"""
def getFrameSize():
	frame = vs.read()
	if frame is None:
		return (config.FRAME_WIDTH,config.FRAME_HEIGHT)
	frame = imutils.resize(frame, width=config.FRAME_WIDTH, height=config.FRAME_HEIGHT)
	(h, w) = frame.shape[:2]
	#print("Webcam size (h, w)",(h, w) )
	return (w,h)

from math import cos, sin, hypot
def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 50):
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)

    return img

def templateTrack(frame, face, shape):
	# tracking using template matching
	global template, prev_pos, prev_match_template_res, search_size, dist_threshold

	if len(face) > 0:
		# rect = dlib.rectangle(left=face[0], top=face[1], 
		# 	right=face[0]+face[2], bottom=face[1]+face[3])
		# shape = landmark_predictor(frame, rect)
		# shape = face_utils.shape_to_np(shape)
		# center = (shape[1] + shape[2]) / 2 # middle of eyes
		# center = (shape[0] + shape[1]) / 2 # center of right eye to make sure enough distinctive feature
		# center = shape[27] # center of two eyes
		# center = shape[33] # nostril
		center = shape[94] # nostril

		(h, w) = frame.shape[:2]
		if len(template) == 0 or hypot(center[0]-template_size-prev_pos[0], center[1]-template_size-prev_pos[1]) > dist_threshold * face[2]:
			# initialize with eyes
			template = frame[int(center[1]-template_size):int(center[1]+template_size), 
					int(center[0]-template_size):int(center[0]+template_size), :]
			prev_pos = [int(center[0]-template_size), int(center[1]-template_size)]

		# if len(template) == 0 or hypot(center[0]-template_size*h-prev_pos[0], center[1]-template_size*h-prev_pos[1]) > dist_threshold * face[2]:
			# initialize with eyes
			# template = frame[int(center[1]-template_size*h):int(center[1]+template_size*h), 
			# 		int(center[0]-template_size*h):int(center[0]+template_size*h), :]
			# prev_pos = [int(center[0]-template_size*h), int(center[1]-template_size*h)]

			# initialize using the center of the face
			# center = (face[0] + face[2] / 2, face[1] + face[3] / 2)
			# template = frame[int(center[1]-template_size*h):int(center[1]+template_size*h), 
			# 			int(center[0]-template_size*h):int(center[0]+template_size*h), :]
			# prev_pos = [int(center[0]-template_size*h), int(center[1]-template_size*h)]

	(th, tw) = template.shape[:2]

	# only search around the previous matching position
	start_y = prev_pos[1]-int(th*search_size*2)
	start_x = prev_pos[0]-int(tw*search_size*2)
	end_y = prev_pos[1]+int((1 + 2*search_size)*th)
	end_x = prev_pos[0]+int((1 + 2*search_size)*tw)
	# start_x, start_y, end_x, end_y = face[0], face[1], face[0]+face[2], face[1]+face[3]
	img = frame[start_y:end_y, start_x:end_x, :]

	
	try:
		res = cv2.matchTemplate(img, template, method)
		prev_match_template_res = res
		# print("prev_match_template_res", prev_match_template_res)
	except Exception:
		res = prev_match_template_res
	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
	# If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
	if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
		top_left = min_loc
	else:
		top_left = max_loc
	bottom_right = (top_left[0] + tw, top_left[1] + th)
	top_left = (top_left[0] + start_x, top_left[1] + start_y)
	bottom_right = (bottom_right[0] + start_x, bottom_right[1] + start_y)

	# update previous matching position
	prev_pos = top_left
	# update template, use previous best match as the new template
	if num_frames % config.TEMPLATE_FREQ == 0:
		template = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0], :]

	# cv2.imshow("Template", template)

	return top_left, bottom_right



def opticalFlow(frame, face):
	frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	global p0, prev_gray, feature_params, lk_params
	if len(prev_gray) == 0:
		# initialize
		face_mask = np.zeros(frame_gray.shape, dtype=np.uint8)
		face_mask[face[1]:face[1]+face[3], face[0]:face[0]+face[2]] = 1
		p0 = cv2.goodFeaturesToTrack(frame_gray, mask=face_mask, **feature_params)
		prev_gray = frame_gray.copy()
	else:
		# Calculate Optical Flow
		p1, st, err = cv2.calcOpticalFlowPyrLK(
			prev_gray, frame_gray, p0, None, **lk_params
		)
		# Select good points
		good_new = p1[st == 1]
		# good_old = p0[st == 1]

		# Update the previous frame and previous points
		prev_gray = frame_gray.copy()
		p0 = good_new.reshape(-1, 1, 2)

		for new in good_new:
			a, b = new.ravel()
			if face[0] + face[2]/3 <= int(a) <= face[0] + face[2]*2/3 \
				and face[1] + face[3]/3 <= int(b) <= face[1] + face[3]*2/3:
				return new	
	return []

from scipy.spatial import distance as dist
def detect_mouth_open(mouth):
	# compute the euclidean distances between the two sets of
	# vertical mouth landmarks (x, y)-coordinates
	A = dist.euclidean(mouth[2], mouth[10]) # 51, 59
	B = dist.euclidean(mouth[4], mouth[8]) # 53, 57

	# compute the euclidean distance between the horizontal
	# mouth landmark (x, y)-coordinates
	C = dist.euclidean(mouth[0], mouth[6]) # 49, 55

	# compute the mouth aspect ratio
	mar = (A + B) / (2.0 * C)

	return mar > MOUTH_AR_THRESH

def detect_eyebrows_raised(left_eyebrow, right_eyebrow, left_eye, right_eye):
	# y distance between eyebrows and eyes relative to the width of eyebrows
	relative_left = abs(left_eyebrow[2][1] - left_eye[-1][1]) / dist.euclidean(left_eyebrow[0], left_eyebrow[-1])
	relative_right = abs(right_eyebrow[2][1] - right_eye[-1][1]) / dist.euclidean(right_eyebrow[0], right_eyebrow[-1])

	return min(relative_left, relative_right) > EYEBROW_DIST_THRESHOLD

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
def trackFaces(trackerState):
	# grab the frame from the threaded video stream and resize it to the global frame size 
	frame = vs.read()

	# return if frame is None
	if frame is None:
		return
	
	frame = imutils.resize(frame, width=config.FRAME_WIDTH, height=config.FRAME_HEIGHT)

	# flip image
	frame = cv2.flip(frame, 1)

	# copy frame for traditional tracking
	ori_frame = frame.copy()

	faces = []
	poses = []
	is_mouth_open = False
	are_eyebrows_raised = False
	shapes = []

	face_confidence = 0
	eyebrow_confidence = 0
	mouth_confidence = 0

	global num_frames
	num_frames += 1
	# detect face every # frames
	if num_frames % config.FACE_FREQ == 0:

		# grab the frame dimensions and convert it to a blob
		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(frame, 1.0,
			(300, 300), (104.0, 177.0, 123.0))
		# pass the blob through the network and obtain the detections and predictions
		net.setInput(blob)
		detections = net.forward()

		

		for i in range(0, detections.shape[2]):
		# extract the confidence associated with the prediction
			confidence = detections[0, 0, i, 2]
			# filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
			if confidence < confidenceThreshold:
				continue

			# compute the (x, y)-coordinates of the bounding box for the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			face = (startX, startY, endX-startX, endY-startY)
			faces.append(face)

			if config.DETECT_POSE:
				# estimate pose
				x1, y1, x2, y2 = startX, startY, endX, endY
				xw1 = max(int(x1 - ad * face[2]), 0)
				yw1 = max(int(y1 - ad * face[3]), 0)
				xw2 = min(int(x2 + ad * face[2]), w - 1)
				yw2 = min(int(y2 + ad * face[3]), h - 1)
				face_img = cv2.resize(frame[yw1:yw2 + 1, xw1:xw2 + 1, :], (image_size, image_size))
				face_img = cv2.normalize(face_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)        
				face_img = np.expand_dims(face_img, axis=0)
				p_result = pose_model(face_img)
				yaw, pitch, roll = p_result[0][0], p_result[0][1], p_result[0][2]
				poses.append((yaw, pitch, roll))
				# draw pose
				draw_pose_result = draw_axis(frame[yw1:yw2 + 1, xw1:xw2 + 1, :], yaw, pitch, roll)
				frame[yw1:yw2 + 1, xw1:xw2 + 1, :] = draw_pose_result


			# Load the input image.
			image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
			
		
			# Detect face landmarks from the input image.
			detection_result = mp_detector.detect(image)
			if detection_result and len(detection_result.face_blendshapes) > 0:
				# print(detection_result.face_blendshapes[0][3])
				# print(detection_result.face_blendshapes[0][25])
				eyebrow_confidence = detection_result.face_blendshapes[0][3].score
				mouth_confidence = detection_result.face_blendshapes[0][25].score

				are_eyebrows_raised = eyebrow_confidence > trackerState.eyebrowGestureThreshold
				is_mouth_open = mouth_confidence > trackerState.mouthGestureThreshold
				# annotated_image = frame.copy()
				# https://github.com/google/mediapipe/blob/a908d668c730da128dfa8d9f6bd25d519d006692/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
				# for f in detection_result.face_landmarks:
				# 	for l in f[94:95]:
				# 		px, py = solutions.drawing_utils._normalized_to_pixel_coordinates(l.x, l.y, frame.shape[1], frame.shape[0])
				# 		cv2.circle(annotated_image, (px, py), 1, (0, 0, 255), -1)
				# cv2.imshow("mediapipe", annotated_image)
				shapes.append([solutions.drawing_utils._normalized_to_pixel_coordinates(l.x, l.y, frame.shape[1], frame.shape[0]) for l in detection_result.face_landmarks[0]])
			else:
				eyebrow_confidence = -1
				mouth_confidence = -1

			# 68 facial landmarks	
			# rect = dlib.rectangle(left=face[0], top=face[1], 
			# right=face[0]+face[2], bottom=face[1]+face[3])
			# shape = landmark_predictor_68(ori_frame, rect)
			# shape = face_utils.shape_to_np(shape)
			# detect mouth open
			# is_mouth_open = detect_mouth_open(shape[48:68])
			# for (x, y) in shape:
			# 	cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
			# if is_mouth_open:
			# 	cv2.putText(frame, "Mouth Open", (endX, endY),
			# 	cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)

			# detect if eyebrows are raised
			# left_eyebrow = shape[17:22]
			# right_eyebrow = shape[22:27]
			# left_eye = shape[36:42]
			# right_eye = shape[42:48]
			# are_eyebrows_raised = detect_eyebrows_raised(left_eyebrow, right_eyebrow, left_eye, right_eye)

			# if are_eyebrows_raised:
			# 	cv2.putText(frame, "Eyebrows raised", (endX, startY),
			# 	cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)
			
			# shapes.append(shape)

			# draw the bounding box of the face along with the associated probability
			face_confidence = confidence
			text = "{:.2f}%".format(confidence * 100)
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 0, 255), 2)
			cv2.putText(frame, text, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
	
	pos = []
	# tracking using template matching 
	if shapes or len(template) != 0:
		# select the largest face
		# sorted_face = sorted(faces, key=lambda x : x[2] * x[3])
		# target_face = sorted_face[-1] if sorted_face else []
		sorted_idx = sorted(range(len(faces)), key=lambda k: faces[k][2] * faces[k][3])
		target_face = faces[sorted_idx[-1]] if sorted_idx else []
		# target_shape = shapes[sorted_idx[-1]] if sorted_idx else []
		if len(shapes) > 0:
			target_shape = shapes[0]
		else:
			target_face = []
			target_shape = []
		top_left, bottom_right = templateTrack(ori_frame, target_face, target_shape)
		# print("template box ", top_left[0] - bottom_right[0])
		# cv2.rectangle(frame, top_left, bottom_right, (0, 0, 0), 2)
		# pos = top_left
		pos = [(top_left[0] + bottom_right[0]) / 2, (top_left[1] + bottom_right[1]) / 2]
		cv2.circle(frame, (int(pos[0]), int(pos[1])), radius=5, color=(0, 0, 255), thickness=-1)


	# tracking using optical flow (not stable)
	# if faces:
	# 	sorted_face = sorted(faces, key=lambda x : x[2] * x[3])
	# 	target_face = sorted_face[0] if sorted_face else []
	# 	pos = opticalFlow(ori_frame, target_face)
	# 	if len(pos) > 0:
	# 		a, b = pos.ravel()
	# 		cv2.circle(frame, (int(a), int(b)), 5, (255, 0, 0), 2)
	# 		pos = (int(a), int(b))
	# 	else:
	# 		pos = [0, 0]

	# tracking with facial landmarks
	# if faces:
	# 	sorted_face = sorted(faces, key=lambda x : x[2] * x[3])
	# 	target_face = sorted_face[0] if sorted_face else []
	# 	rect = dlib.rectangle(left=target_face[0], top=target_face[1], 
	# 		right=target_face[0]+target_face[2], bottom=target_face[1]+target_face[3])
	# 	shape = landmark_predictor(frame, rect)
	# 	shape = face_utils.shape_to_np(shape)
	# 	for (x, y) in shape[:2]:
	# 		cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
	# 	pos = shape[-1]
	startX, startY, endX, endY = drawScalingBox(cv2, frame)

	# crop the image
	cropped = frame.copy()[startY-15:endY+15, startX-15:endX+15]
	global op_window_size
	if op_window_size != None and op_window_size != cropped.shape:
		cv2.destroyWindow("Face Tracker")
	op_window_size = cropped.shape	
	

	cv2.imshow("Face Tracker", cropped)

	# print(pos)
	return faces, poses, pos, [is_mouth_open, are_eyebrows_raised], face_confidence, [eyebrow_confidence, mouth_confidence]

def trackFace(trackerState):
	faces, poses, pos, gesture, face_conf, gesture_confidences = trackFaces(trackerState) or (None,None,None,None)
	if faces:
		if config.DETECT_POSE:
			##TO DO: Select the closest face to the camera
			return faces[0], poses[0], pos, gesture, face_conf, gesture_confidences
		else:
			return faces[0], [], pos, gesture, face_conf, gesture_confidences
	else:
		# print("failed to detect a face!")
		return [], [], pos, gesture, face_conf, gesture_confidences