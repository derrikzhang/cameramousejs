from api.requests import getLatestAppSettingsFromServer
from videoProcessing.track2Command import convertFaceTrackingToMouseMovement, sendTrackingInfo
from videoProcessing.ssdFaceTrack import getFrameSize, trackFace
from videoProcessing.trackerState import trackerState
import cv2
import os
import sys

if __name__ == "__main__":


    frameSize = getFrameSize()
    trackerState.setWebcamFrameSize(frameSize[0], frameSize[1])
    count = 0
    while True:
        face, pose, pos, guesture, face_confidence, gesture_confidences = trackFace(trackerState)
        sendTrackingInfo(face, frameSize, pose, pos, guesture, face_confidence, gesture_confidences)
        convertFaceTrackingToMouseMovement(face, frameSize, pose, pos, guesture, trackerState)
        
        # get config every now and then
        if count % 20 == 0:
            getLatestAppSettingsFromServer(trackerState)

        # Stop if escape key is pressed
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        # On Windows, close both windows if the close button is pressed
        if sys.platform == 'win32':
            if cv2.getWindowProperty('Face Tracker', cv2.WND_PROP_VISIBLE) < 1:        
                break

        cv2.setWindowProperty('Face Tracker', cv2.WND_PROP_TOPMOST, 1)
        # reset to prevent overflow
        count = 0 if count > 100 else count + 1

    cv2.destroyAllWindows()
