from api.requests import getLatestAppSettingsFromServer
from videoProcessing.track2Command import convertFaceTrackingToMouseMovement, sendTrackingInfo
from videoProcessing.ssdFaceTrack import getFrameSize, trackFace
from videoProcessing.trackerState import trackerState
import cv2
import os
import sys

import cProfile
import re
import pstats



if __name__ == "__main__":

    profiler = cProfile.Profile()

    cap = cv2.VideoCapture(0)
    prev_time = 0

    try:

        frameSize = getFrameSize()
        trackerState.setWebcamFrameSize(frameSize[0], frameSize[1])
        count = 0
        while True:

            context = globals().copy()  # Start with a copy of globals
            context.update(locals())    # Update with all local variables
            
            face, pose, pos, guesture, face_confidence, gesture_confidences = trackFace(trackerState)
            
            profiler.enable()
            sendTrackingInfo(face, frameSize, pose, pos, guesture, face_confidence, gesture_confidences)
            convertFaceTrackingToMouseMovement(face, frameSize, pose, pos, guesture, trackerState)
            profiler.disable()


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


    finally:
        profiler.disable()
        with open('profile_output.txt', 'w') as f:
            s = pstats.Stats(profiler, stream=f)
            s.strip_dirs().sort_stats('time').print_stats()
    cap.release()
    cv2.destroyAllWindows()
