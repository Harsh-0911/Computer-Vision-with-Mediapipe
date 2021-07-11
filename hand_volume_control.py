# ==================== Importing libraries ====================

# Classic computer vision module
import cv2
# Inbuilt time module, required for frame rate calculation
from time import time
# Inbuilt math module, required for calculating distance between thumb and index finger
from math import hypot
# Classic module for working with arrays, required for mapping distance to the volume range
from numpy import interp
# Library by google, required for detecting and tracking hand
import mediapipe as mp
# These libraries are used for changing system volume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
# Module written by me on top of 'mediapipe' for detecting hands
from modules import HandDetector as hd


# ==================== Initializing Objects ====================

# This will initialize the volume, to interact with system volume
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# Initializes the hand detector with minimun detection confidence level of 0.70
detector = hd.HandDetector(min_detection_confidence=0.7)
# Initializes video capture object for accessing camera
cap = cv2.VideoCapture(0)


# ==================== Variable Declaration ====================

# Width and Height of camera frame
WIDTH = 640
HEIGHT = 480
# Minimim and Maximum values of system volume range, 3rd return value is not required.
MIN_VOL, MAX_VOL, _ = volume.GetVolumeRange()
# Used for calculating frame rate
current_time = 0
start_time = 0
# Minimum and Maximum values of length of line drawn between thumb and index finger. This depends on the distance between hand and camera. Tweak according to your need.
MIN_LEN = 15
MAX_LEN = 180
# Initial bar height
bar_height = 400

# Setting width and height of camera frame
cap.set(3, WIDTH)
cap.set(4, HEIGHT)


# ==================== Execution ====================

# Infinite loop, breaks when 'q' is pressed
while True:
    # Reading camera frame by frame
    ret, frame = cap.read()
    # Finding hands in frame
    frame = detector.find_hands(frame, annote=True)
    # Finding landmarks, we will use thumb tip and index finger tip particularly
    landmarks = detector.get_landmarks(frame)

    # Check if landmark is detected or not
    if len(landmarks) != 0:
        # Thumb tip, index no. 4
        thumb = landmarks[4]
        # Index finger tip, index no. 8
        index = landmarks[8]
        # Center point of thumb and index finger tip
        center = [(thumb[1] + index[1]) // 2, (thumb[2] + index[2]) // 2]

        # Highlighting thumb tip
        cv2.circle(frame, (thumb[1], thumb[2]), 10, (255, 0, 255), -1)
        # Highlighting index finger tip
        cv2.circle(frame, (index[1], index[2]), 10, (255, 0, 255), -1)
        # Drawing circle at the center of the thumb tip and index finger tip
        cv2.circle(frame, (center[0], center[1]), 10, (255, 0, 255), -1)
        # Drawing line from thumb tip to index finger tip
        cv2.line(frame, (thumb[1], thumb[2]),
                 (index[1], index[2]), (255, 0, 255), 2)

        # calculating length of line
        length = hypot(index[1] - thumb[1], index[2] - thumb[2])

        # Converting length into range of volume range
        volume_level = interp(length, [MIN_LEN, MAX_LEN], [MIN_VOL, MAX_VOL])
        # Converting length into range of height of rectangle
        # 400 means volume is 0 and 150 means volume is 100
        bar_height = interp(length, [MIN_LEN, MAX_LEN], [400, 150])

        # Setting the system volume according to the volume level calculated based on distance between thumb and index finger
        volume.SetMasterVolumeLevel(volume_level, None)

        # Changing color of middle circle when length is less than minimun length to show button pressed like effect
        if length <= MIN_LEN:
            cv2.circle(frame, (center[0], center[1]), 10, (0, 255, 0), -1)

    # Creating volume bar on left

    # Creating outer rectangle
    cv2.rectangle(frame, (30, 150), (65, 400), (0, 255, 0), 2)
    # This rectangle will show the actucal volume level, initial height will be 400 (showing volume 0)
    cv2.rectangle(frame, (30, int(bar_height)), (65, 400), (0, 255, 0), -1)

    # Frame rate calculation
    current_time = time()
    frame_rate = int(1 / (current_time - start_time))
    start_time = current_time
    cv2.putText(frame, str(frame_rate), (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2, cv2.LINE_AA)

    # Displaying frame, if 'q' is pressed then exit the loop
    cv2.imshow('Live Feed', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Releasing camera instance
cap.release()
# Destroying all windows
cv2.destroyAllWindows()
