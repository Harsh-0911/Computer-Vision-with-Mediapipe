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
# Module written by me on top of 'mediapipe' for detecting hands
from modules import HandDetector as hd


# ==================== Initializing Objects ====================

# Initializes the hand detector with minimun detection confidence level of 0.70
detector = hd.HandDetector(min_detection_confidence=0.7)
# Initializes video capture object for accessing camera
cap = cv2.VideoCapture(0)


# ==================== Variable Declaration ====================

# Width and Height of camera frame
WIDTH = 640
HEIGHT = 480
# Used for calculating frame rate
current_time = 0
start_time = 0
# This list will show which finger is up and which is down -> [thumb, index, middle, ring, pinky]
fingers = []


# Setting width and height of camera frame
cap.set(3, WIDTH)
cap.set(4, HEIGHT)

# Execution

# Infinite loop, breaks when 'q' is pressed
while True:
    # Reading camera frame by frame
    ret, frame = cap.read()
    # Finding hands in frame
    frame = detector.find_hands(frame, annote=True)
    # Finding landmarks
    landmarks = detector.get_landmarks(frame)

    # Checking if we got any landmarks or not
    if len(landmarks) > 0:
        # Clearing the fingers list, so that we will not again append in the populated list
        fingers.clear()

        # We will deal with thumb saperately
        # Ids of tip of the fingers will be stored. [index, middle, ring, pinky]
        tips_id = [8, 12, 16, 20]
        # Ids of pip of the fingers will be stored. [index, middle, ring, pinky]
        pips_id = [6, 10, 14, 18]

        # ==================== Dealing with thumb ====================
        # Logic we will use to decide whether the thumb is up or not is, if the x coordinate of the tip of the thumb is lower than the ip of the thumb, then thumb is said to be up. (ONLY RIGHT HAND)
        if landmarks[4][3] == "Right":
            if landmarks[4][1] < landmarks[3][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        elif landmarks[4][3] == "Left":
            if landmarks[4][1] > landmarks[3][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        else:
            pass

        # ==================== Dealing with fingers ====================
        # Logic we will use to decide whether the finger is up or not is, if the y coordinate of the tip of the finger is lower than the pip of the finger, then finger is said to be up.
        for tip, pip in zip(tips_id, pips_id):
            if landmarks[tip][2] < landmarks[pip][2]:
                fingers.append(1)
            else:
                fingers.append(0)

    # Checking how many fingers are up
    if len(fingers) > 0:
        total_no_of_fingers = sum(fingers)
        cv2.putText(frame, f'No. of fingers: {total_no_of_fingers}', (
            10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Frame rate calculation
    current_time = time()
    frame_rate = int(1 / (current_time - start_time))
    start_time = current_time
    cv2.putText(frame, str(frame_rate), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Displaying frame, if 'q' is pressed then exit the loop
    cv2.imshow('Live Feed', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Releasing camera instance
cap.release()
# Destroying all windows
cv2.destroyAllWindows()
