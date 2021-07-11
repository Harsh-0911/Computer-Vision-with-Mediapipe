# ==================== Importing libraries ====================

# Classic os module, used for getting header images from folder
import os
# Classic computer vision module
import cv2
# Inbuilt time module, required for frame rate calculation
from time import time
# Classic module for working with arrays
import numpy as np
# Library by google, required for detecting and tracking hand
import mediapipe as mp
# Module written by me on top of 'mediapipe' for detecting hands
from modules import HandDetector as hd


# ==================== Initializing Objects ====================

# Initializes the hand detector with minimun detection confidence level of 0.85
# We will only track hand if we pretty sure about it
detector = hd.HandDetector(min_detection_confidence=0.60, min_tracking_confidence=0.70)
# Initializes video capture object for accessing camera
cap = cv2.VideoCapture(0)


# ==================== Variable Declaration ====================

# Width and Height of camera frame
WIDTH = 1280
HEIGHT = 720
# Used for calculating frame rate
current_time = 0
start_time = 0
# This list will show which finger is up and which is down -> [thumb, index, middle, ring, pinky]
fingers = []
# Path of the folder where all header file resides
HEADER_PATH = os.path.join(os.getcwd(), 'Header')
# This will store all image names that are in HEADER_PATH folder
files = os.listdir(HEADER_PATH)
# Now, we will read each of the image and store it in a list
overlay_images = []
for image_name in files:
    image = cv2.imread(os.path.join(HEADER_PATH, image_name))
    overlay_images.append(image)

# This will set the default image to be displyed when program starts.
# In first image pink brush will be selected
header = overlay_images[0]
# Starting drawing color
drawing_color = (255, 0, 255)
xp, yp = 0, 0
brush_thickness = 15
eraser_thickness = 100
image_canvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)


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

    # Check if we got any points or not
    if len(landmarks) != 0:
        
        # We will use index and middle finger's tip
        # Tip of index finger
        ix, iy = landmarks[8][1:3]
        # TIp of middle finger
        mx, my = landmarks[12][1:3]

        # Checking which fingers are up
        fingers = detector.find_fingers_up()

        # If index and middle fingers are up, it means we're in selection mode & we wont draw anything
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            cv2.rectangle(frame, (ix, iy - 25), (mx, my + 25), drawing_color, -1)
            # Checking if index finger is in header area
            if iy < 125:
                # Checking on which icon index finger is
                if 250 < ix < 450:
                    # It means pink brush has to be selected, so we will change header image accordingly
                    header = overlay_images[0]
                    drawing_color = (255, 0, 255)
                elif 550 < ix < 750:
                    # It means blue brush has to be selected
                    header = overlay_images[1]
                    drawing_color = (255, 0, 0)
                elif 800 < ix < 950:
                    # It means green brush
                    header = overlay_images[2]
                    drawing_color = (0, 255, 0)
                elif 1050 < ix < 1200:
                    # It means eraser
                    header = overlay_images[3]
                    drawing_color = (0, 0, 0)

        # If only index finger is up, it means we're in drawing mode
        if fingers[1] and fingers[2] == False:
            cv2.circle(frame, (ix, iy), 15, drawing_color, -1)
            if xp == 0 and yp == 0:
                xp, yp = ix, iy
            
            if drawing_color == (0, 0, 0):
                cv2.line(frame, (xp, yp), (ix, iy), drawing_color, eraser_thickness)
                cv2.line(image_canvas, (xp, yp), (ix, iy), drawing_color, eraser_thickness)
            else:
                cv2.line(frame, (xp, yp), (ix, iy), drawing_color, brush_thickness)
                cv2.line(image_canvas, (xp, yp), (ix, iy), drawing_color, brush_thickness)
            xp, yp = ix, iy
    
    image_canvas_gray = cv2.cvtColor(image_canvas, cv2.COLOR_BGR2GRAY)
    _, image_canvas_gray_inv = cv2.threshold(image_canvas_gray, 50, 255, cv2.THRESH_BINARY_INV)
    image_canvas_gray_inv = cv2.cvtColor(image_canvas_gray_inv, cv2.COLOR_GRAY2BGR)

    frame = cv2.bitwise_and(frame, image_canvas_gray_inv)
    frame = cv2.bitwise_or(frame, image_canvas)

    # Adding header to the frame
    frame[0:125, 0:1280] = header

    # Frame rate calculation
    current_time = time()
    frame_rate = int(1 / (current_time - start_time))
    start_time = current_time
    cv2.putText(frame, str(frame_rate), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Displaying frame, if 'q' is pressed then exit the loop
    cv2.imshow('Live Feed', frame)
    cv2.imshow('Drawing', image_canvas)
    if cv2.waitKey(1) == ord('q'):
        break

# Releasing camera instance
cap.release()
# Destroying all windows
cv2.destroyAllWindows()