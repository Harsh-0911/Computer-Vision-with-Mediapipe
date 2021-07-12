# ==================== Importing Libraries ====================

# Classic computer vision module
import cv2
# Library by google, used for detecting hands and hand landmarks
import mediapipe as mp
# For reading json files
import json
# Classic library for automation. Here we'll specifically use chrome browser
from selenium import webdriver
# For setting options for webdriver
from selenium.webdriver.chrome.options import Options
# Used for pressing keyboard keys. Here specifically 'space' key to avoid obstacles in game
from pynput.keyboard import Key, Controller
# This is wrriten by me on top of 'mediapipe', Makes our work easy
from modules import HandDetector as hd

# ==================== Reading json file ====================
# Opening json file
config = open('config.json')

# Reading json file
data = json.load(config)


# ==================== Variable Declaration ====================

# Driver path used for selenium's driver initialization. Download appropriate driver for your browser and add it to the config.json file
DRIVER_PATH = data['path']
# URL of the Website (game in our case)
WEBSITE_URL = "https://chromedino.com/"
# Frame height and width
FRAME_WIDTH = 640
FRAME_HEIGHT = 480


# ==================== Object Initialization ====================

# For setting options for browser window
options = Options()
# This will force the browser window to be open in maximized form
options.add_argument('start-maximized')
# Driver initialization
driver = webdriver.Chrome(executable_path=DRIVER_PATH, options=options)
# Initializing camera instance for accessing camera
cap = cv2.VideoCapture(0)
# Initializing Hand Detector object. max_num_hands is 1 so that it will detect only 1 hand from multiple hands availabe in frame.
detector = hd.HandDetector(max_num_hands=1)
# Initializing the controller of 'pynput' to access the keyboard
keyboard = Controller()


# ==================== Setting Environment ====================

# Setting width of the frame
cap.set(3, FRAME_WIDTH)
# Setting height of the frame
cap.set(4, FRAME_HEIGHT)
# Setting brightness of the frame
cap.set(10, 150)
# Loading the website in browser
driver.get(WEBSITE_URL)


# ==================== Main Execution ====================

# Infinite loop, press 'q' to quit
while True:
    # Reading camera
    ret, frame = cap.read()
    # finding hands in frame and drawing landmarks on it, if you don't want to draw then make 'annote' to False
    frame = detector.find_hands(frame, annote=True)
    # Finding landmarks of hand, it is neccessary in order to find which finger is up and which is down.
    landmarks = detector.get_landmarks(frame)
    # Finding which finger is up and which is down.
    # Order of this list is [thumb, index, middle, ring, pinky]
    # IF finger is up then corrosponding index value will be 1, otherwise it is 0
    fingers = detector.find_fingers_up()
    # You can print it to check whether it identidfies it correctly or not
    # print(fingers)

    # We will press 'space' key when index finger is up.
    # Checking whether there is values in 'fingers' list, because it may be empty if there is no hand detected
    if fingers != []:
        # Checking only index finger, we do not want to do anything if any other finger or thumb is up with index finger
        if fingers[1] == 1 and sum(fingers) == 1:
            # Pressing space key
            keyboard.press(Key.space)
            # As library is created in such a way that we explicitly have to release the key in order to get that 'one time click' effect
            keyboard.release(Key.space)

    # Displaying the frame
    cv2.imshow('Live Feed', frame)
    # If 'q' is pressed then break out of the loop
    if cv2.waitKey(10) == ord('q'):
        break

# Releasing camera, so that other processes can use the resource
cap.release()
# Destroying all windows created in program
cv2.destroyAllWindows()
