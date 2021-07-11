# For basic computer vision stuff
import cv2
# For advance computer vision stuff
import mediapipe as mp
 # Used for showing framerate
import time 

current_time = 0
start_time = 0

mp_drawings = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hand_detector = mp_hands.Hands()

cap = cv2.VideoCapture(0)

while True:
    # Reading camera frames
    ret, frame = cap.read()
    # Fliping frame horizontaly for selfie view
    frame = cv2.flip(frame, 1)
    # Mediapipe processes images in RGB color space
    frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Processing image
    results = hand_detector.process(frame_RGB)

    ### Inspecting result object
    # print(dir(results))
    # print('Index: ', results.index)
    # print('Count: ', results.count)
    # print('Multi Hand Landmarks: ', results.multi_hand_landmarks)
    # print('Multi Handedness: ', results.multi_handedness)

    # Checking whether or not we got any landmark
    if results.multi_hand_landmarks:
        # Drawing each landmark
        for id, landmark in enumerate(results.multi_hand_landmarks):
            classification = results.multi_handedness[id].classification
            print(str(classification[0]).strip().split('\n')[-1].split(':')[-1])
            # for s in classification:
            #     print(s)
            # print(landmark)
            # This only draws landmarks, not connections between the landmarks. for connections add 'mp_hands.HAND_CONNECTIONS'
            mp_drawings.draw_landmarks(frame, landmark, mp_hands.HAND_CONNECTIONS)
            
            for id, location in enumerate(landmark.landmark):
                # print('ID: ', id)
                # print('Location: ', location)
                # Getting original pixel values of location insted of value between 0 and 1
                height, width, channels = frame.shape
                cx, cy = int(width * location.x), int(height * location.y)
                # print(cx, cy)

    # Frame rate calculation
    current_time = time.time()
    frame_rate = int(1 / (current_time - start_time))
    start_time = current_time
    cv2.putText(frame, str(frame_rate), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)

    cv2.imshow('Live Feed', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()