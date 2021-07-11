import time
import cv2
import mediapipe as mp

current_time = 0
start_time = 0
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose_detector.process(frame_RGB)
    height, width, channel = frame.shape
    if result.pose_landmarks:
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        for id, location in enumerate(result.pose_landmarks.landmark):
            x = int(width * location.x)
            y = int(height * location.y)

    current_time = time.time()
    frame_rate = int(1 / (current_time - start_time))
    start_time = current_time

    cv2.putText(frame, str(frame_rate), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)

    cv2.imshow('Live Feed', frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()