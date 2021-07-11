import cv2
import time
import mediapipe as mp

current_time = 0
start_time = 0
mp_drawing = mp.solutions.drawing_utils
drawing_specifications = mp_drawing.DrawingSpec(thickness=1, circle_radius=2)
mp_face_mesh = mp.solutions.face_mesh
face_mesh_detector = mp_face_mesh.FaceMesh()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh_detector.process(frame_RGB)

    if results.multi_face_landmarks:
        for face_landmark in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(frame, face_landmark, landmark_drawing_spec=drawing_specifications)

            height, width, channels = frame.shape
            for id, landmark in enumerate(face_landmark.landmark):
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                print(f'ID: {id}, X: {x}, Y: {y}')

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