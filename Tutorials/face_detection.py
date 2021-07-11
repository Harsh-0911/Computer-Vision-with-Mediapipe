import time
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_detector.process(frame_RGB)
    if results.detections:
        for detection in results.detections:
            # mp_drawing.draw_detection(frame, detection)
            # print(dir(detection))
            height, width, channels = frame.shape
            bounding_box = detection.location_data.relative_bounding_box
            bx = int(bounding_box.xmin * width)        
            by = int(bounding_box.ymin * height)        
            bw = int(bounding_box.width * width)        
            bh = int(bounding_box.height * height)
            points = bx, by, bw, bh

            cv2.rectangle(frame, points, (255, 0, 255), 2)        
            cv2.rectangle(frame, (bx, by), (bw, bh), (0, 0, 255), 2)        

    cv2.imshow('Live Feed', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()