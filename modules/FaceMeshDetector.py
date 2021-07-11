import cv2
import time
import mediapipe as mp

class FaceMeshDetector:
    def __init__(self, static_image_mode = False, max_num_faces = 1, min_detection_confidence = 0.5, min_tracking_confidence = 0.5):
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh_detector = self.mp_face_mesh.FaceMesh(self.static_image_mode, self.max_num_faces, self.min_detection_confidence, self.min_tracking_confidence)

    def find_face_mesh(self, frame, annote = True):
        frame = cv2.flip(frame, 1)
        frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        self.results = self.face_mesh_detector.process(frame_RGB)
        if self.results.multi_face_landmarks:
            if annote:
                for face_landmark in self.results.multi_face_landmarks:
                    self.mp_drawing.draw_landmarks(frame, face_landmark)
        return frame
    
    def get_landmark_locations(self, frame):
        landmarks = []
        height, width, channels = frame.shape

        if self.results.multi_face_landmarks:
            for face_landmark in self.results.multi_face_landmarks:
                lm_points = []
                for id, landmark in enumerate(face_landmark.landmark):
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    lm_points.append([id, x, y])
                landmarks.append(lm_points)
        return landmarks

def main():
    current_time = 0
    start_time = 0
    cap = cv2.VideoCapture(0)
    detector = FaceMeshDetector()

    while True:
        ret, frame = cap.read()
        frame = detector.find_face_mesh(frame)
        landmarks = detector.get_landmark_locations(frame)
        # print(landmarks)

        current_time = time.time()
        frame_rate = int(1 / (current_time - start_time))
        start_time = current_time

        cv2.putText(frame, str(frame_rate), (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)

        cv2.imshow('Live Feed', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()