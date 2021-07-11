import cv2
import time
import mediapipe as mp


class FaceDetector:
    def __init__(self, min_detection_confidence=0.5):
        self.min_detection_confidence = min_detection_confidence

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detector = self.mp_face_detection.FaceDetection(
            self.min_detection_confidence)

    def find_faces(self, frame, annote=True):
        frame = cv2.flip(frame, 1)
        frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.face_detector.process(frame_RGB)
        face_info = []

        if self.results.detections:
            if annote:
                for id, detection in enumerate(self.results.detections):
                    height, width, channels = frame.shape
                    bounding_box = detection.location_data.relative_bounding_box
                    bx = int(bounding_box.xmin * width)
                    by = int(bounding_box.ymin * height)
                    bw = int(bounding_box.width * width)
                    bh = int(bounding_box.height * height)
                    points = bx, by, bw, bh

                    face_info.append([points, detection.score])

                    # cv2.rectangle(frame, points, (255, 0, 255), 2)
                    cv2.putText(frame, str(int(
                        detection.score[0] * 100)), (bx, by - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
                    # cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
                    frame = self.draw_bounding_box(frame, points)
        return frame, face_info

    def draw_bounding_box(self, frame, box, length=20, corner_thickness=5, rect_thickness = 1):
        bx, by, bw, bh = box
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (255, 0, 255), rect_thickness)

        cv2.line(frame, (bx, by), (bx + length, by), (255, 0, 255), corner_thickness)
        cv2.line(frame, (bx, by), (bx, by + length), (255, 0, 255), corner_thickness)

        cv2.line(frame, (bx + bw, by), (bx + bw - length, by),
                 (255, 0, 255), corner_thickness)
        cv2.line(frame, (bx + bw, by), (bx + bw, by + length),
                 (255, 0, 255), corner_thickness)

        cv2.line(frame, (bx, by + bh), (bx + length,
                                        by + bh), (255, 0, 255), corner_thickness)
        cv2.line(frame, (bx, by + bh), (bx, by + bh - length),
                 (255, 0, 255), corner_thickness)

        cv2.line(frame, (bx + bw, by + bh), (bx + bw -
                                             length, by + bh), (255, 0, 255), corner_thickness)
        cv2.line(frame, (bx + bw, by + bh), (bx + bw, by +
                                             bh - length), (255, 0, 255), corner_thickness)

        return frame


def main():
    current_time = 0
    start_time = 0
    cap = cv2.VideoCapture(0)
    detector = FaceDetector()

    while True:
        ret, frame = cap.read()
        frame, result = detector.find_faces(frame)

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
