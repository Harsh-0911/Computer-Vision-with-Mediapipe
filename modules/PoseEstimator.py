import cv2
import mediapipe as mp


class PoseEstimator:
    def __init__(self, static_image_mode=False, upper_body_only=False, smooth_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.upper_body_only = upper_body_only
        self.smooth_landmarks = smooth_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose_estimator = self.mp_pose.Pose(self.static_image_mode, self.upper_body_only,
                                                self.smooth_landmarks, self.min_detection_confidence, self.min_tracking_confidence)

    def find_pose(self, frame, annote=True):
        frame = cv2.flip(frame, 1)
        frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        self.result = self.pose_estimator.process(frame_RGB)

        if self.result.pose_landmarks:
            if annote:
                self.mp_drawing.draw_landmarks(
                    frame, self.result.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        return frame

    def get_landmarks(self, frame, annote = False):
        landmarks = []
        height, width, channels = frame.shape

        if self.result.pose_landmarks:
            for id, location in enumerate(self.result.pose_landmarks.landmark):
                x = int(width * location.x)
                y = int(height * location.y)
                landmarks.append([id, x, y])

                if annote:
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
        return landmarks


def main():
    cap = cv2.VideoCapture(0)
    estimtor = PoseEstimator()
    while True:
        ret, frame = cap.read()
        frame = estimtor.find_pose(frame)
        landmarks = estimtor.get_landmarks(frame)
        cv2.imshow('Live Feed', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
