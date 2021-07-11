import cv2
from math import hypot
import mediapipe as mp

class HandDetector:
    def __init__(self, static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_drawings = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hand_detector = self.mp_hands.Hands(
            self.static_image_mode, self.max_num_hands, self.min_detection_confidence, self.min_tracking_confidence)
        
        

    def find_hands(self, frame, annote = True):
        frame = cv2.flip(frame, 1)
        frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.result = self.hand_detector.process(frame_RGB)

        if self.result.multi_hand_landmarks:
            if annote:
                for landmark in self.result.multi_hand_landmarks:
                    self.mp_drawings.draw_landmarks(frame, landmark, self.mp_hands.HAND_CONNECTIONS)
        return frame

    def get_landmarks(self, frame, hand_index = 0, annote = False):
        self.landmarks = []
        height, width, channels = frame.shape

        if self.result.multi_hand_landmarks:
            classification = self.result.multi_handedness[hand_index].classification
            label = str(classification[0]).strip().split('\n')[-1].split(':')[-1]
            label = label.replace('"', '').strip()
            
            hand = self.result.multi_hand_landmarks[hand_index]
            for id, location in enumerate(hand.landmark):
                x = int(width * location.x)
                y = int(height * location.y)
                self.landmarks.append([id, x, y, label])

                if annote:
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        return self.landmarks
    
    def find_fingers_up(self):
        # This list will store which fingers are up -> [thumb, index, middle, ring, pinky]
        fingers = []

        # We will deal with thumb saperately
        # Ids of tip of the fingers will be stored. [index, middle, ring, pinky]
        tips_id = [8, 12, 16, 20]
        # Ids of pip of the fingers will be stored. [index, middle, ring, pinky]
        pips_id = [6, 10, 14, 18]

        if self.result.multi_hand_landmarks:
            # ==================== Dealing with thumb ====================
            # Logic we will use to decide whether the thumb is up or not is, if the x coordinate of the tip of the thumb is lower than the ip of the thumb, then thumb is said to be up. (ONLY RIGHT HAND)
            if self.landmarks[4][3] == "Right":
                if self.landmarks[4][1] < self.landmarks[3][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            elif self.landmarks[4][3] == "Left":
                if self.landmarks[4][1] > self.landmarks[3][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                pass

            # ==================== Dealing with fingers ====================
            # Logic we will use to decide whether the finger is up or not is, if the y coordinate of the tip of the finger is lower than the pip of the finger, then finger is said to be up.
            for tip, pip in zip(tips_id, pips_id):
                if self.landmarks[tip][2] < self.landmarks[pip][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

        return fingers

    def find_distance_between(self, frame, landmark1, landmark2, annote = False, circle_radius = 15, line_thickness = 3):
        x1, y1 = self.landmarks[landmark1][1:3]
        x2, y2 = self.landmarks[landmark2][1:3]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if annote:
            cv2.circle(frame, (x1, y1), circle_radius, (255, 0, 255), -1)
            cv2.circle(frame, (x2, y2), circle_radius, (255, 0, 255), -1)
            cv2.circle(frame, (cx, cy), circle_radius, (255, 0, 255), -1)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), line_thickness)
        
        length = hypot(x2 - x1, y2 - y1)

        return length, frame, [(x1, y1), (x2, y2), (cx. cy)]


def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:
        ret, frame = cap.read()
        frame = detector.find_hands(frame)
        landmarks = detector.get_landmarks(frame)
        if len(landmarks) != 0:
            print(landmarks[4])

        cv2.imshow('Live Feed', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()