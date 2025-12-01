# body.py
import cv2
import mediapipe as mp
import numpy as np

class BodyDetector:
    def __init__(self,
                 min_detection_confidence=0.8,
                 min_tracking_confidence=0.5,
                 model_complexity=1,
                 enable_segmentation=True,
                 debug=False):
        self.debug = debug

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity,
            static_image_mode=False,
            enable_segmentation=enable_segmentation
        )

        self.mp_drawing = mp.solutions.drawing_utils
        self.landmarks = None  # Store latest landmarks

    def process_frame(self, frame):
        """
        Process a BGR frame, return frame with drawn landmarks
        and update self.landmarks
        """
        # Flip for mirror-like view
        img = cv2.flip(frame, 1)
        img.flags.writeable = False
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Pose detection
        results = self.pose.process(rgb_img)
        self.landmarks = results.pose_landmarks

        img.flags.writeable = True
        img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

        # Draw landmarks if available
        if self.landmarks and self.debug:
            self.mp_drawing.draw_landmarks(
                img,
                self.landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
            )

        return img

    def get_landmarks_array(self):
        """
        Return landmarks as a numpy array of shape (33, 3) for x, y, z
        """
        if self.landmarks is None:
            return None

        arr = np.zeros((33,3), dtype=np.float32)
        for i, lm in enumerate(self.landmarks.landmark):
            arr[i,0] = lm.x
            arr[i,1] = lm.y
            arr[i,2] = lm.z
        return arr
