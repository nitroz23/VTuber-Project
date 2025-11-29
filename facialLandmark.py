import cv2
import mediapipe as mp
import numpy as np

class FaceMeshDetector:
    def __init__(self,
                 staticImageMode=False,
                 maxNumFaces=1,
                 minDetectionConfidence=0.5,
                 minTrackingConfidence=0.5):

        self.staticImageMode = staticImageMode
        self.maxNumFaces = maxNumFaces
        self.minDetectionConfidence = minDetectionConfidence
        self.minTrackingConfidence = minTrackingConfidence

        # Initialize MediaPipe FaceMesh
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            static_image_mode=self.staticImageMode,
            max_num_faces=self.maxNumFaces,
            refine_landmarks=True,  # allows iris tracking
            min_detection_confidence=self.minDetectionConfidence,
            min_tracking_confidence=self.minTrackingConfidence
        )

        # Drawing utilities
        self.mpDrawing = mp.solutions.drawing_utils
        self.drawingSpec = self.mpDrawing.DrawingSpec(thickness=1, circle_radius=1)

    def findFaceMesh(self, img, draw=True):
        rgb_img = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)
        rgb_img.flags.writeable = False

        # Process face mesh
        self.results = self.faceMesh.process(rgb_img)

        rgb_img.flags.writeable = True
        img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

        self.imgH, self.imgW, _ = img.shape
        self.faces = []

        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:

                if draw:
                    self.mpDrawing.draw_landmarks(
                        image=img,
                        landmark_list=faceLms,
                        connections=self.mpFaceMesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=self.drawingSpec,
                        connection_drawing_spec=self.drawingSpec
                    )

                face = []
                for lm in faceLms.landmark:
                    x = int(lm.x * self.imgW)
                    y = int(lm.y * self.imgH)
                    face.append([x, y])

                self.faces.append(face)

        return img, self.faces


def main():
    detector = FaceMeshDetector()
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        img, faces = detector.findFaceMesh(img)
        cv2.imshow('MediaPipe FaceMesh', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
