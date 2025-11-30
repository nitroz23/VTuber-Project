import cv2 as cv
import numpy as np

class poseEstimator:

    def __init__(self, imgSize=(480, 640)):
        self.size = imgSize
        self.modelPointsFull = self.getFullModelPoints()

        #camera internals
        self.focalLengths = self.size[1]
        self.cameraCenter = (self.size[1]/2, self.size[0] / 2)
        self.cameraMatrix = np.array(
            [[self.focalLengths, 0, self.cameraCenter[0]],
             [0, self.focalLengths, self.cameraCenter[1]],
             [0, 0, 1]], dtype = "double"
        )

        #assume no lens distortion
        self.distCoefs = np.zeros((4,1))

        #rotation vector and translation vector
        self.rVec = None
        self.tVec = None

    def getFullModelPoints(self, filename='model.txt'):
        """Get all 468 3D model points from file"""
        rawValue = []

        with open(filename) as file:
            for line in file:
                rawValue.append(line)

        modelPoints = np.array(rawValue, dtype=np.float32)
        modelPoints = np.reshape(modelPoints, (-1, 3))

        return modelPoints
    
    def solvePoseByAllPoints(self, imagePoints):
        """
        Solve pose from all the 468 image points
        Return (rotation_vector, translation_vector) as pose.
        """

        if self.rVec is None:
            (_, rotationVector, translationVector) = cv.solvePnP(
                self.modelPointsFull, imagePoints, self.cameraMatrix, self.distCoefs)
            self.rVec = rotationVector
            self.tVec = translationVector

        (_, rotationVector, translationVector) = cv.solvePnP(
            self.modelPointsFull,
            imagePoints,
            self.cameraMatrix,
            self.distCoefs,
            rvec=self.rVec,
            tvec=self.tVec,
            useExtrinsicGuess=True)
        
        return (rotationVector, translationVector)
    
    def drawAnnotationBox(self, image, rotationVector, translation_vector, color=(255, 255, 255), lineWidth=2):
        """Draw a 3D box as annotation of pose"""
        # Define the 3D box points.
        point_3d = []
        rearSize = 150
        rearDepth = 0
        frontSize = 100
        frontDepth = 400

        # rear box
        point_3d.append((-rearSize, -rearSize, rearDepth))
        point_3d.append((-rearSize, rearSize, rearDepth))
        point_3d.append((rearSize, rearSize, rearDepth))
        point_3d.append((rearSize, -rearSize, rearDepth))

        # front box
        point_3d.append((-frontSize, -frontSize, frontDepth))
        point_3d.append((-frontSize, frontSize, frontDepth))
        point_3d.append((frontSize, frontSize, frontDepth))
        point_3d.append((frontSize, -frontSize, frontDepth))
        point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

        # Map to 2D image points
        (point_2d, _) = cv.projectPoints(point_3d,
                                         rotationVector,
                                         translation_vector,
                                         self.cameraMatrix,
                                         self.distCoefs)
        point_2d = np.int32(point_2d.reshape(-1, 2))

        # Draw all the lines
        cv.polylines(image, [point_2d], True, color, lineWidth, cv.LINE_AA)
        cv.line(image, tuple(point_2d[1]), tuple(
            point_2d[6]), color, lineWidth, cv.LINE_AA)
        cv.line(image, tuple(point_2d[2]), tuple(
            point_2d[7]), color, lineWidth, cv.LINE_AA)
        cv.line(image, tuple(point_2d[3]), tuple(
            point_2d[8]), color, lineWidth, cv.LINE_AA)
        
    def drawAxis(self, img, R, t):
        axisLength = 20
        axis = np.float32(
            [[axisLength, 0, 0],
             [0, axisLength, 0],
             [0, 0, axisLength]]
        ).reshape(-1, 3)

        axisPoints, _ = cv.projectPoints(
            axis, R, t, self.cameraMatrix, self.distCoefs
        )

        img = cv.line(img, tuple(axisPoints[3].ravel()), tuple(
            axisPoints[0].ravel()), (255, 0, 0), 3)
        img = cv.line(img, tuple(axisPoints[3].ravel()), tuple(
            axisPoints[1].ravel()), (0, 255, 0), 3)
        img = cv.line(img, tuple(axisPoints[3].ravel()), tuple(
            axisPoints[2].ravel()), (0, 0, 255), 3)
        
    def drawAxes(self, img, R, t):
        img = cv.drawFrameAxes(img, self.cameraMatrix, self.distCoefs, R, t, 20)

    def reset_rVec_tVec(self):
        """Reset the rotation vector and translation vector"""
        self.rVec = None
        self.tVec = None