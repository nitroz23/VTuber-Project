import cv2 as cv
import numpy as np
from enum import Enum

class Eyes(Enum):
    LEFT = 0
    RIGHT = 1

class FacialFeatures:
    eyeKeyIndicies = [
        [
        # Left eye
        # eye lower contour
        33,
        7,
        163,
        144,
        145,
        153,
        154,
        155,
        133,
        # eye upper contour (excluding corners)
        246,
        161,
        160,
        159,
        158,
        157,
        173
        ],
        [
        # Right eye
        # eye lower contour
        263,
        249,
        390,
        373,
        374,
        380,
        381,
        382,
        362,
        # eye upper contour (excluding corners)
        466,
        388,
        387,
        386,
        385,
        384,
        398
        ]
    ]

    def resizeImg(img, scalePercent):
        width = int(img.shape[1] * scalePercent / 100.0)
        height = int(img.shape[0] * scalePercent / 100.0)

        return cv.resize(img, (width, height), interpolation = cv.INTER_AREA)
    
    def eyeAspectRatio(imagePoints, side):

        p1, p2, p3, p4, p5, p6 = 0, 0, 0, 0, 0, 0
        tipOfEyebrow = 0

        if side == Eyes.LEFT:

            EyeKeyLeft = FacialFeatures.eyeKeyIndicies[0]
            p1 = imagePoints[EyeKeyLeft[0]]

            p2 = np.true_divide(
                np.sum([imagePoints[EyeKeyLeft[10]], imagePoints[EyeKeyLeft[11]]], axis=0), 2)
            
            p3 = np.true_divide(
                np.sum([imagePoints[EyeKeyLeft[13]], imagePoints[EyeKeyLeft[14]]], axis=0),
                2)
            
            p4 = imagePoints[EyeKeyLeft[8]]

            p5 = np.true_divide(
                np.sum([imagePoints[EyeKeyLeft[5]], imagePoints[EyeKeyLeft[6]]], axis=0),
                2)

            p6 = np.true_divide(
                np.sum([imagePoints[EyeKeyLeft[2]], imagePoints[EyeKeyLeft[3]]], axis=0),
                2)
            
            tipOfEyebrow = imagePoints[105]

        elif side == Eyes.RIGHT:

            eyeKeyRight = FacialFeatures.eyeKeyIndicies[1]
            
            p1 = imagePoints[eyeKeyRight[8]]

            p2 = np.true_divide(
                np.sum([imagePoints[eyeKeyRight[13]], imagePoints[eyeKeyRight[14]]], axis=0), 
                2)
            
            p3 = np.true_divide(
                np.sum([imagePoints[eyeKeyRight[10]], imagePoints[eyeKeyRight[11]]], axis=0),
                2)
            
            p4 = imagePoints[eyeKeyRight[0]]

            p5 = np.true_divide(
                np.sum([imagePoints[eyeKeyRight[2]], imagePoints[eyeKeyRight[3]]], axis=0),
                2)

            p6 = np.true_divide(
                np.sum([imagePoints[eyeKeyRight[5]], imagePoints[eyeKeyRight[6]]], axis=0),
                2)

            tipOfEyebrow = imagePoints[334]
        
        ear = np.linalg.norm(p2-p6) + np.linalg.norm(p3-p5)
        ear /= (2 * np.linalg.norm(p1-p4) + 1e-6)
        ear = ear * (np.linalg.norm(tipOfEyebrow-imagePoints[2]) / np.linalg.norm(imagePoints[6]-imagePoints[2]))
        
        return ear   

    def mouthAspectRatio(imagePoints):
        p1 = imagePoints[78]
        p2 = imagePoints[81]
        p3 = imagePoints[13]
        p4 = imagePoints[311]
        p5 = imagePoints[308]
        p6 = imagePoints[402]
        p7 = imagePoints[14]
        p8 = imagePoints[178]

        mar = (np.linalg.norm(p2-p8) + np.linalg.norm(p3-p7 + np.linalg.norm(p4-p6)))
        mar /= (2 * np.linalg.norm(p1-p5) + 1e-6)

        return mar 
    
    def mouthDistance(imagePoints):
        p1 = imagePoints[78]
        p5 = imagePoints[308]

        return np.linalg.norm(p1-p5)
    
    def detectIris(imagePoints, irisImagePoints, side):

        irisImgPoint = -1
        p1, p4 = 0, 0
        eye_y_high, eye_y_low = 0, 0
        x_rate, y_rate = 0.5, 0.5

        if side == Eyes.LEFT:
            irisImgPoint = 468

            eyeKeyLeft = FacialFeatures.eyeKeyIndicies[0]
            p1 = imagePoints[eyeKeyLeft[0]]
            p4 = imagePoints[eyeKeyLeft[8]]

            eye_y_high = imagePoints[eyeKeyLeft[12]]
            eye_y_low = imagePoints[eyeKeyLeft[4]]

        elif side == Eyes.RIGHT:
            irisImgPoint = 473

            eyeKeyRight = FacialFeatures.eyeKeyIndicies[1]
            p1 = imagePoints[eyeKeyRight[8]]
            p4 = imagePoints[eyeKeyRight[0]]

            eye_y_high = imagePoints[eyeKeyRight[12]]
            eye_y_low = imagePoints[eyeKeyRight[4]]
        p_iris = irisImagePoints[irisImgPoint - 468]

        vecP1Iris = [p_iris[0] - p1[0], p_iris[1] - p1[1]]
        vec_p1_p4 = [p4[0] - p1[0], p4[1] - p1[1]]

        x_rate = (np.dot(vecP1Iris, vec_p1_p4) / (np.linalg.norm(p1-p4) + 1e-06)) / (np.linalg.norm(p1-p4) + 1e-06)

        vecEyeHighIris = [p_iris[0] - eye_y_high[0], p_iris[1] - eye_y_high[1]]
        vec_eye_high_low = [eye_y_low[0] - eye_y_high[0], eye_y_low[1] - eye_y_high[1]]

        y_rate = (np.dot(vec_eye_high_low, vecEyeHighIris) / (np.linalg.norm(eye_y_high - eye_y_low) + 1e-06)) / (np.linalg.norm(eye_y_high - eye_y_low) + 1e-06)

        return x_rate, y_rate