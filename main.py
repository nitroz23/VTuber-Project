from argparse import ArgumentParser
import cv2 as cv
import mediapipe as mp
import numpy as np

import socket

from facialLandmark import FaceMeshDetector
from poseEstimation import poseEstimator
from stabilizer import Stabilizer
from facialFeatures import FacialFeatures, Eyes

import sys

port = 5066

def initTCP():
    port = args.port

    address = ('127.0.0.1', port)

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(address)
        print("connected to address: ", socket.gethostbyname(socket.gethostname()) + ":" + str(port))
        return s
    except OSError as e:
        print("error while connecting :: %s" % e)
        sys.exit()

def sendInfoToUnity(s, args):
    msg = '%.4f ' * len(args) % args

    try:
        s.send(bytes(msg, "utf-8"))
    except socket.error as e:
        print("error while sending :: " + str(e))
        sys.exit()

def printDebugInfo(args):
    msg = '%.4f ' * len(args) % args
    print(msg)

def main():
    cap = cv.VideoCapture(args.cam)

    detector = FaceMeshDetector()
    success, img = cap.read()

    poseEstimation = poseEstimator((img.shape[0], img.shape[1]))
    imagePoints = np.zeros((poseEstimation.modelPointsFull.shape[0], 2))

    irisImagePoints = np.zeros((10, 2))

    poseStabilizers = [Stabilizer(
        stateNum=2,
        measureNum=1,
        covProcess=0.1,
        covMeasure=0.1) for _ in range(6)]
    
    eyeStabilizers = [Stabilizer(
        stateNum=2,
        measureNum=1,
        covProcess=0.1,
        covMeasure=0.1) for _ in range(6)]
    
    mouthDistStabilizer = Stabilizer(
        stateNum=2,
        measureNum=1,
        covProcess=0.1,
        covMeasure=0.1)
    
    if args.connect:
        socket = initTCP()

    while cap.isOpened():
        success, img = cap.read()

        if not success:
            print("ignoring empty camera frame.")
            continue

        imgFacemesh, faces = detector.findFaceMesh(img)

        img = cv.flip(img, 1)

        if faces:
            for i in range(len(imagePoints)):
                imagePoints[i, 0] = faces[0][i][0]
                imagePoints[i, 1] = faces[0][i][1]
            
            for j in range(len(irisImagePoints)):
                irisImagePoints[j, 0] = faces [0][j + 468][0]
                irisImagePoints[j, 1] = faces [0][j + 468][1]

            pose = poseEstimation.solvePoseByAllPoints(imagePoints)

            xRatioLeft, yRatioLeft = FacialFeatures.detectIris(imagePoints, irisImagePoints, Eyes.LEFT)
            xRatioRight, yRatioRight = FacialFeatures.detectIris(imagePoints, irisImagePoints, Eyes.RIGHT)

            earLeft = FacialFeatures. eyeAspectRatio(imagePoints, Eyes.LEFT)
            earRight = FacialFeatures.eyeAspectRatio(imagePoints, Eyes.RIGHT)

            poseEye = [earLeft, earRight, xRatioLeft, yRatioLeft, xRatioRight, yRatioRight]

            mar = FacialFeatures.mouthAspectRatio(imagePoints)
            mouthDist = FacialFeatures.mouthDistance(imagePoints)

            steadyPose = []
            pose_np = np.array(pose).flatten()

            for value, ps in zip(pose_np, poseStabilizers):
                ps.update([value])
                steadyPose.append(ps.state[0])

            steadyPose = np.reshape(steadyPose, (-1, 3))

            steadyPoseEye = []
            for value, es in zip(poseEye, eyeStabilizers):
                es.update([value])
                steadyPoseEye.append(es.state[0])

            mouthDistStabilizer.update([mouthDist])
            steadyMouthDist = mouthDistStabilizer.state[0]

            roll = np.clip(np.degrees(steadyPose[0][1]), -90, 90)
            pitch = np.clip(-(180 + np.degrees(steadyPose[0][0])), -90, 90)
            yaw = np.clip(np.degrees(steadyPose[0][2]), -90, 90)

            if args.connect:

                sendInfoToUnity(socket,
                    (roll, pitch, yaw,
                     earLeft, earRight, xRatioLeft, yRatioLeft, xRatioRight, yRatioRight,
                     mar, steadyMouthDist)
                )
            
            if args.debug:
                printDebugInfo(
                    (roll, pitch, yaw,
                     earLeft, earRight, xRatioLeft, yRatioLeft, xRatioRight, yRatioRight,
                     mar, steadyMouthDist)
                )
            
            poseEstimation.drawAxes(imgFacemesh, steadyPose[0], steadyPose[1])
        
        else:
            poseEstimation = poseEstimator((imgFacemesh.shape[0], imgFacemesh.shape[1]))

        cv.imshow('Facial Landmark', imgFacemesh)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--connect", action = "store_true",
                       help = "connect to unity character",
                       default = False)
    
    parser.add_argument("--port", type=int,
                        help = "specify the port of the connections to unity.",
                        default=5066)
    
    parser.add_argument("--cam", type=int,
                        help="specify the camera number if you have multiple cameras",
                        default=0)
    
    parser.add_argument("--debug", action="store_true",
                        help="showing raw values of detection in the terminal",
                        default=False)
    
    args = parser.parse_args()

    main()