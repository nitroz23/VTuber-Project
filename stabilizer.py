import numpy as np
import cv2 as cv

class Stabilizer:
    """Using Kalman filter as a point stabilizer."""

    def __init__(self,
                 stateNum = 4,
                 measureNum = 2,
                 covProcess = 0.0001,
                 covMeasure = 0.1):
        """Initialization"""
        assert stateNum == 4 or stateNum ==2, "Only scalar and point supported, Check stateNum please."

        self.stateNum = stateNum
        self.measureNum = measureNum

        self.filter = cv.KalmanFilter(stateNum, measureNum, 0)
        self.state = np.zeros((stateNum, 1), dtype=np.float32)
        self. measurement = np.array((measureNum, 1), np.float32)
        self.prediction = np.zeros((stateNum, 1), np.float32)

        if self.measureNum == 1:
            self.filter.transitionMatrix = np.array([[1, 1],
                                                     [0, 1]], np.float32)
            
            self.filter.measurementMatrix = np.array([[1, 1]], np.float32)

            self.filter.processNoiseCov = np.array([[1, 0],
                                                    [0, 1]], np.float32) * covProcess
            
            self.filter.measurementNoiseCov = np.array(
                [[1]], np.float32) * covMeasure
            
        if self.measureNum == 2:
            self.filter.transitionMatrix = np.array([[1, 0, 1, 0],
                                                     [0, 1, 0, 1],
                                                     [0, 0, 1, 0],
                                                     [0, 0, 0, 1]], np.float32)
            
            self.filter.measurementMatrix = np.array([[1, 0, 0, 0],
                                                      [0, 1, 0, 0]], np.float32)
            
            self.filter.processNoiseCov = np.array([[1, 0, 0, 0],
                                                    [0, 1, 0, 0],
                                                    [0, 0, 1, 0],
                                                    [0, 0, 0, 1]], np.float32) * covProcess
            
            self.filter.measurementNoiseCov = np.array([[1, 0],
                                                        [0, 1]], np.float32) * covMeasure
        
    def update(self, measurement):
        """update the filter"""

        self.prediction = self.filter.predict()

        if self.measureNum == 1:
            self.measurement = np.array([[np.float32(measurement[0])]])
        else:
            self.measurement = np.array([[np.float32(measurement[0])],
                                            [np.float32(measurement[1])]])
            
        self.filter.correct(self.measurement)

        self.state = self.filter.statePost
    
    def set_q_r(self, covProcess = 0.1, covMeasure = 0.001):
        """set new value for processNoiseCov and measurementNoiseCov"""

        if self.measureNum == 1:
            self.filter.processNoiseCov = np.array([[1, 0],
                                                    [0, 1]], np.float32) * covProcess
            self.filter.measurementNoiseCov = np.array(
                [[1]], np.float32) * covMeasure
        else:
            self.filter.processNoiseCov = np.array([[1, 0, 0, 0],
                                                    [0, 1, 0, 0],
                                                    [0, 0, 1, 0],
                                                    [0, 0, 0, 1]], np.float32) * covProcess
            self.filter.measurementNoiseCov = np.array([[1, 0],
                                                        [0, 1]], np.float32) * covMeasure

def main():
    """test code"""
    global mp
    mp = np.array((2, 1), np.float32)

    def onmouse(k, x, y, s, p):
        global mp
        mp = np.array([[np.float32(x)], [np.float32(y)]])

    cv.namedWindow("kalman")
    cv.setMouseCallback("kalman", onmouse)
    kalman = Stabilizer(4, 2)
    frame = np.zeros((480, 640, 3), np.uint8)

    while True:
        kalman.update(mp)
        point = kalman.prediction
        state = kalman.filter.statePost
        cv.circle(frame, (state[0], state[1]), 2, (255, 0, 0), -1)
        cv.circle(frame, (point[0], point[1]), 2, (0, 255, 0), -1)
        cv.imshow("kalman", frame)
        k = cv.waitKey(30) & 0xFF
        if k == 27:
            break

if __name__ == "__main__":
    main()