import time
import cv2


class WebcamStream:
    ''' Stream for webcam frames

    TODO: Implement a singleton!
    '''

    def __init__(self):
        # open stream for webcam
        self.capture = cv2.VideoCapture(0)
        self.frame_number = 0
        self.frame = None
        self.timestamp = None

    def read(self):
        ''' Retrieve and return a new frame from hardware. '''
        ret, frame = self.capture.read()
        timestamp = time.time()
        if ret:
            self.frame = frame
            self.frame_number += 1
            self.timestamp = timestamp
        return (self.frame_number, self.timestamp, self.frame)

    def stop(self):
        ''' Release hardware resources. '''
        self.capture.release()
