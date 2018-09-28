from utils import BufferedWebcamStream
import numpy as np
import cv2
import logging
import time
import sys

# Number of frames to read before any multiprocessing
# It also sets the time taken to update any data
WINDOW_SIZE = 20

# I still don't know if I'll ever use this ¯\_(ツ)_/¯
WINDOW_OVERLAP = 0.2

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
                    format='[%(levelname)s] %(message)s')
log = logging.getLogger()

def color_distance(c1, c2):
    r = (c1[2] + c2[2])/2
    R = c1[2] - c2[2]
    G = c1[1] - c2[1]
    B = c1[0] - c2[0]
    return ((2 + r/256) * R**2 + 4 * G**2 + (2 + (255-r)/256)*B**2)**0.5

def average_radius(image, x, y, radius):
    return image[y-radius//2:y+radius//2, x-radius//2:x+radius//2].mean(axis=0)

def average_refcolor(image, xywh, ref_image, ref_color, dist_func, sensibility):
    x, y, w, h = xywh
    roi = ref_image[y:y+h, x:x+w]

    distances = dist_func(roi, ref_color)

    # true-false mapping for all distances
    bitmap = distances <= sensibility
    clone = roi.copy()
    clone[bitmap.nonzero()] = 255
    return roi[bitmap.nonzero()].sum() / bitmap.sum(), clone


if __name__ == '__main__':
    # Create stream object
    stream = BufferedWebcamStream(threaded=True)

    log.debug('Loading cascade classifier...')
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    log.debug('Dispach process!')
    stream.start()

    log.debug('Start iteration over frames.')
    for (pos, timestamp, frame) in stream:
        # try to find a face
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        log.info('Got {} faces.'.format(len(faces)))
        if len(faces) > 0:
            log.debug('Reading {} frames.'.format(WINDOW_SIZE))
            for _ in range(WINDOW_SIZE):
                (pos, timestamp, frame) = next(stream)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                for (x, y, w, h) in faces:
                    # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    # cv2.circle(frame, (int(x+h*.25), int(y+w*.6)), 3, (0, 255, 0), -1)
                    # cv2.circle(frame, (int(x+h*.75), int(y+w*.6)), 3, (0, 255, 0), -1)

                    # Average region around both left and right cheeks
                    avg1 = average_radius(gray, int(x+h*.25), int(y+w*.6), w//4)
                    avg2 = average_radius(gray, int(x+h*.75), int(y+w*.6), w//4)
                    avg = (avg1+avg2)/2
                    x, i = average_refcolor(frame, (x, y, w, h), gray, avg, lambda x, y: np.abs(x - y), 20)
                    cv2.imshow('frame',frame)
                    cv2.imshow('cut', i)
                    cv2.waitKey(1)
            log.debug('Done!')
            log.debug('Processing... [fake]')
        else:
            cv2.imshow('frame',frame)
            cv2.waitKey(1)
