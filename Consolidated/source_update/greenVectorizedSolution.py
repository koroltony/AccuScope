import cv2
import os
import time
import numpy as np
from numba import njit

start_time = time.time()

def checkGreenFrame(frame):

    # Green Logical Mask:

    Green_condition_full = np.sum((frame[:, :, 2] < 10) & (frame[:, :, 1] > 90) & (frame[:, :, 0] < 10))>60000
    Green_condition_part = np.any((frame[:, :, 2] < 10) & (frame[:, :, 1] > 90) & (frame[:, :, 0] < 10))

    if Green_condition_full:
        return 1

    elif Green_condition_part:
        return 2

    else:
        return 0
    
@njit
def checkGreenFrame_numba(frame):
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            r = frame[i, j, 2]
            g = frame[i, j, 1]
            b = frame[i, j, 0]
            if r < 10 and g > 90 and b < 10:
                return 1
    return 0