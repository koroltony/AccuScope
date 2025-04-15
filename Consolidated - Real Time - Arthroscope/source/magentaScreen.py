import cv2
import os
import time
import numpy as np
from numba import njit

start_time = time.time()

def checkMagentaFrame(frame):

    # Magenta Logical Mask:

    Magenta_condition = np.any((frame[:, :, 2] > 120) & (frame[:, :, 1] < 50) & (frame[:, :, 0] > 120))

    if Magenta_condition:
        return 1

    else:
        return 0
    
@njit
def checkMagentaFrame_numba(frame):
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            r = frame[i, j, 2]
            g = frame[i, j, 1]
            b = frame[i, j, 0]
            if r > 120 and g < 50 and b > 120:
                return 1
    return 0