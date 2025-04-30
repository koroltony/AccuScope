import cv2
import numpy as np
from numba import njit

def checkBlackFrame(frame,mask = None):

    # Create mask for footage
    if mask is not None:
        frame = cv2.bitwise_and(frame, frame, mask=mask)

    Black_condition = ~np.any((frame[:, :, 2] > 20) & (frame[:, :, 1] > 20) & (frame[:, :, 0] > 20))

    if Black_condition:
        return 1

    else:
        return 0
    
@njit
def checkBlackFrame_numba(frame, mask):
    height, width = frame.shape[:2]
    for i in range(height):
        for j in range(width):
            if mask[i, j] > 0:
                r = frame[i, j, 2]
                g = frame[i, j, 1]
                b = frame[i, j, 0]
                if r > 20 and g > 20 and b > 20:
                    return 0
    return 1
    
