import cv2
import numpy as np
from numba import njit

def checkBlackFrame(frame,mask):

    # Create mask for footage

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
    
@njit
def checkDropoutNoMask(frame):
    height, width, _ = frame.shape
    black_pixel_count = 0
    threshold = 30  # below this is considered "black"
    for i in range(height):
        for j in range(width):
            pixel = frame[i, j]
            if pixel[0] < threshold and pixel[1] < threshold and pixel[2] < threshold:
                black_pixel_count += 1
    
    total_pixels = height * width
    if black_pixel_count / total_pixels > 0.9:
        return True  # Dropout
<<<<<<< HEAD
    return False
=======
    return False
>>>>>>> 8ba35549fe810fd118bf3749bb11a73958b241fe
