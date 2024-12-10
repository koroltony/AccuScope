import cv2
import os
import time
import numpy as np

start_time = time.time()

def checkBlackFrame(frame):

    # Magenta Logical Mask:

    Black_condition = ~np.any((frame[:, :, 2] > 20) & (frame[:, :, 1] > 20) & (frame[:, :, 0] > 20))

    if Black_condition:
        return 1

    else:
        return 0