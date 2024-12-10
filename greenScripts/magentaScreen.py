import cv2
import os
import time
import numpy as np

start_time = time.time()

def checkMagentaFrame(frame):

    # Magenta Logical Mask:

    Magenta_condition = np.any((frame[:, :, 2] > 120) & (frame[:, :, 1] < 50) & (frame[:, :, 0] > 120))

    if Magenta_condition:
        return 1

    else:
        return 0