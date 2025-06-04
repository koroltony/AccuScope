import cv2
import os
import time
import numpy as np

start_time = time.time()

def checkGreenFrame(frame):

    # Green Logical Mask:

    Green_condition_full = np.sum((frame[:, :, 2] < 10) & (frame[:, :, 1] > 90) & (frame[:, :, 0] < 10))>60000
    Green_condition_mini = np.any((frame[:, :, 2] < 10) & (frame[:, :, 1] > 90) & (frame[:, :, 0] < 10))

    if Green_condition_full:
        return 1

    elif Green_condition_mini:
        return 2

    else:
        return 0