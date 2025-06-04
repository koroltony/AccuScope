import time
import numpy as np
from numba import njit

start_time = time.time()

def checkMagentaFrame(frame, threshold_ratio=0.1):

    magenta_mask = (frame[:, :, 2] > 120) & (frame[:, :, 1] < 50) & (frame[:, :, 0] > 120)


    magenta_count = np.sum(magenta_mask)

    total_pixels = 640*480

    # If the proportion of magenta pixels is greater than a certain value:
    return 1 if magenta_count / total_pixels > threshold_ratio else 0

@njit
def checkMagentaFrame_numba(frame, threshold_ratio=0.1):
    count = 0
    total_pixels = 640*480
    threshold = total_pixels * threshold_ratio

    for i in range(480):
        for j in range(640):
            r = frame[i, j, 2]
            g = frame[i, j, 1]
            b = frame[i, j, 0]
            if r > 120 and g < 50 and b > 120:
                count += 1
                if count > threshold:
                    return 1
    return 0