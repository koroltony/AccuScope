import time
import numpy as np
from numba import njit

start_time = time.time()

def checkMagentaFrame(frame, threshold = 50):

    magenta_mask = (frame[:, :, 2] > 120) & (frame[:, :, 1] < 50) & (frame[:, :, 0] > 120)


    magenta_count = np.sum(magenta_mask)

    # If the proportion of magenta pixels is greater than a certain value:
    return 1 if magenta_count > threshold else 0

@njit
def checkMagentaFrame_numba(frame, threshold = 50):
    count = 0

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