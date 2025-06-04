import numpy as np
from numba import njit

def checkGreenFrame(frame):
    # Full-frame green condition
    Green_condition_full = np.sum((frame[:, :, 2] < 10) & (frame[:, :, 1] > 90) & (frame[:, :, 0] < 10)) > 200000

    # Top-left 80x80 region green condition
    patch = frame[:80, :80]
    Green_condition_patch = np.any((patch[:, :, 2] < 10) & (patch[:, :, 1] > 90) & (patch[:, :, 0] < 10))

    if Green_condition_full:
        return 1
    elif Green_condition_patch:
        return 2
    else:
        return 0
    
@njit
def checkGreenFrame_numba(frame):
    height, width = 480,640
    green_count = 0
    green_threshold = 200000
    patch_has_green = False

    for i in range(height):
        for j in range(width):
            r = frame[i, j, 2]
            g = frame[i, j, 1]
            b = frame[i, j, 0]
            if r < 10 and g > 90 and b < 10:
                green_count += 1

                if i < 80 and j < 80:
                    patch_has_green = True
                if green_count > green_threshold:
                    return 1

    if patch_has_green:
        return 2

    return 0