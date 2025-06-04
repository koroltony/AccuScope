import cv2
import numpy as np
from numba import njit

@njit
def fast_pixel_diff(crop1, crop2, threshold):
    h, w = crop1.shape
    count = 0
    for i in range(h):
        for j in range(w):
            if abs(int(crop1[i, j]) - int(crop2[i, j])) > threshold:
                count += 1
    return count / (h * w)

def general_detection(frame1, frame2,
                      slice_y1=250, slice_y2=350,
                      slice_width=120,
                      threshold=10,
                      bins=16,
                      sensitivity=0.05,
                      weight_pixel=0.3,
                      weight_hist=0.7,
                      downsample=2):
    """
    Fast general anomaly detection using Numba for pixel & edge diff.
    Histogram kept standard via OpenCV for accuracy.
    """
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    h, w = gray1.shape
    center = w // 2
    x1 = center - slice_width // 2
    x2 = center + slice_width // 2

    crop1 = gray1[slice_y1:slice_y2, x1:x2]
    crop2 = gray2[slice_y1:slice_y2, x1:x2]

    if downsample > 1:
        crop1 = cv2.resize(crop1, (crop1.shape[1] // downsample, crop1.shape[0] // downsample))
        crop2 = cv2.resize(crop2, (crop2.shape[1] // downsample, crop2.shape[0] // downsample))

    # Histogram (OpenCV, unmodified)
    hist1 = cv2.calcHist([crop1], [0], None, [bins], [0, 256])
    hist2 = cv2.calcHist([crop2], [0], None, [bins], [0, 256])
    hist1 /= (np.sum(hist1) + 1e-6)
    hist2 /= (np.sum(hist2) + 1e-6)
    hist_score = 1 - cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    # Pixel difference (Numba)
    pixel_score = fast_pixel_diff(crop1, crop2, threshold)

    # Final score
    score = (weight_pixel * pixel_score +
             weight_hist * hist_score)
    if sensitivity != 0.5:
        return score > sensitivity
