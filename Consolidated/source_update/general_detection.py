import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def general_detection(frame1, frame2,
                      slice_y1=200, slice_y2=400,
                      slice_width=160,
                      threshold=10,
                      bins=16,
                      sensitivity=0.2,
                      weight_pixel=0,
                      weight_hist=1,
                      weight_ssim=0):
    """
    General anomaly detection using cropped region from frames,
    combining pixel difference, histogram difference, and SSIM.
    """
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    h, w = gray1.shape
    center = w // 2
    x1 = center - slice_width // 2
    x2 = center + slice_width // 2

    crop1 = gray1[slice_y1:slice_y2, x1:x2]
    crop2 = gray2[slice_y1:slice_y2, x1:x2]

    # Histogram
    hist1 = np.histogram(crop1, bins=bins, range=(0, 256))[0]
    hist2 = np.histogram(crop2, bins=bins, range=(0, 256))[0]
    hist1 = hist1 / np.sum(hist1)
    hist2 = hist2 / np.sum(hist2)
    hist_score = 1 - np.dot(hist1, hist2) / (np.linalg.norm(hist1) * np.linalg.norm(hist2) + 1e-6)

    # Pixel diff
    diff = np.abs(crop1.astype(np.int16) - crop2.astype(np.int16))
    pixel_score = np.count_nonzero(diff > threshold) / diff.size

    # SSIM (returns similarity, so we invert it)
    ssim_score = 1 - ssim(crop1, crop2)

    # Combined anomaly score
    score = (weight_pixel * pixel_score +
             weight_hist * hist_score +
             weight_ssim * ssim_score)

    return score > sensitivity
