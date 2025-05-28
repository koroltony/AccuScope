import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def general_detection(frame1, frame2,
                      slice_y1=200, slice_y2=400,
                      slice_width=160,
                      threshold=10,
                      bins=16,
                      sensitivity=0.2,
                      weight_pixel=0.2,
                      weight_hist=0.2,
                      weight_edges=0.4):
    """
    General anomaly detection using cropped region from frames,
    combining pixel difference, Bhattacharyya histogram distance,
    SSIM, and refined long edge difference (detects thin long lines).
    """
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    h, w = gray1.shape
    center = w // 2
    x1 = center - slice_width // 2
    x2 = center + slice_width // 2

    crop1 = gray1[slice_y1:slice_y2, x1:x2]
    crop2 = gray2[slice_y1:slice_y2, x1:x2]

    # Histogram (Bhattacharyya)
    hist1 = cv2.calcHist([crop1], [0], None, [bins], [0, 256])
    hist2 = cv2.calcHist([crop2], [0], None, [bins], [0, 256])
    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()
    hist_score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)

    # Pixel difference
    diff = np.abs(crop1.astype(np.int16) - crop2.astype(np.int16))
    pixel_score = np.count_nonzero(diff > threshold) / diff.size

    # --- Edge Detection with Long Thin Edge Filter ----

    edges1 = cv2.Canny(crop1, 50, 150)
    edges2 = cv2.Canny(crop2, 50, 150)

    edge_diff = np.count_nonzero(edges1 != edges2) / edges1.size

    # Final combined anomaly score
    score = (weight_pixel * pixel_score +
             weight_hist * hist_score +
             weight_edges * edge_diff)

    return score > sensitivity
