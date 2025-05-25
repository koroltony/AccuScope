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
                      weight_ssim=0.2,
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

    # SSIM
    ssim_score = 1 - ssim(crop1, crop2)

    # --- Edge Detection with Long Thin Edge Filter ---
    blur1 = cv2.GaussianBlur(crop1, (3, 3), 1.0)
    blur2 = cv2.GaussianBlur(crop2, (3, 3), 1.0)

    edges1 = cv2.Canny(blur1, 50, 150)
    edges2 = cv2.Canny(blur2, 50, 150)

    def extract_significant_edges(edge_img, min_len=80, min_width=2):
        contours, _ = cv2.findContours(edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(edge_img)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            length = cv2.arcLength(cnt, closed=False)
            if (length >= min_len) and (w >= min_width or h >= min_width):
                cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)
        return mask

    mask1 = extract_significant_edges(edges1, min_len=80)
    mask2 = extract_significant_edges(edges2, min_len=80)

    edge_diff = np.count_nonzero(mask1 != mask2) / mask1.size

    # Final combined anomaly score
    score = (weight_pixel * pixel_score +
             weight_hist * hist_score +
             weight_ssim * ssim_score +
             weight_edges * edge_diff)

    return score > sensitivity
