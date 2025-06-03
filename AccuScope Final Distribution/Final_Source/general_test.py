import cv2
import numpy as np

def detect_anomalies(frame1, frame2, diff_thresh=1, sudden_change_ratio=0.9, line_thresh=1e7, pixelation_thresh = 2500):
    '''
    # --- 1. Sudden Frame Change Detection ---
    diff = cv2.absdiff(frame1, frame2)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    diff_pixels = np.count_nonzero(gray_diff > diff_thresh)
    total_pixels = frame1.shape[0] * frame1.shape[1]
    #percent_changed = (diff_pixels / total_pixels) * 100
    #print(f"Different pixels: {diff_pixels} / {total_pixels} ({percent_changed:.2f}%)")

    sudden_change = diff_pixels > (sudden_change_ratio * total_pixels)
    '''

    # --- 2. Line Artifact Detection ---
    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    line_intensity = np.sum(np.abs(sobelx)) + np.sum(np.abs(sobely))
    #print(f"Line intensity: {line_intensity:.2e}")

    line_artifacts = line_intensity > line_thresh

    '''
    # --- 3. Pixelation Detection ---
    edges1 = cv2.Canny(cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY), 100, 200)
    edges2 = cv2.Canny(cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY), 100, 200)

    edge_sum1 = np.sum(edges1 > 0)
    edge_sum2 = np.sum(edges2 > 0)

    edge_diff = edge_sum1 - edge_sum2
    #print(f"Edge sum diff: {edge_sum1} -> {edge_sum2} (Δ = {edge_diff})")

    pixelation = edge_diff > pixelation_thresh
    '''

    # Final Decision
    return line_artifacts
    #return sudden_change or line_artifacts or pixelation 
