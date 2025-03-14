import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import sys
from scipy.signal import correlate, find_peaks

# Set up paths (assumes you have a git repo structure)
repo_root = subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).strip().decode()
helper_scripts_dir = os.path.join(repo_root, 'Helper Scripts')
sys.path.append(helper_scripts_dir)

# Global list to store repeated region detection flag for each frame (autocorrelation based)
repeated_region_array = []

# ----------------------------------------------------------------------------------------
def checkPanoEdge_test(frame, prev_frame, lmask, diff_pix_array, edge_array):
    diff = cv2.absdiff(frame, prev_frame)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    masked_diff = cv2.bitwise_and(gray_diff, gray_diff, mask=lmask)

    mask_size = np.count_nonzero(lmask)
    diff_pixels = np.exp(20 * (np.count_nonzero(masked_diff > 10) / mask_size))
    diff_pix_array.append(diff_pixels)

    edges = cv2.Canny(frame, threshold1=2, threshold2=200)
    edges = cv2.bitwise_and(edges, edges, mask=lmask)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    min_edge_length = 100
    horizontal_edges = []

    for contour in contours:
        if cv2.arcLength(contour, closed=False) > min_edge_length:
            [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
            angle = np.arctan2(vy, vx) * (180 / np.pi)
            if abs(angle) < 10:
                horizontal_edges.append(1)

    edge_array.append(len(horizontal_edges))
    pano_detected = (len(horizontal_edges) > 2)
    return pano_detected, edges

# ----------------------------------------------------------------------------------------
# Improved autocorrelation analysis with z-score normalization and region averaging.

def analyze_autocorr(frame, pix_array, auto_corr_array, use_edge_correlation=False):
    # Use edge image if specified; otherwise use grayscale frame.
    if use_edge_correlation:
        img = cv2.Canny(frame, threshold1=2, threshold2=200)
    else:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    height, width = img.shape
    center = width // 2
    region_width = 10  # Averaging over 10 adjacent columns
    start_col = max(center - region_width // 2, 0)
    end_col = min(center + region_width // 2, width)

    # Average over the selected region to reduce noise.
    region_avg = np.mean(img[20:400, start_col:end_col], axis=1)

    # Apply z-score normalization.
    mean_val = np.mean(region_avg)
    std_val = np.std(region_avg)
    if std_val == 0:
        std_val = 1  # Prevent division by zero
    norm_region = (region_avg - mean_val) / std_val
    pix_array.append(norm_region)

    # Compute the full autocorrelation and take the second half.
    autocorr_full = correlate(norm_region, norm_region, mode='full')
    autocorr = autocorr_full[len(autocorr_full) // 2:]
    if np.max(autocorr) != 0:
        autocorr = autocorr / np.max(autocorr)

    # Log-transform the autocorrelation to compress the dynamic range.
    autocorr_log = np.log1p(np.abs(autocorr))
    auto_corr_array.append(autocorr_log)

    # Detect peaks in the autocorrelation signal.
    peaks, properties = find_peaks(autocorr_log, prominence=0.5)
    # Use a threshold (e.g., at least 3 peaks) to flag repeated patterns.
    repeated_detected = (len(peaks) >= 3)
    repeated_region_array.append(repeated_detected)
    # (Optionally, you can also print or log the detection result here.)

# ----------------------------------------------------------------------------------------
def plot_autocorr(auto_corr_array):
    auto_corr_array = np.array(auto_corr_array)
    plt.figure(figsize=(10, 5))
    plt.imshow(auto_corr_array.T, aspect='auto', cmap='magma',
               extent=[0, len(auto_corr_array), 0, auto_corr_array.shape[1]], origin='lower')
    plt.title('Autocorrelation Spectrogram')
    plt.xlabel('Frame Count')
    plt.ylabel('Lag')
    plt.colorbar(label='Autocorrelation Magnitude')
    plt.show()

def plot_error_frame(frame, auto_corr, pix_array, frame_idx, timeStamp):
    plt.figure(figsize=(10, 4))
    plt.plot(auto_corr)
    plt.title(f'Autocorrelation for Frame {frame_idx} at {round(timeStamp, 2)}s')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation Magnitude')
    plt.grid(True)

    plt.figure(figsize=(10, 4))
    plt.plot(pix_array)
    plt.title(f'Normalized Pixel Array for Frame {frame_idx} at {round(timeStamp, 2)}s')
    plt.xlabel('Pixel')
    plt.ylabel('Normalized Magnitude')
    plt.grid(True)

    plt.figure(figsize=(10, 4))
    plt.imshow(frame)
    plt.title(f'Frame {frame_idx} at {round(timeStamp, 2)}s')

    edges = cv2.Canny(frame, threshold1=2, threshold2=200)
    plt.figure(figsize=(10, 4))
    plt.imshow(edges)
    plt.title(f'Edge Frame {frame_idx} at {round(timeStamp, 2)}s')

    plt.show()

# ----------------------------------------------------------------------------------------
def checkPanoEdge(frame, prev_frame, lmask):
    if prev_frame is not None:
        diff = cv2.absdiff(frame, prev_frame)
    else:
        diff = np.zeros(frame.shape, dtype=np.uint8)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    masked_diff = cv2.bitwise_and(gray_diff, gray_diff, mask=lmask)
    mask_size = np.count_nonzero(lmask)
    diff_pixels = np.exp(20 * (np.count_nonzero(masked_diff > 10) / mask_size))
    edges = cv2.Canny(frame, threshold1=2, threshold2=200)
    edges = cv2.bitwise_and(edges, edges, mask=lmask)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    min_edge_length = 100
    horizontal_edges = []
    for contour in contours:
        if cv2.arcLength(contour, closed=False) > min_edge_length:
            [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
            angle = np.arctan2(vy, vx) * (180 / np.pi)
            if abs(angle) < 10:
                horizontal_edges.append(1)
    pano_detected = (len(horizontal_edges) > 2) #and (diff_pixels > 0.3*10**(7))
    return pano_detected, edges

# ----------------------------------------------------------------------------------------
if __name__ == "__main__":
    from auto_mask import create_mask

    diff_pix_array = []
    edge_array = []
    auto_corr_array = []
    pix_array = []

    video = cv2.VideoCapture("C:/Users/korol/Documents/Arthrex Code/ece188a-arthrex/Consolidated - Real Time - Arthroscope/Raw_Videos/RawVideo194.mp4")
    if not video.isOpened():
        print("Error: Could not open video.")
        exit()

    fps = video.get(cv2.CAP_PROP_FPS)
    time_interval = 1 / fps

    frameRead, prev_frame = video.read()
    if not frameRead:
        print("Error: Could not read first frame.")
        exit()

    lmask = create_mask(prev_frame)
    kernel = np.ones((3, 3), np.uint8)
    for _ in range(20):
        lmask = cv2.erode(lmask, kernel)

    overlayed_frame = prev_frame.copy()
    overlayed_frame[lmask > 0] = cv2.add(overlayed_frame[lmask > 0], (255, 255, 255, 0))
    overlayed_frame = cv2.cvtColor(overlayed_frame, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(overlayed_frame)
    plt.title('First Frame with Mask Overlay')
    plt.axis('off')

    while video.isOpened():
        currentFrame = video.get(cv2.CAP_PROP_POS_FRAMES)
        timeStamp = currentFrame / fps

        frameRead, frame = video.read()
        if not frameRead:
            break

        # Improved autocorrelation analysis using the updated function.
        analyze_autocorr(frame, pix_array, auto_corr_array)

        # Edge-based detection as before.
        panoState_edge, edges = checkPanoEdge_test(frame, prev_frame, lmask, diff_pix_array, edge_array)

        # Combine detection metrics:
        # Use either the edge detection flag or the autocorrelation repeated-region flag.
        if panoState_edge or repeated_region_array[-1]:
            print(f"Pano-to-70 error at: {round(timeStamp, 4)} seconds (Frame {int(currentFrame)})")
            plot_error_frame(frame, auto_corr_array[-1], pix_array[-1], int(currentFrame), timeStamp)

        # Optionally, plot detailed analysis for specific frames.
        if currentFrame in [2415, 500]:
            plot_error_frame(frame, auto_corr_array[-1], pix_array[-1], int(currentFrame), timeStamp)

        prev_frame = frame

    video.release()

    # Final plots for analysis over time.
    time_axis = np.arange(0, len(diff_pix_array) / 60, 1/60)
    plot_autocorr(auto_corr_array)

    plt.figure()
    plt.plot(time_axis, diff_pix_array)
    plt.title('Average Difference per Frame')

    plt.figure()
    plt.plot(time_axis, edge_array)
    plt.title('Number of Valid Contours per Frame')

    plt.show()
