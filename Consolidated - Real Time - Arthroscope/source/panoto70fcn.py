import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import sys
from scipy.signal import correlate, find_peaks

repo_root = subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).strip().decode()
helper_scripts_dir = os.path.join(repo_root, 'Helper Scripts')
sys.path.append(helper_scripts_dir)

repeated_region_array = []

# ----------------------------------------------------------------------------------------
def checkPanoEdge_test(frame, prev_frame, lmask, edge_array):

    # Equalize the frame before doing edge detection (Normalizes for brightness)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    equalized_frame = cv2.equalizeHist(gray_frame)
    
    edges = cv2.Canny(equalized_frame, threshold1=60, threshold2=300)
    edges = cv2.bitwise_and(edges, edges, mask=lmask)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    min_edge_length = 300
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
# Apply Autocorrelation to check for pano-to-70 repeated pixels

def analyze_autocorr(frame, pix_array, auto_corr_array, use_edge_correlation=False):
    
    # Tried doing autocorrelation on the canny-filtered frame (Very poor results)
    if use_edge_correlation:
        img = cv2.Canny(frame, threshold1=2, threshold2=200)
    else:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Extract the center 10 columns of pixels

    height, width = img.shape
    center = width // 2
    region_width = 10
    start_col = max(center - region_width // 2, 0)
    end_col = min(center + region_width // 2, width)

    # Average over the columns (also from 20:400 to remove irrelevant end pieces)
    region_avg = np.mean(img[20:400, start_col:end_col], axis=1)

    # Normalize by the standard deviation (makes smaller fluctuations more visible)
    mean_val = np.mean(region_avg)
    std_val = np.std(region_avg)
    if std_val == 0:
        std_val = 1
    norm_region = (region_avg - mean_val) / std_val
    pix_array.append(norm_region)

    # Compute Autocorrelation with the normalized column
    autocorr_full = correlate(norm_region, norm_region, mode='full')
    autocorr = autocorr_full[len(autocorr_full) // 2:]
    if np.max(autocorr) != 0:
        autocorr = autocorr / np.max(autocorr)

    # Look at the autocorrelation in the log domain, so that it is interpretable
    autocorr_log = np.log1p(np.abs(autocorr))
    auto_corr_array.append(autocorr_log)

    # Detect peaks in the autocorrelation signal
    peaks, properties = find_peaks(autocorr_log, prominence=0.3)
    # There is repetition if we see more than 3 peaks
    repeated_detected = (len(peaks) >= 3)
    repeated_region_array.append(repeated_detected)

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

    # Convert frame to grayscale and apply histogram equalization
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    equalized_frame = cv2.equalizeHist(gray_frame)
    
    edges = cv2.Canny(equalized_frame, threshold1=60, threshold2=300)
    edges = cv2.bitwise_and(edges, edges, mask=lmask)
    plt.figure(figsize=(10, 4))
    plt.imshow(edges)
    plt.title(f'Edge Frame {frame_idx} at {round(timeStamp, 2)}s')

    plt.figure(figsize=(10, 4))
    plt.imshow(frame)
    plt.title(f'Frame {frame_idx} at {round(timeStamp, 2)}s')

    plt.show()

# ----------------------------------------------------------------------------------------
def checkPanoEdge(frame, prev_frame, lmask):
    edges = cv2.Canny(frame, threshold1=60, threshold2=300)
    edges = cv2.bitwise_and(edges, edges, mask=lmask)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    min_edge_length = 300
    horizontal_edges = []
    for contour in contours:
        if cv2.arcLength(contour, closed=False) > min_edge_length:
            [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
            angle = np.arctan2(vy, vx) * (180 / np.pi)
            if abs(angle) < 10:
                horizontal_edges.append(1)
    pano_detected = (len(horizontal_edges) > 2)
    return pano_detected, edges

# ----------------------------------------------------------------------------------------
if __name__ == "__main__":
    from auto_mask import create_mask

    edge_array = []
    auto_corr_array = []
    pix_array = []

    video = cv2.VideoCapture("C:/Users/korol/Documents/Arthrex Code/ece188a-arthrex/Consolidated - Real Time - Arthroscope/source/Magenta480p.mp4")
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
        panoState_edge, edges = checkPanoEdge_test(frame, prev_frame, lmask, edge_array)

        # Combine detection metrics:
        # Use either the edge detection flag or the autocorrelation repeated-region flag.
        if panoState_edge or repeated_region_array[-1]:
            print(f"Pano-to-70 error at: {round(timeStamp, 4)} seconds (Frame {int(currentFrame)}) Edge: {panoState_edge} Repeated Region: {repeated_region_array[-1]}")
            plot_error_frame(frame, auto_corr_array[-1], pix_array[-1], int(currentFrame), timeStamp)

        # # plot detailed analysis for random frame.
        # if currentFrame in range(410,430):
        #      plot_error_frame(frame, auto_corr_array[-1], pix_array[-1], int(currentFrame), timeStamp)

        prev_frame = frame

    video.release()

    # Final plots for analysis over time.
    time_axis = np.arange(0, len(edge_array) / 60, 1/60)
    plot_autocorr(auto_corr_array)

    plt.figure()
    plt.plot(time_axis, edge_array)
    plt.title('Number of Valid Contours per Frame')

    plt.show()
