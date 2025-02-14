import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import sys

# Get the root of the repository so that we can access all repo files
repo_root = subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).strip().decode()

# find helper scripts in our repository to get auto_mask
helper_scripts_dir = os.path.join(repo_root, 'Helper Scripts')

# Append the path to sys.path so Python can find auto_mask
sys.path.append(helper_scripts_dir)

# ----------- Function for Testing at Home (with Post-Process Footage) --------

def checkPanoEdge_test(frame, prev_frame, lmask, diff_pix_array):

    # Compute absolute difference within the mask
    diff = cv2.absdiff(frame, prev_frame)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    masked_diff = cv2.bitwise_and(gray_diff, gray_diff, mask=lmask)

    # Count nonzero pixels within the mask
    mask_size = np.count_nonzero(lmask)

    diff_pixels = np.exp(20*(np.count_nonzero(masked_diff > 10) / mask_size))
    diff_pix_array.append(diff_pixels)

    # Detect horizontal edges within the mask
    edges = cv2.Canny(frame, threshold1=2, threshold2=400)
    edges = cv2.bitwise_and(edges, edges, mask=lmask)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    min_edge_length = 100
    horizontal_edges = []

    for contour in contours:
        if cv2.arcLength(contour, closed=False) > min_edge_length:
            # Ensure edges are nearly horizontal (angle < 10 degrees)
            [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
            angle = np.arctan2(vy, vx) * (180 / np.pi)
            if abs(angle) < 10:
                horizontal_edges.append(contour)

    pano_detected = (len(horizontal_edges) > 1) and (diff_pixels > 0.3*10**(7))

    return pano_detected, edges


# -----------------------------------------------------------------------------

def checkPanoEdge(frame, prev_frame, lmask):

    # Compute absolute difference within the mask
    if prev_frame is not None:
        diff = cv2.absdiff(frame, prev_frame)
    else:
        diff = (np.zeros(np.shape(frame))).astype(np.uint8)

    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    masked_diff = cv2.bitwise_and(gray_diff, gray_diff, mask=lmask)

    # Count nonzero pixels within the mask
    mask_size = np.count_nonzero(lmask)

    diff_pixels = np.exp(20*(np.count_nonzero(masked_diff > 10) / mask_size))

    # Detect horizontal edges within the mask
    edges = cv2.Canny(frame, threshold1=2, threshold2=400)
    edges = cv2.bitwise_and(edges, edges, mask=lmask)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    min_edge_length = 100
    horizontal_edges = []

    for contour in contours:
        if cv2.arcLength(contour, closed=False) > min_edge_length:
            # Ensure edges are nearly horizontal (angle < 10 degrees)
            [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
            angle = np.arctan2(vy, vx) * (180 / np.pi)
            if abs(angle) < 10:
                horizontal_edges.append(contour)

    pano_detected = (len(horizontal_edges) > 1) and (diff_pixels > 0.3*10**(7))

    return pano_detected, edges


# ---------- Main Function -----------------------
if __name__ == "__main__":

    from auto_mask import create_mask

    diff_pix_array = []

    w, h = 1920,1080
    video = cv2.VideoCapture("C:/Users/korol/Documents/Arthrex Code/ece188a-arthrex/Consolidated - Real Time - Arthroscope/Raw_Videos/RawVideo168.mp4")
    #video = cv2.VideoCapture("C:/Users/korol/Documents/Arthrex Code/ece188a-arthrex/panoto70/Pano to 70 glitch.mp4")

    if not video.isOpened():
        print("Error: Could not open video.")
        exit()

    fps = video.get(cv2.CAP_PROP_FPS)
    time_interval = 1 / fps

    # Read the first frame
    frameRead, prev_frame = video.read()
    if not frameRead:
        print("Error: Could not read first frame.")
        exit()

    prev_frame = cv2.resize(prev_frame, (w, h))
    lmask = create_mask(prev_frame)  # Generate mask using the first frame

    while video.isOpened():
        currentFrame = video.get(cv2.CAP_PROP_POS_FRAMES)
        timeStamp = currentFrame / fps

        frameRead, frame = video.read()
        if not frameRead:
            break

        frame = cv2.resize(frame, (w, h))

        # Pass both current and previous frame to checkPanoEdge
        panoState, edges = checkPanoEdge_test(frame, prev_frame, lmask,diff_pix_array)

        if panoState:
            print(f'Non Minimap Pano-70 Error Found at: {round(timeStamp, 4)} seconds')

        prev_frame = frame

    video.release()

    plt.plot(diff_pix_array)
    plt.title('Average difference per frame')

