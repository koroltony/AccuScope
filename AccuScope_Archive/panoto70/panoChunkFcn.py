import cv2
import os
import numpy as np
import time
import subprocess
import sys
from HelperScripts.auto_mask import create_mask

# Get the root of the repository so that we can access all repo files
repo_root = subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).strip().decode()

# find helper scripts in our repository to get auto_mask
helper_scripts_dir = os.path.join(repo_root, 'Helper Scripts')

# Append the path to sys.path so Python can find auto_mask
sys.path.append(helper_scripts_dir)

from HelperScripts.auto_mask import create_mask


start_time = time.time()

def capture_frames(video, chunk_size=40):
    # Preallocate array to hold the frames
    frames = np.zeros((chunk_size, 1080, 1920, 3), dtype=np.uint8)

    # Count variable to keep track of frames opened in the chunk
    frame_count = 0

    # Check if Video Was Opened Successfully
    if not video.isOpened():
        print("Video could not be opened")

    # Save only the part of the video in the current chunk
    for i in range(chunk_size):
        # Read the frame
        frame_read, frame = video.read()

        # If there are frames left to read, put them in the frames array
        if frame_read:
            frame = cv2.resize(frame, (1920, 1080))  # Resize to 1080p
            frames[i] = frame
            frame_count += 1

        # If no more frames are left, pad the end with the last frame
        else:
            if frame_count > 0:
                frames[i:] = frames[frame_count - 1]
            else:
                return np.array([])
            break

    return frames


def chunkPano(chunk, smask, lmask, fps, currentFrame):
    # Check for an empty chunk
    if chunk.size == 0:
        return [], []

    minimap_timestamps = []
    main_timestamps = []

    # Get chunk size to calculate frame offset
    chunk_size = np.shape(chunk)[0]

    # Iterate over each frame in the chunk
    for i, frame in enumerate(chunk):
        # Convert the frame to grayscale
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply the mask using bitwise and (lmask is for main, smask for minimap)
        lhist_mask = cv2.calcHist([grayFrame], [0], lmask, [256], [1, 256])
        shist_mask = cv2.calcHist([grayFrame], [0], smask, [256], [1, 256])

        # Calculate statistics for outliers in the main video
        lmean = np.mean(lhist_mask)
        lstd_dev = np.std(lhist_mask)
        lz_scores = (lhist_mask - lmean) / (lstd_dev if lstd_dev != 0 else 1)
        loutlier_bins = np.where(np.abs(lz_scores) > 5)[0]
        loutliersExist = (loutlier_bins.size > 0)

        # Calculate statistics for outliers in the minimap
        smean = np.mean(shist_mask)
        sstd_dev = np.std(shist_mask)
        sz_scores = (shist_mask - smean) / (sstd_dev if sstd_dev != 0 else 1)
        soutlier_bins = np.where(np.abs(sz_scores) > 5)[0]
        soutliersExist = (soutlier_bins.size > 0)

        # Calculate the timestamp for the current frame
        frame_index = currentFrame - chunk_size + i + 1
        timeStamp = frame_index / fps

        # Collect errors for both minimap and main screen
        if loutliersExist and np.all(loutlier_bins < 2):
            main_timestamps.append(round(timeStamp, 4))
        if soutliersExist and np.all(soutlier_bins < 2):
            minimap_timestamps.append(round(timeStamp, 4))

    return minimap_timestamps, main_timestamps


# Main Function with Test
if __name__ == "__main__":
    # OpenCV Declaration
    video = cv2.VideoCapture("C:/Users/korol/Documents/Capstone TK/Pano to 70 glitch.mp4")
    totalFrames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = video.get(cv2.CAP_PROP_FPS)

    # ---------- Create Mask From First Video Frame -----------------------
    # First, read the starting frame
    frame_read, frame = video.read()
    frame = cv2.resize(frame, (1920, 1080))

    # Next, create masks for the main image and minimap: (lmask is main, smask is minimap)
    lmask, smask = create_mask(frame)

    # Reset the video frame grabber to start at frame 0
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    # ----------------------------------------------------------------------

    if not video.isOpened():
        print("Video could not be opened")
        exit()

    minimap_errors = []
    main_errors = []

    while video.isOpened():
        # Capture frames in chunks
        chunk = capture_frames(video, 60)
        currentFrame = video.get(cv2.CAP_PROP_POS_FRAMES)

        # Check Pano-70 for the current chunk
        minimap_timestamps, main_timestamps = chunkPano(chunk, smask, lmask, fps, currentFrame)

        # Collect the timestamps
        minimap_errors.extend(minimap_timestamps)
        main_errors.extend(main_timestamps)

        # Exit loop if no more chunks
        if chunk.size == 0:
            break

    # Print results
    print("\nMinimap Pano-70 Errors Found At (Seconds):", minimap_errors)
    print("\nMain Screen Pano-70 Errors Found At (Seconds):", main_errors)
    print("\n--- %s seconds ---" % (time.time() - start_time))