import cv2
import os
import numpy as np
import time
import subprocess
import sys

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


def chunkVectorSearch(chunk, smask, lmask):

    # Green Logical Mask:
    green_condition = (chunk[:, :, :, 1] > 90) & (chunk[:, :, :, 0] < 10) & (chunk[:, :, :, 2] < 10)

    green_mask_l = green_condition & lmask[None, :, :]
    green_mask_s = green_condition & smask[None, :, :]

    # Magenta Logical Mask:

    Magenta_condition = (chunk[:, :, :, 2] > 120) & (chunk[:, :, :, 1] < 50) & (chunk[:, :, :, 0] > 120)

    Magenta_mask_l = Magenta_condition & lmask[None, :, :]
    Magenta_mask_s = Magenta_condition & smask[None, :, :]

    # Dropout Logical Mask:

    Black_condition = (chunk[:, :, :, 1] > 20) | (chunk[:, :, :, 0] > 20) | (chunk[:, :, :, 2] > 20)

    # Now that we have the masks for each error type, we simply need to search the arrays for these errors

    gflagged_inds_l = np.where(np.any(green_mask_l, axis=(1, 2)))[0]
    gflagged_inds_s = np.where(np.any(green_mask_s, axis=(1, 2)))[0]

    mflagged_inds_l = np.where(np.any(Magenta_mask_l, axis=(1, 2)))[0]
    mflagged_inds_s = np.where(np.any(Magenta_mask_s, axis=(1, 2)))[0]

    bflagged_inds = np.where(~np.any(Black_condition, axis=(1, 2)))[0]

    # Calculate offsets for main video and minimap for all error types

    gOffsetsl = np.array([], dtype=int)
    mOffsetsl = np.array([], dtype=int)
    bOffsets = np.array([], dtype=int)

    gOffsetss = np.array([], dtype=int)
    mOffsetss = np.array([], dtype=int)

    # update list of indices for each error type and return

    # green list update:

    if gflagged_inds_l.size > 0:
        gOffsetsl = gflagged_inds_l - chunk.shape[0] + 1

    if gflagged_inds_s.size > 0:
        gOffsetss = gflagged_inds_s - chunk.shape[0] + 1

    # magenta list update:

    if mflagged_inds_l.size > 0:
        mOffsetsl = mflagged_inds_l - chunk.shape[0] + 1

    if mflagged_inds_s.size > 0:
        mOffsetss = mflagged_inds_s - chunk.shape[0] + 1

    # dropout list update:

    if bflagged_inds.size > 0:
        bOffsets = bflagged_inds - chunk.shape[0] + 1

    # return green indices first, then magenta then dropout

    return gOffsetsl, gOffsetss, mOffsetsl, mOffsetss, bOffsets

# Main Function with Test
if __name__ == "__main__":
    # OpenCV Declaration
    video = cv2.VideoCapture("C:/Users/korol/Documents/Capstone TK/Pano to 70 glitch.mp4")
    totalFrames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = video.get(cv2.CAP_PROP_FPS)

    # ---------- Create Mask From First Video Frame -----------------------

    # First, read the starting frame
    frame_read, frame = video.read()

    frame = cv2.resize(frame,(1920,1080))

    # Next, create masks for the main image and minimap: (lmask is main, smask is minimap)
    lmask, smask = create_mask(frame)

    # Reset the video frame grabber to start at frame 0
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # ----------------------------------------------------------------------

    if not video.isOpened():
        print("Video could not be opened")

    gmain_timestamps = []
    gminimap_timestamps = []

    mmain_timestamps = []
    mminimap_timestamps = []

    b_timestamps = []

    while video.isOpened():
        # Check Pano-70
        chunk = capture_frames(video, 40)

        # Ensure chunk is valid
        if chunk.size == 0:
            break

        # Get current frame position
        currentFrame = video.get(cv2.CAP_PROP_POS_FRAMES)

        # Get offsets for main video and minimap
        gOffsetsl, gOffsetss, mOffsetsl, mOffsetss, bOffsets = chunkVectorSearch(chunk, smask, lmask)

        # Calculate timestamps
        if gOffsetsl.size > 0:
            timeStampsl = (currentFrame + gOffsetsl) / fps
            gmain_timestamps.extend(timeStampsl)

        if gOffsetss.size > 0:
            timeStampss = (currentFrame + gOffsetss) / fps
            gminimap_timestamps.extend(timeStampss)

        if mOffsetsl.size > 0:
            timeStampsl = (currentFrame + mOffsetsl) / fps
            mmain_timestamps.extend(timeStampsl)

        if mOffsetss.size > 0:
            timeStampss = (currentFrame + mOffsetss) / fps
            mminimap_timestamps.extend(timeStampss)

        if bOffsets.size > 0:
            timeStamps = (currentFrame + bOffsets) / fps
            b_timestamps.extend(timeStamps)

    # Display results

    print("Green Main video errors (seconds):", [round(t, 4) for t in gmain_timestamps])
    print("Green Minimap errors (seconds):", [round(t, 4) for t in gminimap_timestamps])

    print("Magenta Main video errors (seconds):", [round(t, 4) for t in mmain_timestamps])
    print("Magenta Minimap errors (seconds):", [round(t, 4) for t in mminimap_timestamps])

    print("Dropout video errors (seconds):", [round(t, 4) for t in b_timestamps])

    print("--- %s seconds ---" % (time.time() - start_time))