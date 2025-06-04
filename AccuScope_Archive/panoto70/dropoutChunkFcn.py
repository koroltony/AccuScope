import cv2
import os
import numpy as np
import time

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


def chunkBlack(chunk):

    # Apply masks with logical conditions
    Black_condition = (chunk[:, :, :, 1] > 20) | (chunk[:, :, :, 0] > 20) | (chunk[:, :, :, 2] > 20)

    # Find indices of flagged frames

    flagged_inds = np.where(~np.any(Black_condition, axis=(1, 2)))[0]

    # Calculate offsets for main video and minimap
    Offsets = np.array([], dtype=int)

    if flagged_inds.size > 0:
        Offsets = flagged_inds - chunk.shape[0] + 1

    return Offsets

# Main Function with Test
if __name__ == "__main__":
    # OpenCV Declaration
    video = cv2.VideoCapture("C:/Users/korol/Documents/Capstone TK/Pano to 70 glitch.mp4")
    #video = cv2.VideoCapture("C:/Users/korol/Documents/Capstone TK/green flash and lag.mp4")
    totalFrames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = video.get(cv2.CAP_PROP_FPS)

    if not video.isOpened():
        print("Video could not be opened")

    main_timestamps = []

    while video.isOpened():
        # Check Pano-70
        chunk = capture_frames(video, 40)

        # Ensure chunk is valid
        if chunk.size == 0:
            break

        # Get current frame position
        currentFrame = video.get(cv2.CAP_PROP_POS_FRAMES)

        # Get offsets for main video and minimap
        Offsets = chunkBlack(chunk)

        # Calculate timestamps
        if Offsets.size > 0:
            timeStamps = (currentFrame + Offsets) / fps
            main_timestamps.extend(timeStamps)

    # Display results
    print("Dropout errors (seconds):", [round(t, 4) for t in main_timestamps])

    print("--- %s seconds ---" % (time.time() - start_time))