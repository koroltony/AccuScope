import cv2
import os
import numpy as np
import time
from auto_mask import create_mask
from matplotlib import pyplot as plt

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
            break

    return frames


def pano70(video, chunk_size=40):
    fps = video.get(cv2.CAP_PROP_FPS)
    num_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_offset = 0

    if not video.isOpened():
        print("Video could not be opened")

    # ---------- Create Mask From First Video Frame -----------------------
    # first, read the starting frame:
    frame_read, frame = video.read()

    # Create masks for the main image and minimap (lmask is main, smask is minimap)
    lmask, smask = create_mask(frame)

    # Reset the video frame grabber to start at frame 0
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    lhist_mask = cv2.calcHist([frame], [0], lmask, [256], [1, 256])

    # Show initial frame and histogram
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title("Normal Frame")
    plt.subplot(1, 2, 2)
    plt.plot(lhist_mask)
    plt.axhline(y=np.mean(lhist_mask), color='r', linestyle='-', linewidth=3)
    plt.title("Normal Histogram")

    # ----------------------------------------------------------------------

    while video.isOpened() and frame_offset <= num_frames:
        # Capture a chunk of frames
        frames = capture_frames(video, chunk_size)

        # Iterate over each frame in the chunk
        for i,frame in enumerate(frames):
            # Convert the frame to grayscale
            greyFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply the mask using bitwise and (lmask is for main, smask for minimap)
            lhist_mask = cv2.calcHist([greyFrame], [0], lmask, [256], [1, 256])
            shist_mask = cv2.calcHist([greyFrame], [0], smask, [256], [1, 256])

            # Calculate statistics for outliers in the main video
            lmean = np.mean(lhist_mask)
            lstd_dev = np.std(lhist_mask)

            if lstd_dev == 0:
                lz_scores = [0]
            else:
                lz_scores = (lhist_mask - lmean) / lstd_dev

            # Sensitivity seems to be around 5
            outlierThreshold = 5
            loutlier_bins = np.where(np.abs(lz_scores) > outlierThreshold)[0]
            lnumOutliers = loutlier_bins.size
            loutliersExist = (lnumOutliers > 0)

            # Calculate statistics for outliers in the minimap
            smean = np.mean(shist_mask)
            sstd_dev = np.std(shist_mask)

            if sstd_dev == 0:
                sz_scores = [0]
            else:
                sz_scores = (shist_mask - smean) / sstd_dev

            soutlier_bins = np.where(np.abs(sz_scores) > outlierThreshold)[0]
            snumOutliers = soutlier_bins.size
            soutliersExist = (snumOutliers > 0)

            # Check for errors based on outliers
            if (soutliersExist and np.all(soutlier_bins < 2)) or (loutliersExist and np.all(loutlier_bins < 2)):
                timeStamp = (video.get(cv2.CAP_PROP_POS_FRAMES)-(chunk_size - i)+1) / fps
                print('Pano to 70 at ', round(timeStamp, 2), 'seconds')

                # Show the individual frame and its histogram
                plt.figure()
                plt.subplot(1, 2, 1)
                plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                plt.title("Extracted Pano to 70 Glitch")
                plt.subplot(1, 2, 2)
                plt.plot(lhist_mask)
                plt.axhline(y=np.mean(lhist_mask), color='r', linestyle='-', linewidth=3)
                plt.title("Error Histogram")

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        frame_offset += chunk_size

    video.release()
    cv2.destroyAllWindows()

#Main Function with Test
if __name__=="__main__":
    video = cv2.VideoCapture('Pano to 70 glitch.mp4')
    pano70(video)
    print("--- %s seconds ---" % (time.time() - start_time))