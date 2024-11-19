import cv2
import os
import numpy as np
import time
import subprocess
import sys
# Get the root of the repository so that we can access all repo files
repo_root = subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).strip().decode()

# find helper scripts in our repository to get auto_mask
helper_scripts_dir = os.path.join(repo_root, 'HelperScripts')

# Append the path to sys.path so Python can find auto_mask
sys.path.append(helper_scripts_dir)

# Now you can import auto_mask
from auto_mask import create_mask

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


def chunkPano(chunk,smask,lmask):

        # at the end of the stream, we pass an empty np array, so we need to check
        # for that and make sure we still have chunks to go through

        if chunk.size == 0:
            return -1,-1

        resultFlag = 0
        timeOffset = 0

        # get chunk size to calculate frame offset
        chunk_size = np.shape(chunk)[0]

        # Iterate over each frame in the chunk
        for i,frame in enumerate(chunk):
            # Convert the frame to grayscale
            grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply the mask using bitwise and (lmask is for main, smask for minimap)
            lhist_mask = cv2.calcHist([grayFrame], [0], lmask, [256], [1, 256])
            shist_mask = cv2.calcHist([grayFrame], [0], smask, [256], [1, 256])

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

            # if there is an error in both minimap and main footage:

            if (soutliersExist and np.all(soutlier_bins < 2)) and (loutliersExist and np.all(loutlier_bins < 2)):

                # return 3 if there is an error in both

                # subtract timeOffset from cap_prop_pos to get the position of the error in this chunk
                timeOffset = -(chunk_size - i)+1

                resultFlag = 3

            # if there is an error in just the main image:

            if (loutliersExist and np.all(loutlier_bins < 2)):

                # return 1 if there is an error in large

                # subtract timeOffset from cap_prop_pos to get the position of the error in this chunk
                timeOffset = -(chunk_size - i)+1

                resultFlag = 1

            # if there is an error in just the minimap image:

            if (soutliersExist and np.all(soutlier_bins < 2)):

                # return 2 if there is an error in small

                # subtract timeOffset from cap_prop_pos to get the position of the error in this chunk
                timeOffset = -(chunk_size - i)+1

                resultFlag = 2

        return resultFlag,timeOffset


#Main Function with Test
if __name__=="__main__":
    #OpenCV Declaration
    video = cv2.VideoCapture("C:/Users/korol/Documents/Capstone TK/Pano to 70 glitch.mp4")
    totalFrames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = video.get(cv2.CAP_PROP_FPS)

    # ---------- Create Mask From First Video Frame -----------------------

    # first, read the starting frame:

    frame_read, frame = video.read()

    frame = cv2.resize(frame,(1920,1080))

    # Next, create masks for the main image and minimap: (lmask is main, smask is minimap)

    lmask,smask = create_mask(frame)

    # Reset the video frame grabber to start at frame 0

    video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    #----------------------------------------------------------------------

    if not video.isOpened():
        print("Video could not be opened")

    while video.isOpened():

         #Check Pano-70

        chunk = capture_frames(video,60)

        currentFrame = video.get(cv2.CAP_PROP_POS_FRAMES)

        panoState,indexOffset = chunkPano(chunk,smask,lmask)

        timeStamp = (currentFrame + indexOffset)/fps

        if(panoState == 1):
            print('Non Minimap Pano-70 Error Found at: ', round(timeStamp, 4), 'seconds')
        elif(panoState == 2):
            print('Minimap Pano-70 Error Found at: ', round(timeStamp, 4), 'seconds')
        elif(panoState == 3):
            print('Minimap and Main Screen Pano-70 Error Found at: ', round(timeStamp, 4), 'seconds')
        elif(panoState == -1):
            break

    print("--- %s seconds ---" % (time.time() - start_time))