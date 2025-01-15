import cv2
import os
import numpy as np

import subprocess
import sys

# Get the root of the repository so that we can access all repo files
repo_root = subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).strip().decode()

# find helper scripts in our repository to get auto_mask
helper_scripts_dir = os.path.join(repo_root, 'Helper Scripts')

# Append the path to sys.path so Python can find auto_mask
sys.path.append(helper_scripts_dir)

def checkPano(frame,smask,lmask):

    resultFlag = 0

    # change to grayscale for analysis

    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Apply the mask using a bitwise and of the frame on itself with the mask we just defined.
    #lmasked_grayFrame = cv2.bitwise_and(grayFrame, grayFrame, mask=lmask)
    #smasked_grayFrame = cv2.bitwise_and(grayFrame, grayFrame, mask=smask)

    # Calculate the histogram with and without the mask
    #hist_full = cv2.calcHist([grayFrame], [0], None, [256], [1, 256])

    # Calculate separate histograms for the minimap and the main image:

    lhist_mask = cv2.calcHist([grayFrame], [0], lmask, [256], [1, 256])
    shist_mask = cv2.calcHist([grayFrame], [0], smask, [256], [1, 256])

    #Calculate Statistics for Outliers in main video

    #print(f'frame: {frameRead}')
    lmean = np.mean(lhist_mask)
    #print(f'mean: {lmean}')
    lstd_dev = np.std(lhist_mask)
    #print(f'stand dev: {lstd_dev}')

    if lstd_dev == 0:
        lz_scores = [0]

    else:
        lz_scores = (lhist_mask - lmean) / lstd_dev

    #print(f'zscore: {lz_scores}')

    #Sensitivity seems to be around 5
    outlierThreshold = 5

    loutlier_bins = np.where(np.abs(lz_scores) > outlierThreshold)[0]
    lnumOutliers = loutlier_bins.size

    loutliersExist = (lnumOutliers > 0)


    #Calculate Statistics for Outliers in minimap

    #print(f'frame: {frameRead}')
    smean = np.mean(shist_mask)
    #print(f'mean: {smean}')
    sstd_dev = np.std(shist_mask)
    #print(f'stand dev: {sstd_dev}')

    if sstd_dev == 0:
        sz_scores = [0]

    else:
        sz_scores = (shist_mask - smean) / sstd_dev

    #print(f'zscore: {sz_scores}')

    soutlier_bins = np.where(np.abs(sz_scores) > outlierThreshold)[0]
    snumOutliers = soutlier_bins.size

    soutliersExist = (snumOutliers > 0)

    # if there is an error in both minimap and main footage:

    if (soutliersExist and np.all(soutlier_bins < 2)) and (loutliersExist and np.all(loutlier_bins < 2)):

        # return 3 if there is an error in both

        resultFlag = 3

    # if there is an error in just the main image:

    if (loutliersExist and np.all(loutlier_bins < 2)):

        # return 1 if there is an error in large

        resultFlag = 1

    # if there is an error in just the minimap image:

    if (soutliersExist and np.all(soutlier_bins < 2)):

        # return 2 if there is an error in small

        resultFlag = 2

    return resultFlag

#Main Function with Test
if __name__=="__main__":
    from auto_mask import create_mask
    #OpenCV Declaration
    video = cv2.VideoCapture("Pano to 70 glitch.mp4")
    totalFrames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = video.get(cv2.CAP_PROP_FPS)
    time_interval = 1/fps

    # ---------- Create Mask From First Video Frame -----------------------

    # first, read the starting frame:

    frame_read, frame = video.read()

    # Next, create masks for the main image and minimap: (lmask is main, smask is minimap)

    lmask,smask = create_mask(frame)

    # Reset the video frame grabber to start at frame 0

    video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if not video.isOpened():
        print("Video could not be opened")

    while video.isOpened():
        currentFrame = video.get(cv2.CAP_PROP_POS_FRAMES)
        timeStamp = currentFrame/fps
        frameRead, frame = video.read()
        if not frameRead:
            break

         #Check Pano-70

        panoState = checkPano(frame,smask,lmask)

        if(panoState == 1):
            print('Non Minimap Pano-70 Error Found at: ', round(timeStamp, 4), 'seconds')
        elif(panoState == 2):
            print('Minimap Pano-70 Error Found at: ', round(timeStamp, 4), 'seconds')
        elif(panoState == 3):
            print('Minimap and Main Screen Pano-70 Error Found at: ', round(timeStamp, 4), 'seconds')