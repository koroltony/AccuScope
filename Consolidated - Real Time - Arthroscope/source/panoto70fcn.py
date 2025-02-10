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

def checkPano(frame,mask):

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

    if (loutliersExist and np.all(loutlier_bins < 2)):

        # return 1 if there is an error in large

        resultFlag = 1

    return resultFlag

def checkPanoEdge(frame, lmask):

    # Detect edges using Canny edge filter
    # Thresholds are intentionally high to make only strong edges appear

    # Thresholds are optimized for 1080p video

    edges = cv2.Canny(frame, threshold1=400, threshold2=600)

    # # Shrink the mask inward by a few pixels because the edge of the mask gets in the way
    # kernel = np.ones((3, 3), np.uint8)
    # shrunk_mask = cv2.erode(lmask, kernel, iterations=40)

    edges = cv2.bitwise_and(edges, edges, mask=lmask)

    # cv2.imshow('Edge',edges)

    # # Wait for a key press for 1ms and check if 'k' is pressed
    # cv2.waitKey(1) & 0xFF

    # Find contours from the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Filter contours: keep only edges longer than threshold

    min_edge_length = 100

    longEdges = []

    for contour in contours:
        if cv2.arcLength(contour, closed=False) > min_edge_length:
            longEdges.append(contour)

    # If there is more than 1 edge, it is probably a pano to 70 glitch
    return len(longEdges)>1


#Main Function with Test
if __name__=="__main__":

    w = 4000
    h = 3000
    from auto_mask import create_mask
    #OpenCV Declaration
    video = cv2.VideoCapture("C:/Users/korol/Documents/Arthrex Code/ece188a-arthrex/panoto70/Pano to 70 glitch.mp4")
    totalFrames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = video.get(cv2.CAP_PROP_FPS)
    time_interval = 1/fps

    # ---------- Create Mask From First Video Frame -----------------------

    # first, read the starting frame:

    frame_read, frame = video.read()

    frame = cv2.resize(frame,[w,h])

    # Next, create masks for the main image and minimap: (lmask is main, smask is minimap)

    lmask = create_mask(frame)
    plt.imshow(lmask)

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
        frame = cv2.resize(frame,[w,h])

        panoState = checkPanoEdge(frame,lmask)

        if(panoState == 1):
            print('Non Minimap Pano-70 Error Found at: ', round(timeStamp, 4), 'seconds')