import cv2
import os
import numpy as np
import time
from auto_mask import create_mask
from matplotlib import pyplot as plt

start_time = time.time()

def pano70(video):
    fps = video.get(cv2.CAP_PROP_FPS)

    if not video.isOpened():
        print("Video could not be opened")

    # ---------- Create Mask From First Video Frame -----------------------

    # first, read the starting frame:

    frame_read, frame = video.read()

    # Next, create masks for the main image and minimap: (lmask is main, smask is minimap)

    lmask,smask = create_mask(frame)

    # Reset the video frame grabber to start at frame 0

    video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    lhist_mask = cv2.calcHist([frame], [0], lmask, [256], [1, 256])

    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title("Normal Frame")
    plt.subplot(1,2,2)
    plt.plot(lhist_mask)
    plt.axhline(y=np.mean(lhist_mask), color='r', linestyle='-', linewidth=3)
    plt.title("Normal Histogram")

    # ----------------------------------------------------------------------

    while video.isOpened():
        frameRead, frame = video.read()
        #Break if Frame was Read Unsuccessfully
        if not frameRead:
            break

        currentFrame = video.get(cv2.CAP_PROP_POS_FRAMES)
        greyFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #Resize from 4K into 1080p (My Monitor Only Supports 1080p)
        displayFrame = cv2.resize(greyFrame, (960, 540))

        #Apply the mask using a bitwise and of the frame on itself with the mask we just defined.
        #lmasked_greyFrame = cv2.bitwise_and(greyFrame, greyFrame, mask=lmask)
        #smasked_greyFrame = cv2.bitwise_and(greyFrame, greyFrame, mask=smask)

        # Calculate the histogram with and without the mask
        #hist_full = cv2.calcHist([greyFrame], [0], None, [256], [1, 256])

        # Calculate separate histograms for the minimap and the main image:

        lhist_mask = cv2.calcHist([greyFrame], [0], lmask, [256], [1, 256])
        shist_mask = cv2.calcHist([greyFrame], [0], smask, [256], [1, 256])

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

        if (soutliersExist and np.all(soutlier_bins < 2)) or (loutliersExist and np.all(loutlier_bins < 2)):
            timeStamp = video.get(cv2.CAP_PROP_POS_FRAMES)/ fps
            print('Pano to 70 at ', round(timeStamp, 2), 'seconds')
        #print("frame: ", currentFrame, " time: ", round(currentFrame/fps, 2))
            #Show the Individual Frame and its histogram
            plt.figure()
            plt.subplot(1,2,1)
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.title("Extracted Pano to 70 Glitch")
            plt.subplot(1,2,2)
            plt.plot(lhist_mask)
            plt.axhline(y=np.mean(lhist_mask), color='r', linestyle='-', linewidth=3)
            plt.title("Error Histogram")

        #Press q to break
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

#Main Function with Test
if __name__=="__main__":
    video = cv2.VideoCapture('Pano to 70 glitch.mp4')
    pano70(video)
    print("--- %s seconds ---" % (time.time() - start_time))