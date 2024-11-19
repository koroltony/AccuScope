import cv2
import os
import numpy as np
import time
import matplotlib.pyplot as plt

start_time = time.time()

def highlights(video):
    totalFrames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = video.get(cv2.CAP_PROP_FPS)
    width = round(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = round(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #Radius of Circular Mask
    radius = 1100
    #Middle Coordinate of a 4K Synergy Image
    middleCord = np.array([width//2, height//2])

    #Used MATLAB Image Viewer Toolbox to find the coordinates of the the top left and bottom right
    #of the minimap. The format is ex: (662, 63) or (width, height)
    miniMapTopLeftCord = np.array([35, 63])
    miniMapBottomRightCord = np.array([662, 414])

    #Declare a mask with same dimensions as 4K image
    mask = np.zeros((height, width), np.uint8)

    #Draw a circle on the image. On the mask, go to the middle coordinate, extend out a radius. Draw white
    #(White=255) on the black mask. (White indicates the parts of the mask that will be let through)
    cv2.circle(mask, middleCord, radius, 255, -1)

    #Draw a rectangle on the image. On the Mask, go to the topleft and bottom right of the minimap
    #The rectangle defined in this area will be white upon the black (zeros) mask.
    cv2.rectangle(mask, miniMapTopLeftCord, miniMapBottomRightCord, 255, -1)

    if not video.isOpened():
        print("Video could not be opened")
    
    while video.isOpened(): 
        frameRead, frame = video.read()
        #Break if Frame was Read Unsuccessfully
        if not frameRead:
            break

        currentFrame = video.get(cv2.CAP_PROP_POS_FRAMES)
        greyFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #Resize from 4K into 1080p (My Monitor Only Supports 1080p).
        displayFrame = cv2.resize(greyFrame, (960, 540))
        #Show the Individual Frame
        cv2.imshow('Surgery Video', displayFrame)

        #Apply the mask using a bitwise and of the frame on itself with the mask we just defined.
        masked_greyFrame = cv2.bitwise_and(greyFrame, greyFrame, mask=mask)

        # Calculate the histogram with and without the mask
        #hist_full = cv2.calcHist([greyFrame], [0], None, [256], [1, 256])
        hist_mask = cv2.calcHist([greyFrame], [0], mask, [256], [1, 256])

        #Calculate Statistics for Outliers
        mean = np.mean(hist_mask)
        std_dev = np.std(hist_mask)
        z_scores = (hist_mask - mean) / std_dev

        #Sensitivity seems to be around 3-4.
        outlierThreshold = 3

        outlier_bins = np.where(np.abs(z_scores) > outlierThreshold)[0]
        numOutliers = outlier_bins.size
        outliersExist = numOutliers > 0

        if outliersExist and np.all(outlier_bins > 200):
            timeStamp = currentFrame/fps
            print('Highlight Shimmer at ', round(timeStamp, 2), 'seconds')
        #print("frame: ", currentFrame, " time: ", round(currentFrame/fps, 2))

        #Press q to break
        if cv2.waitKey(25) & 0xFF == ord('q'): # Press 'q' to exit
            break
    
    video.release()
    cv2.destroyAllWindows()

#Main Function with Test
if __name__=="__main__":
    video = cv2.VideoCapture('HDR shimmer coracoid.mp4')
    highlights(video)
    print("--- %s seconds ---" % (time.time() - start_time))