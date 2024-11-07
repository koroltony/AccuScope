import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

#Read in Grey Frame
greyFrame = cv2.imread('frame73.jpg', cv2.IMREAD_GRAYSCALE)

#Threshold values above 1 in a Greyscale Space of [0-255] to black. Values == 1 are white.
threshholdValue, mask = cv2.threshold(greyFrame, 1, 255, cv2.THRESH_BINARY_INV)

#Save Dimensions of the 4K Image
height, width = greyFrame.shape[:2]

#Radius of Circular Mask
radius = 1100

#Middle Coordinate of a 4K Synergy Image
middleCord = np.array([width//2, height//2])

#Declare a mask with same dimensions as 4K image
mask = np.zeros((height, width), np.uint8)

#Used MATLAB Image Viewer Toolbox to find the coordinates of the the top left and bottom right
#of the minimap. The format is ex: (662, 63) or (width, height)
miniMapTopLeftCord = np.array([35, 63])
miniMapBottomRightCord = np.array([662, 414])

#Draw a circle on the image. On the mask, go to the middle coordinate, extend out a radius. Draw white
# (White=255) on the black mask. (White indicates the parts of the mask that will be let through)
cv2.circle(mask, middleCord, radius, 255, -1)

#Draw a rectangle on the image. On the Mask, go to the topleft and bottom right of the minimap
#The rectangle defined in this area will be white upon the black (zeros) mask.
cv2.rectangle(mask, miniMapTopLeftCord, miniMapBottomRightCord, 255, -1)

#Apply the mask using a bitwise and of the frame on itself with the mask we just defined.
masked_greyFrame = cv2.bitwise_and(greyFrame, greyFrame, mask=mask)

# Calculate the histogram with and without the mask
#hist_full = cv2.calcHist([greyFrame], [0], None, [256], [1, 256])
hist_mask = cv2.calcHist([greyFrame], [0], mask, [256], [1, 256])

mean = np.mean(hist_mask)
std_dev = np.std(hist_mask)
z_scores = (hist_mask - mean) / std_dev

outlierThreshold = 4
outlier_bins = np.where(np.abs(z_scores) > outlierThreshold)[0]
numOutliers = outlier_bins.size
outliersExist = numOutliers > 0

# Display images and histograms
#plt.subplot(221), plt.imshow(greyFrame, 'gray'), plt.title('Original Image')
#plt.subplot(222), plt.imshow(mask, 'gray'), plt.title('Circular Mask')
#plt.subplot(223), plt.imshow(masked_greyFrame, 'gray'), plt.title('Masked Image')
#plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
#plt.xlim([0, 256])

#plt.show()