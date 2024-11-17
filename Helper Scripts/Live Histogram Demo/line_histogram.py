import cv2
import os
import numpy as np
from auto_mask import create_mask

# Open the video file
input_path = 'short_shimmer.mp4'
output_path = 'hist_vid.mp4'
cap = cv2.VideoCapture(input_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = 256
height = 256

# Create the VideoWriter for the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# ---------- Create Mask From First Video Frame -----------------------
# first, read the starting frame:
frame_read, frame = cap.read(cv2.IMREAD_GRAYSCALE)

# Create masks for the main image and minimap (lmask is main, smask is minimap)
lmask, smask = create_mask(frame)

# Reset the video frame grabber to start at frame 0
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# ----------------------------------------------------------------------


while cap.isOpened():

    frame_read, frame = cap.read(cv2.IMREAD_GRAYSCALE)

    if not frame_read:
        break

    # Apply the mask using bitwise and (lmask is for main video)
    lhist_mask = cv2.calcHist([frame], [0], lmask, [256], [1, 256])

    # makes sure the histogram value is between 0 and 255 so that the plotting will work
    cv2.normalize(lhist_mask, lhist_mask, 0, 255, cv2.NORM_MINMAX)

    # Display images and histograms

    hist_image = np.zeros((256, 256), dtype=np.uint8)

    # Draw the histogram lines (image, start point, end point)
    for i in range(1, 256):
        cv2.line(hist_image, (i-1, 255 - int(lhist_mask[i][0])), (i-1, 255), 255, 1)

    # # Display the histogram
    # cv2.imshow('line histogram', hist_image)
    # cv2.waitKey(0)

    # Write the frame to the output video
    out.write(cv2.cvtColor(hist_image, cv2.COLOR_GRAY2BGR))

# Release resources
cap.release()
out.release()
print("Histogram video saved successfully!")