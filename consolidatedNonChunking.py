import cv2
import os
import numpy as np
import time
import matplotlib.pyplot as plt
import subprocess
import sys
from greenScripts.greenScreen import checkGreenFrame
from Highlights.highlights import checkHighlightsFrame
from Frozen.lagff15 import detect_frozen_frame
from HelperScripts.auto_mask import create_mask
from panoto70.panoto70fcn import checkPano

codeStart = time.time()

#OpenCV Declaration
video = cv2.VideoCapture("panoto70/Pano to 70 glitch.mp4")
totalFrames = video.get(cv2.CAP_PROP_FRAME_COUNT)
fps = video.get(cv2.CAP_PROP_FPS)
time_interval = 1/fps
tempTime = time.time()

frameRead, prev_frame = video.read()
frozen_frame_flags = []

# ---------- Create Mask From First Video Frame -----------------------

# first, read the starting frame:

frame_read, frame = video.read()

# Next, create masks for the main image and minimap: (lmask is main, smask is minimap)

lmask,smask = create_mask(frame)

# Reset the video frame grabber to start at frame 0

video.set(cv2.CAP_PROP_POS_FRAMES, 0)

# ---------------------------------------------------------------------

if not video.isOpened():
    print("Video could not be opened")

while video.isOpened():
    currentFrame = video.get(cv2.CAP_PROP_POS_FRAMES)
    timeStamp = currentFrame/fps
    frameRead, frame = video.read()
    if not frameRead:
        break

    #Check Green
    greenState = checkGreenFrame(frame)
    if(greenState == 1):
        print('Non Minimap Green Screen Error Found at: ', round(timeStamp, 4), 'seconds')
    elif(greenState == 2):
        print('Minimap Green Screen Error Found at: ', round(timeStamp, 4), 'seconds')

    #Check Highlights
    if checkHighlightsFrame:
        print('Highlight Shimmer at ', round(timeStamp, 2), 'seconds')

    #Check Frozen Frames
    current_time = time.time()

    start_time = time.time()
    if current_time - tempTime >= time_interval:
        tempTime = current_time
        if detect_frozen_frame(prev_frame, frame):
            frozen_frame_flags.append(1)  # Append 1 when frozen frame is detected
            print("Frozen frame detected! ", round((currentFrame/fps), 4), 'seconds')
        else:
            frozen_frame_flags.append(0)  # Append 0 when frozen frame is detected
        prev_frame = frame


    #Check Pano-70

    panoState = checkPano(frame,smask,lmask)

    if(panoState == 1):
        print('Non Minimap Pano-70 Error Found at: ', round(timeStamp, 4), 'seconds')
    elif(panoState == 2):
        print('Minimap Pano-70 Error Found at: ', round(timeStamp, 4), 'seconds')
    elif(panoState == 3):
        print('Minimap and Main Screen Pano-70 Error Found at: ', round(timeStamp, 4), 'seconds')


    #Resize from 4K into 1080p (My Monitor Only Supports 1080p)
    displayFrame = cv2.resize(frame, (960, 540))

    #Show the Individual Frame
    cv2.imshow('Surgery Video', displayFrame)

    #Press q to break
    if cv2.waitKey(25) & 0xFF == ord('q'): # Press 'q' to exit
        break

    # Parameters

print("--- %s seconds ---" % (time.time() - codeStart))

window_size = 10

# Sliding window sum
window_sums = [sum(frozen_frame_flags[i:i+window_size]) for i in range(len(frozen_frame_flags) - window_size + 1)]

# Plotting with window sum (window sum for visualizing concentration otherwise we would see a bunch of 1's in a row)
plt.figure(figsize=(10, 5))
plt.plot(window_sums, label='Window Sums')
#plt.axhline(y=np.mean(window_sums), color='r', linestyle='--', label='Mean')
plt.xlabel('Index')
plt.ylabel('Sum of Ones')
plt.title('Concentration of Detections')
plt.legend()
plt.show()