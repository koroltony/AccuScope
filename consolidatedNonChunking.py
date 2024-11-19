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

#OpenCV Declaration
video = cv2.VideoCapture("C:/Users/16262/Desktop/arthrex/green flash and lag 3.mp4")
totalFrames = video.get(cv2.CAP_PROP_FRAME_COUNT)
fps = video.get(cv2.CAP_PROP_FPS)
time_interval = 1/fps
codeStart = time.time()

frameRead, prev_frame = video.read()
frozen_frame_flags = []

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
    #if checkHighlightsFrame:
        #print('Highlight Shimmer at ', round(timeStamp, 2), 'seconds')

    #Check Frozen Frames
    current_time = time.time()
      
    start_time = time.time()
    if current_time - codeStart >= time_interval:
        codeStart = current_time
        if detect_frozen_frame(prev_frame, frame):
            frozen_frame_flags.append(1)  # Append 1 when frozen frame is detected
            print("Frozen frame detected! ", round((currentFrame/fps), 4), 'seconds')
        else: 
            frozen_frame_flags.append(0)  # Append 0 when frozen frame is detected
        prev_frame = frame
        

    #Resize from 4K into 1080p (My Monitor Only Supports 1080p)
    displayFrame = cv2.resize(frame, (960, 540))

    #Show the Individual Frame
    cv2.imshow('Surgery Video', displayFrame)

    #Press q to break
    if cv2.waitKey(25) & 0xFF == ord('q'): # Press 'q' to exit
        break

print("--- %s seconds ---" % (time.time() - codeStart))