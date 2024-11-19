import cv2
import os
import numpy as np
import time
import matplotlib.pyplot as plt
from greenScripts.greenScreen import checkGreenFrame
from Highlights.highlights import checkHighlightsFrame

start_time = time.time()

#OpenCV Declaration
video = cv2.VideoCapture("C:/Users/zionc/Documents/Arthrex/green flash and lag.mp4")
totalFrames = video.get(cv2.CAP_PROP_FRAME_COUNT)
fps = video.get(cv2.CAP_PROP_FPS)

if not video.isOpened():
    print("Video could not be opened")

while video.isOpened(): 
    currentFrame = video.get(cv2.CAP_PROP_POS_FRAMES)
    frameRead, frame = video.read()
    if not frameRead:
        break
    
    greenState = checkGreenFrame(frame)
    if(greenState == 1):
        print('Non Minimap Green Screen Error Found at: ', round((currentFrame/fps), 4), 'seconds')
    elif(greenState == 2):
        print('Minimap Green Screen Error Found at: ', round((currentFrame/fps), 4), 'seconds')

    #Resize from 4K into 1080p (My Monitor Only Supports 1080p)
    displayFrame = cv2.resize(frame, (960, 540))

    #Show the Individual Frame
    cv2.imshow('Surgery Video', displayFrame)

    #Press q to break
    if cv2.waitKey(25) & 0xFF == ord('q'): # Press 'q' to exit
        break

print("--- %s seconds ---" % (time.time() - start_time))