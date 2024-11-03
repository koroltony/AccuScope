import cv2
import os
import time
import numpy as np

start_time = time.time()

def highlights(video):
    totalFrames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = video.get(cv2.CAP_PROP_FPS)
    width = round(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = round(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not video.isOpened():
        print("Video could not be opened")

    while video.isOpened(): 
        currentFrame = video.get(cv2.CAP_PROP_POS_FRAMES)
        #print(round(currentFrame*100/totalFrames, 2),"% Done After", round(time.time() - start_time, 2), " Seconds")
        #frameRead is whether the frame was successfully read
        frameRead, frame = video.read()
        
        #Break if Frame was Read Unsuccessfully
        if not frameRead:
            break
        
        #Convert to Grey Scale
        greyScale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #Resize from 4K into 1080p (My Monitor Only Supports 1080p)
        displayFrame = cv2.resize(greyScale, (960, 540))

        #Show the Individual Frame
        cv2.imshow('Surgery Video', displayFrame)

        #print("frame: ", currentFrame, " time: ", round(currentFrame/fps, 2))

        #Press q to break
        if cv2.waitKey(25) & 0xFF == ord('q'): # Press 'q' to exit
            break
    
    video.release()
    #Close all open windows
    cv2.destroyAllWindows()

#Main Function with Test
if __name__=="__main__":
    video = cv2.VideoCapture('HDR shimmer coracoid.mp4')
    highlights(video)
    print("--- %s seconds ---" % (time.time() - start_time))