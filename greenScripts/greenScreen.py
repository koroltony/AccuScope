import cv2
import os
import time
import numpy as np

start_time = time.time()

def checkGreenFrame(frame):
    BRthreshold = 50
    Gthreshold = 100
    miniMapTopLeft = np.array([63, 35])
    miniMapBottomRight = np.array([416, 664])
    miniMapMiddleCord = np.add(miniMapTopLeft, np.subtract(miniMapBottomRight, miniMapTopLeft)//2)
    height, width = frame.shape[:2]
    Bframemiddle = frame[height//2, width//2, 0]
    Gframemiddle = frame[height//2, width//2, 1]
    Rframemiddle = frame[height//2, width//2, 2]
    Bminimapmiddle = frame[miniMapMiddleCord[1], miniMapMiddleCord[0], 0]
    Gminimapmiddle = frame[miniMapMiddleCord[1], miniMapMiddleCord[0], 1]
    Rminimapmiddle = frame[miniMapMiddleCord[1], miniMapMiddleCord[0], 2]

    if(Bframemiddle < BRthreshold) and (Gframemiddle > Gthreshold) and (Rframemiddle < BRthreshold):
        return 1
        #print('Non Minimap Green Screen Error Found at: ', round((currentFrame/fps), 4), 'seconds')
    elif(Bminimapmiddle < BRthreshold) and (Gminimapmiddle > Gthreshold) and (Rminimapmiddle < BRthreshold):
        return 2
        #print('Minimap Green Screen Error Found at: ', round((currentFrame/fps), 4), 'seconds')

def checkGreen(video):
    totalFrames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = video.get(cv2.CAP_PROP_FPS)
    width = round(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = round(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    BRthreshold = 50
    Gthreshold = 100
    miniMapTopLeft = np.array([63, 35])
    miniMapBottomRight = np.array([416, 664])
    miniMapMiddleCord = np.add(miniMapTopLeft, np.subtract(miniMapBottomRight, miniMapTopLeft)//2)
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
        
        #Resize from 4K into 1080p (My Monitor Only Supports 1080p)
        #displayFrame = cv2.resize(frame, (960, 540))

        #Show the Individual Frame
        #cv2.imshow('Surgery Video', displayFrame)

        #Minimap exists from (35, 63) and ends at (664, 416)
        #Get R, G, B pixel value in the middle of each frame
        Bframemiddle = frame[height//2, width//2, 0]
        Gframemiddle = frame[height//2, width//2, 1]
        Rframemiddle = frame[height//2, width//2, 2]
        Bminimapmiddle = frame[miniMapMiddleCord[1], miniMapMiddleCord[0], 0]
        Gminimapmiddle = frame[miniMapMiddleCord[1], miniMapMiddleCord[0], 1]
        Rminimapmiddle = frame[miniMapMiddleCord[1], miniMapMiddleCord[0], 2]

        if(Bframemiddle < BRthreshold) and (Gframemiddle > Gthreshold) and (Rframemiddle < BRthreshold):
            print('Non Minimap Green Screen Error Found at: ', round((currentFrame/fps), 4), 'seconds')
        elif(Bminimapmiddle < BRthreshold) and (Gminimapmiddle > Gthreshold) and (Rminimapmiddle < BRthreshold):
            print('Minimap Green Screen Error Found at: ', round((currentFrame/fps), 4), 'seconds')

        #print("frame: ", currentFrame, " time: ", round(currentFrame/fps, 2))

        #Press q to break
        if cv2.waitKey(25) & 0xFF == ord('q'): # Press 'q' to exit
            break
    
    video.release()
    #Close all open windows
    cv2.destroyAllWindows()

#Main Function with Test
if __name__=="__main__":
    frame = cv2.imread('C:/Users/zionc/Documents/GitHub/ece188a-arthrex/Green Screen Scripts/frame831.jpg')
    cv2.imshow('Image', frame)
    checkGreenFrame(frame)
    print("--- %s seconds ---" % (time.time() - start_time))