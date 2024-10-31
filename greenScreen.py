import cv2
import os
import time

start_time = time.time()

def checkGreen(video):
    totalFrames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = video.get(cv2.CAP_PROP_FPS)
    width = round(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = round(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    BRthreshold = 50
    Gthreshold = 100
    if not video.isOpened():
        print("Video could not be opened")

    while video.isOpened(): 
        currentFrame = video.get(cv2.CAP_PROP_POS_FRAMES)
        print(round(currentFrame*100/totalFrames, 2),"% Done After", round(time.time() - start_time, 2), " Seconds")
        #frameRead is whether the frame was successfully read
        frameRead, frame = video.read()

        #Break if Frame was Read Unsuccessfully
        if not frameRead:
            break
        
        #Get R, G, B pixel value in the middle of each frame
        Bframemiddle = frame[height//2, width//2, 0]
        Gframemiddle = frame[height//2, width//2, 1]
        Rframemiddle = frame[height//2, width//2, 2]
        

        if(Bframemiddle < BRthreshold) and (Gframemiddle > Gthreshold) and (Rframemiddle < BRthreshold):
            print('Non Minimap Green Screen Error Found at: ', round((currentFrame/fps), 4), 'seconds')

        #print("frame: ", currentFrame, " time: ", round(currentFrame/fps, 2))

        #Press q to break
        if cv2.waitKey(25) & 0xFF == ord('q'): # Press 'q' to exit
            break
    
    video.release()
    #Close all open windows
    cv2.destroyAllWindows()

#Main Function with Test
if __name__=="__main__":
    video = cv2.VideoCapture('green flash and lag.mp4')
    checkGreen(video)
    print("--- %s seconds ---" % (time.time() - start_time))