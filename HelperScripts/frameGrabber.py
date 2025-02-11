import cv2
import os

#Pass in a VideoCapture Object, starting time and ending time and function will return every frame
#For some reason you have to replace / with \ when passing into path
def captureFrames(video, startTime, endTime, path):

    fps = video.get(cv2.CAP_PROP_FPS)

    #Initialize Current Frame to 1
    currentFrame = 1

    #Check if Video Was Opened Successfully
    if not video.isOpened():
        print("Video could not be opened")

    while video.isOpened():
        #frameRead is whether the frame was successfully read
        frameRead, frame = video.read()

        #Break if Frame was Read Unsuccessfully
        if not frameRead:
            break

        #Resize from 4K into 1080p (My Monitor Only Supports 1080p)
        displayFrame = cv2.resize(frame, (960, 540))

        #Show the Individual Frame
        cv2.imshow('Surgery Video', displayFrame)

        #Save Frames of Green
        startFrame = round(fps * startTime)
        endFrame = round(fps * endTime)
        if currentFrame > startFrame and currentFrame < endFrame:
            filePathAndOutputName = os.path.join(path, f'frame{currentFrame}.jpg')
            cv2.imwrite(filePathAndOutputName, frame)
            print('Frame saved! Frame: ', currentFrame)

        #Press q to break
        if cv2.waitKey(25) & 0xFF == ord('q'): # Press 'q' to exit
            break

        currentFrame = currentFrame + 1

    video.release()
    #Close all open windows
    cv2.destroyAllWindows()

#Main Function with Test
if __name__=="__main__":
    video = cv2.VideoCapture('C:/Users/zionc/Videos/videoWithText.mkv')
    startTime = 0
    endTime = 4
    path = 'OpticalCharacterRecognition/testImages'

    captureFrames(video, startTime, endTime, path)