import threading
from greenScreen import checkGreen 
from highlights import highlights
import cv2 
import time

#Threading is not going to help us because of Python GIL. It's only good
#for tasks that are I/O heavy. We have a CPU intensive task, so there
#is no downtime for the code to run the other thread.
start_time = time.time()

video1 = cv2.VideoCapture('green flash and lag.mp4')
video2 = cv2.VideoCapture('HDR shimmer coracoid.mp4')

greenThread = threading.Thread(target=checkGreen(video1))
highlightsThread = threading.Thread(target=highlights(video2))

greenThread.start()
highlightsThread.start()

greenThread.join()
highlightsThread.join()

print("--- %s seconds ---" % (time.time() - start_time))