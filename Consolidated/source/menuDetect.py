#Function takes in a matrix of grayscale pixels and a folder path to reference kernels
#returns a boolean if a menu exists inside.
#The Fast Fourier Transform is used to speed up operations.


#Returns True if there exists a Kernel that has a match of over 0.9 confidence.
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time

def hasMenu(frame, kernelsPath):
    greyFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    menuThresholds = []
    for filename in os.listdir(kernelsPath):
        fullPath = os.path.join(kernelsPath, filename)
        if os.path.isfile(fullPath):
            grayKernel = cv2.imread(fullPath, cv2.IMREAD_GRAYSCALE)
            res = cv2.matchTemplate(greyFrame, grayKernel, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            menuThresholds.append(max_val)
    print(menuThresholds)
    menuThresholds = np.array(menuThresholds)
    return np.any(menuThresholds > 0.9)

start_time = time.time()
frame = cv2.imread('OpticalCharacterRecognition/testImages/frame2.jpg')
print(hasMenu(frame, 'OpticalCharacterRecognition/menuImages'))
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")