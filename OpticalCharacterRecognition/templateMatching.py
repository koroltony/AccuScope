import cv2
import numpy as np

target = cv2.imread('OpticalCharacterRecognition/menuImages/whiteBalance.jpg', cv2.IMREAD_GRAYSCALE)
frame = cv2.imread('OpticalCharacterRecognition/testImages/frame2.jpg', cv2.IMREAD_GRAYSCALE)

crossCorrNorm = cv2.matchTemplate(frame, target, cv2.TM_CCOEFF_NORMED)

# Get the best match location
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(crossCorrNorm)

# Draw a rectangle around the detected feature
h, w = target.shape  # Get template dimensions
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(frame, top_left, bottom_right, 255, 2)

# Save or display the result
cv2.imwrite("result.png", frame)
cv2.imshow("Detected Feature", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()