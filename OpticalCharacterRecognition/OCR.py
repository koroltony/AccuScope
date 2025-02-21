#Implementation of OCR using EasyOCR

import cv2
import easyocr
import matplotlib.pyplot as plt
import time

def draw_bounding_boxes(image, detections, threshold=0.25):
    for bbox, text, score in detections:
        if score > threshold:
            cv2.rectangle(image, tuple(map(int, bbox[0])), tuple(map(int, bbox[2])), (0, 255, 0), 5)
            cv2.putText(image, text, tuple(map(int, bbox[0])), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.65, (255, 0, 0), 2)

image_path = "OpticalCharacterRecognition/testImages/frame1.jpg"
img = cv2.imread(image_path)

startTime = time.time()
reader = easyocr.Reader(['en'])
result = reader.readtext(image_path)
print(f'number of detections: len(result)')
endTime = time.time()
print(endTime - startTime)

for (bbox, text, prob) in result:
    print(f'Text: {text}, Probability: {prob}')

threshold = 0.25
draw_bounding_boxes(img, result, threshold)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGBA))
plt.show()