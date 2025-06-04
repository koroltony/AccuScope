import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Read the images
img = cv.imread('OpticalCharacterRecognition/testImages/frame198.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
template = cv.imread('OpticalCharacterRecognition/menuImages/Success.jpg', cv.IMREAD_GRAYSCALE)
assert template is not None, "file could not be read, check with os.path.exists()"

# Resize images to 80% of their original size
scale_percent = 1  # Scaling factor
img = cv.resize(img, None, fx=scale_percent, fy=scale_percent, interpolation=cv.INTER_AREA)
template = cv.resize(template, None, fx=scale_percent, fy=scale_percent, interpolation=cv.INTER_AREA)

w, h = template.shape[::-1]

# Define the methods
methods = ['TM_CCOEFF', 'TM_CCOEFF_NORMED', 'TM_CCORR',
           'TM_CCORR_NORMED', 'TM_SQDIFF', 'TM_SQDIFF_NORMED']

fig, axes = plt.subplots(2, len(methods) + 1, figsize=(3 * (len(methods) + 1), 6))  # Extra column for labels

# Set the first column labels
axes[0, 0].text(0.5, 0.5, "Matching Result", fontsize=12, ha='center', va='center', rotation=90)
axes[1, 0].text(0.5, 0.5, "Detected Point", fontsize=12, ha='center', va='center', rotation=90)

# Remove axis for label cells
axes[0, 0].axis('off')
axes[1, 0].axis('off')

# Loop through methods and plot results
for i, meth in enumerate(methods):
    img_copy = img.copy()
    method = getattr(cv, meth)

    # Apply template matching
    res = cv.matchTemplate(img_copy, template, method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    # Determine the best match location
    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # Draw rectangle around detected template
    cv.rectangle(img_copy, top_left, bottom_right, (255), 2)

    # Add method name to the top row
    axes[0, i + 1].set_title(meth, fontsize=10)

    # Plot matching result
    axes[0, i + 1].imshow(res, cmap='gray')
    axes[0, i + 1].axis('off')

    # Plot detected template in the original image
    axes[1, i + 1].imshow(img_copy, cmap='gray')
    axes[1, i + 1].axis('off')

plt.tight_layout()
plt.show()