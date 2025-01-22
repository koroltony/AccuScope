import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import measure
from skimage.draw import disk

# Create a mask for the first frame of the video:

# (only care about the 2 circular regions of video)

# Function to extract mask:

def create_mask(frame):

    # Extract dimensions of frame for size thresholds

    height,width,_ = np.shape(frame)
    area = height*width

    # convert to grayscale for thresholding

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Use Skimage measure to create labels for 0 and 1 regions

    labeled_image = measure.label(binary)

    # use regionprops to get the geometric properties of our mask regions

    properties = measure.regionprops(labeled_image)

    # Declare an array for the mask image

    large_mask = np.zeros_like(gray, dtype=np.uint8)

    # Create Threshold to only mask the circular regions (no numbers and letters):

    min_area = area/1500

    # For small mask:

    max_area = area/10

    # go through all of the properties (regions labeled 1) and find their radius and center
    # If the size of this property is within the threshold, we make it a filled in circle of ones

    for prop in properties:
        # Use regionprops to get necessary values

        radius = int(prop.equivalent_diameter / 2)
        center_y, center_x = map(int, prop.centroid)

        # Reject smaller circles using our threshold (leaving us with minimap and main footage)

        if prop.area >= min_area and prop.area >=max_area:
            # create a disk with the extracted properties and fill it with ones

            rr, cc = disk((center_y, center_x), radius, shape=large_mask.shape)
            large_mask[rr, cc] = 255

    return(large_mask)


if __name__ == "__main__":

    video_path = 'C:/Users/korol/Documents/Capstone TK/green flash and lag.mp4'

    video = cv2.VideoCapture(video_path)

    fps = video.get(cv2.CAP_PROP_FPS)
    num_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)

    frame_read, frame = video.read()

    frame = cv2.resize(frame, (1920, 1080))

    plt.figure()
    plt.title("Original Frame")
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    lmask,smask = create_mask(frame)

    # overlay the masks to see if it is working properly:

    frame[:,:,0] = frame[:,:,0] + 0.5*lmask + 0.5*smask

    # Display the mask and original frame

    plt.figure()
    plt.title("Mask of Main Video")
    plt.imshow(lmask, cmap='gray')

    plt.figure()
    plt.title("Mask of Minimap")
    plt.imshow(smask, cmap='gray')

    plt.figure()
    plt.title("Overlay Mask and Frame")
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    video.release()
    cv2.destroyAllWindows()