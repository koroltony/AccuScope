import cv2
import numpy as np
import time
import keyboard
import sys
import os
import matplotlib.pyplot as plt
from source.greenVectorizedSolution import checkGreenFrame
from source.magentaScreen import checkMagentaFrame
from source.dropoutScreen import checkBlackFrame
from source.highlights import checkHighlightsFrame
from source.lagff15 import detect_frozen_frame
from source.auto_mask import create_mask
from source.panoto70fcn import checkPanoEdge

error_video_path = 'intermediate_video.mp4'

codeStart = time.time()
frozen_frame_flags = []

# Initialize video stream
video = cv2.VideoCapture(2)

# Set the MJPG codec
video.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# set the frame rate:

video.set(cv2.CAP_PROP_FPS,60)

# Make sure it is not repeating frames

video.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not video.isOpened():
    print("Camera could not be opened")
    exit()

fps = int(video.get(cv2.CAP_PROP_FPS))
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_size = (2*frame_width, frame_height)

# Create an output stream to hold the prototype video
out = cv2.VideoWriter(error_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 60, output_size)

# create mask for arthroscope footage based on first frame

print('Press "k" to set mask for the footage')

while True:
    _, initial_frame = video.read()
    lmask = create_mask(initial_frame)
    cv2.imshow('Footage Mask',lmask)

    # Wait for a key press for 1ms and check if 'k' is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('k'):
        print("Mask Set")
        break

    if keyboard.is_pressed('k'):
        print("Mask Set")
        break

cv2.destroyAllWindows()

# Create mask for pano to 70

shrunk_mask = lmask
keypress = False
keypresg = False

print("Shrink/grow mask until you don't see any edges")
print('press "s" to shrink the mask')
print('press "g" to grow mask')
print('press "d" to begin processing')

while True:

    # Detect edges using Canny edge filter
    # Thresholds are intentionally high to make only strong edges appear

    edges = cv2.Canny(initial_frame, threshold1=100, threshold2=200)

    # Shrink the mask inward by a few pixels because the edge of the mask gets in the way

    kernel = np.ones((3, 3), np.uint8)

    if keypress and not keyboard.is_pressed('s'):
        print("Pano-to-70 Mask Shrunk")
        shrunk_mask = cv2.erode(shrunk_mask, kernel, iterations=1)
        keypress = False

    elif keyboard.is_pressed('s') and not keypress:
        keypress = True

    if keypresg and not keyboard.is_pressed('g'):
        print("Pano-to-70 Mask Grown")
        shrunk_mask = cv2.dilate(shrunk_mask, kernel, iterations=1)
        keypresg = False

    elif keyboard.is_pressed('g') and not keypress:
        keypresg = True

    edges = cv2.bitwise_and(edges, edges, mask=shrunk_mask)

    # Convert edges to a 3-channel red overlay (R=255, G=0, B=0)
    edges_red = cv2.merge([np.zeros_like(edges), np.zeros_like(edges), edges])

    # Convert the mask to a 3-channel green overlay (R=0, G=255, B=0)
    mask_green = cv2.merge([np.zeros_like(shrunk_mask), shrunk_mask, np.zeros_like(shrunk_mask)])
    mask_green = (mask_green * 0.3).astype(np.uint8)

    # Overlay edges onto the mask
    overlay = cv2.addWeighted(mask_green, 0.5, edges_red, 1, 0)

    cv2.imshow('Pano-70 Mask',overlay)

    # Wait for a key press for 1ms and check if 'k' is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('d'):
        print("Pano-70 Mask Set")
        break

    if keyboard.is_pressed('d'):
        print("Pano-70 Mask Set")
        break

cv2.destroyAllWindows()

# Create variables for the error visualization
prev_frame = None
black_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
error_frame = black_frame.copy()
error_text = "No Error"
# This makes the error pop up on the screen for a good amount of time (so you can actually see it)
error_duration = 2*fps
error_counter = 0

print('Press "q" to exit recording')

# Raw Video Saving
# ------------------------------------------------
output_folder = "Raw_Videos"
os.makedirs(output_folder, exist_ok=True)

# Get a list of all existing files in the folder
existing_files = os.listdir(output_folder)

# Find the highest existing video index
video_indices = [
    int(f.split("RawVideo")[1].split(".")[0]) for f in existing_files if f.startswith("RawVideo") and f.endswith(".mp4")
]
next_index = max(video_indices, default=0) + 1

# Define the output video path with the new name
output_video_path = os.path.join(output_folder, f"RawVideo{next_index}.mp4")

saved_vid = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, [frame_width,frame_height])

frame_count = 0
start_time = time.time()

# -------------------------------------------------

# ---------------- Code for frame Grabbing (frozen frame debugging)-----
currentFrame = 1
path = 'C:/Users/korol/Documents/Arthrex Code/ece188a-arthrex/frame'
# ----------------------------------------------------------------------

while True:

    ret, frame = video.read()
    if not ret:
        break

    filePathAndOutputName = os.path.join(path, f'frame{currentFrame}.jpg')
    cv2.imwrite(filePathAndOutputName, frame)

    currentFrame += 1
    # print(frame_count)
    # print(f"Frame {frame_count} sum: {np.sum(frame)}")
    frame_count += 1

    time_stamp = time.time() - codeStart

    # Check Errors
    green_state = checkGreenFrame(frame)
    if green_state == 1:
        # Create error text and error frame variables to display later
        error_text = f"Majority Green Screen Error at {time_stamp:.2f}s"
        print(f"Majority Green Screen Error at {time_stamp:.2f}s")
        error_frame = frame.copy()
        error_counter = error_duration
    elif green_state == 2:
        error_text = f"Partial Green Screen Error at {time_stamp:.2f}s"
        error_frame = frame.copy()
        print(f"Partial Green Screen Error at {time_stamp:.2f}s")
        error_counter = error_duration

    magenta_state = checkMagentaFrame(frame)
    if magenta_state == 1:
        # Create error text and error frame variables to display later
        error_text = f"Magenta Screen Error at {time_stamp:.2f}s"
        print(f"Magenta Screen Error at {time_stamp:.2f}s")
        error_frame = frame.copy()
        error_counter = error_duration

    black_state = checkBlackFrame(frame,lmask)
    if black_state == 1:
        # Create error text and error frame variables to display later
        error_text = f"Dropout Error at {time_stamp:.2f}s"
        print(f"Dropout Error at {time_stamp:.2f}s")
        error_frame = frame.copy()
        error_counter = error_duration

    # if checkHighlightsFrame(frame,lmask):
    #     error_text = f"Highlight Shimmer at {time_stamp:.2f}s"
    #     print(f"Highlight Shimmer at {time_stamp:.2f}s")
    #     error_frame = frame.copy()
    #     error_counter = error_duration

    # if prev_frame is not None and detect_frozen_frame(prev_frame, frame):
    #     print(f"Frozen Frame at {time_stamp:.2f}s")
    #     error_text = f"Frozen Frame at {time_stamp:.2f}s"
    #     frozen_frame_flags.append(1)
    #     error_frame = frame.copy()
    #     error_counter = error_duration

    # else:
    #     frozen_frame_flags.append(0)

    # only check pano if we did not already detect a dropout (because pano flags dropout)
    if black_state != 1:
        pano_state = checkPanoEdge(frame,shrunk_mask)
        if pano_state == 1:
            error_text = f"Pano-70 Error at {time_stamp:.2f}s"
            print(f"Pano-70 Error at {time_stamp:.2f}s")
            error_frame = frame.copy()
            error_counter = error_duration

    # Calculate where to put the text
    frame_height, frame_width = frame.shape[:2]
    top_left = (int(0.02 * frame_width), int(0.05 * frame_height))
    center_pos = (int(0.4 * frame_width), int(0.5 * frame_height))
    error_pos = (int(0.02 * frame_width), int(0.15 * frame_height))

    # Create the error frames to be shown side by side with video
    if error_counter > 0:
        error_display = error_frame.copy()
        cv2.putText(error_display, 'Error Stream', top_left, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(error_display, error_text, error_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        error_counter -= 1
    else:
        error_display = black_frame.copy()
        cv2.putText(error_display, 'Error Stream', top_left, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(error_display, 'No Error', center_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Label the input frame
    saved_vid.write(frame)

    cv2.putText(frame, 'Input Video', top_left, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Combine the frames side by side
    combined_frame = np.hstack((frame, error_display))

    # Write to Output Video
    out.write(combined_frame)

    prev_frame = frame.copy()

    # Exit on 'q' key press
    if keyboard.is_pressed('q'):  # Detect 'q' key globally
        print("Ending Recording")
        break

video.release()
out.release()
saved_vid.release()
cv2.destroyAllWindows()

# Calculate actual FPS
end_time = time.time()
actual_duration = end_time - start_time  # Total duration in seconds
actual_fps = int(frame_count / actual_duration)
print(f"Original FPS: {fps}, Calculated FPS: {actual_fps:.2f}")

# Define the folder to save videos
output_folder = "Error_Videos"
os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist

# Get a list of all existing files in the folder
existing_files = os.listdir(output_folder)

# Find the highest existing video index
video_indices = [
    int(f.split("finalVideo")[1].split(".")[0]) for f in existing_files if f.startswith("finalVideo") and f.endswith(".mp4")
]
next_index = max(video_indices, default=0) + 1

# Define the output video path with the new name
output_video_path = os.path.join(output_folder, f"finalVideo{next_index}.mp4")

if np.abs(actual_fps - fps) > 5:

    # Rewrite video with actual FPS
    print("Rewriting video to get correct FPS after processing")
    cap = cv2.VideoCapture(error_video_path)

    out_corrected = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), actual_fps, output_size)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out_corrected.write(frame)

    cap.release()
    out_corrected.release()

    print(f"Video rewritten with FPS: {actual_fps:.2f}. Saved as '{output_video_path}'")

else:
    time.sleep(2)
    os.rename(error_video_path, output_video_path)
    print(f"Saved intermediate video as final output: '{output_video_path}'")


print("--- %s seconds ---" % (time.time() - codeStart))

# window_size = 10

# # Sliding window sum
# window_sums = [sum(frozen_frame_flags[i:i+window_size]) for i in range(len(frozen_frame_flags) - window_size + 1)]

# # Plotting with window sum (window sum for visualizing concentration otherwise we would see a bunch of 1's in a row)
# plt.figure(figsize=(10, 5))
# plt.plot(window_sums, label='Window Sums')
# #plt.axhline(y=np.mean(window_sums), color='r', linestyle='--', label='Mean')
# plt.xlabel('Index')
# plt.ylabel('Sum of Ones')
# plt.title('Concentration of Detections')
# plt.legend()
# plt.show()
