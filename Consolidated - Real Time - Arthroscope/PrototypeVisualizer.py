import cv2
import numpy as np
import time
import keyboard
import os
import shutil
import matplotlib.pyplot as plt
from source.greenVectorizedSolution import checkGreenFrame_numba as checkGreenFrame
from source.magentaScreen import checkMagentaFrame_numba as checkMagentaFrame
from source.dropoutScreen import checkBlackFrame
from source.highlights import checkHighlightsFrame
from source.lagff15 import detect_frozen_frame
from source.auto_mask import create_mask
from source.panoto70fcn import checkPanoEdge
from source.panoto70fcn import repeated_region
from source.menuDetect import hasMenu

# Define the folder to save videos
output_folder = "Error_Videos"
os.makedirs(output_folder, exist_ok=True)

# Get a list of all existing files in the folder
existing_files = os.listdir(output_folder)

# Find the highest existing video index
video_indices = [
    int(f.split("finalVideo")[1].split(".")[0]) for f in existing_files if f.startswith("finalVideo") and f.endswith(".mp4")
]
next_index = max(video_indices, default=0) + 1

# Define the output video path with the new name
output_video_path = os.path.join(output_folder, f"finalVideo{next_index}.mp4")

codeStart = time.time()
frozen_frame_flags = []

# Initialize video stream

# OBS is 2
# Built in Camera is 1

video = cv2.VideoCapture(2)

# Set the MJPG codec

video.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# set the frame rate:

video.set(cv2.CAP_PROP_FPS,60)

ret, frame = video.read()
if ret:
    print("Processed Frame shape:", frame.shape)

# Make sure the video reader is not repeating frames

video.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not video.isOpened():
    print("Camera could not be opened")
    exit()

fps = int(video.get(cv2.CAP_PROP_FPS))
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(frame_width)
print(frame_height)
output_size = (2*frame_width, frame_height)

# Create an output stream to hold the prototype video

out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 60, output_size)

# ---------------- Create masks for video stream ------------------------------

print('shrink/grow pano-to-70 mask until no "camera-induced" outer edges are visible using "s" and "g"')
print('Press "k" to set both masks when ready')

# shrink and grow variables:
keypress = False
keypresg = False
shrunk_mode = False
# Keep track of frames to update every 10 frames
frame_counter = 0
# update red edges every 10 frames (to reduce latency)
update_interval = 10

# Initial frame read to initialize masks
_, curr_frame = video.read()

# Create kernel for shrinking pano to 70 mask
kernel = np.ones((3, 3), np.uint8)

# Initialize lmask and initial pano mask before shrink operation
lmask = create_mask(curr_frame).astype(np.uint8)
shrunk_mask = lmask.copy()

# Initialize pano to 70 mask
pano_overlay = np.zeros_like(curr_frame)

while True:
    ret, curr_frame = video.read()
    if not ret:
        print("Failed to read frame.")
        break

    # Update lmask until 'k' every 10 frames to avoid bad lag
    if not keyboard.is_pressed('k'):
        if frame_counter % update_interval == 0 or frame_counter == 1:
            lmask = create_mask(curr_frame).astype(np.uint8)

        # Shrunk_mask follows lmask until first shrink/grow operation
        if not shrunk_mode:
            shrunk_mask = lmask.copy()

    # shrink the pano to 70 mask every time 's' is clicked
    if keyboard.is_pressed('s') and not keypress:
        keypress = True
    if keypress and not keyboard.is_pressed('s'):
        shrunk_mask = cv2.erode(shrunk_mask, kernel, iterations=3)
        shrunk_mode = True
        keypress = False

    # grow the pano to 70 mask every time 'g' is clicked
    if keyboard.is_pressed('g') and not keypresg:
        keypresg = True
    if keypresg and not keyboard.is_pressed('g'):
        shrunk_mask = cv2.dilate(shrunk_mask, kernel, iterations=3)
        shrunk_mode = True
        keypresg = False

    # Update edges in pano mask every 10 frames
    if frame_counter % update_interval == 0:
        # Create edges in red that are eventually placed on top of the pano mask
        edges = cv2.Canny(curr_frame, 100, 200)
        masked_edges = cv2.bitwise_and(edges, edges, mask=shrunk_mask)
        edges_red = cv2.merge([
            np.zeros_like(edges),
            np.zeros_like(edges),
            masked_edges
        ])

    # update pano to 70 mask to show the latest size
    pano_green = cv2.merge([
        np.zeros_like(shrunk_mask),
        shrunk_mask,
        np.zeros_like(shrunk_mask)
    ])
    
    # put the red edges on top of the green pano mask
    pano_green = (pano_green * 0.3).astype(np.uint8)
    pano_overlay = cv2.addWeighted(pano_green, 0.5, edges_red, 1.0, 0)

    # Show both masks
    cv2.imshow("Footage Mask", lmask)
    cv2.imshow("Pano-70 Mask", pano_overlay)

    frame_counter += 1

    # Exit when 'k' is clicked
    if cv2.waitKey(1) & 0xFF == ord('k') or keyboard.is_pressed('k'):
        print("Both masks set.")
        break

cv2.destroyAllWindows()


# Compile JIT-enabled codes before starting
_ = checkBlackFrame(curr_frame, lmask)
_ = checkGreenFrame(curr_frame)
_ = checkMagentaFrame(curr_frame)


# ------------------------------------------------------------------------------------------------

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

# ---------------------------------

output_folder2 = "Edge_images"
os.makedirs(output_folder2, exist_ok=True)
# Get a list of all existing files in the folder
existing_files2 = os.listdir(output_folder2)

# Find the highest existing video index
video_indices2 = [
    int(f.split("Edge")[1].split(".")[0]) for f in existing_files2 if f.startswith("Edge") and f.endswith(".mp4")
]
next_index2 = max(video_indices2, default=0) + 1

# Define the output video path with the new name
output_video_path2 = os.path.join(output_folder2, f"Edge{next_index2}.mp4")

savededge_vid = cv2.VideoWriter(output_video_path2, cv2.VideoWriter_fourcc(*'mp4v'), fps, [frame_width,frame_height])



# -------------------------------------------------

# ---------------- Code for frame Grabbing (frozen frame debugging)-----
currentFrame = 1
# path = 'Real_Time_Frames'
# for filename in os.listdir(path):
#         file_path = os.path.join(path, filename)
#         if os.path.isfile(file_path):
#             try:
#                 os.remove(file_path)
#             except OSError as e:
#                 print(f"Error deleting {file_path}: {e}")
# time.sleep(5)
# print("Frames Folder Cleared, Capturing New Footage")

# ----------------------------------------------------------------------q

window_size = 10
frozen_frame_buffer = []

while True:

    ret, frame = video.read()
    if not ret:
        break

    # filePathAndOutputName = os.path.join(path, f'frame{currentFrame}.jpg')
    # cv2.imwrite(filePathAndOutputName, frame)

    currentFrame += 1
    # print(frame_count)
    # print(f"Frame {frame_count} sum: {np.sum(frame)}")
    frame_count += 1

    time_stamp = currentFrame/fps

    # Check Errors
    #print(hasMenu(frame))
    green_state = checkGreenFrame(frame)
    #print(frame.shape)q
    if green_state == 1:
        # Create error text and error frame variables to qdisplay later
        error_text = f"Green Screen Error at {time_stamp:.2f}s and frame: {currentFrame}"
        print(f"Green Screen Error at {time_stamp:.2f}s and frame: {currentFrame}")
        error_frame = frame.copy()
        error_counter = error_duration
    
    # Uncomment if looking for more specific green screen labels
    # elif green_state == 2:
    #     error_text = f"Partial Green Screen Error at {time_stamp:.2f}s and frame: {currentFrame}"
    #     error_frame = frame.copy()
    #     print(f"Partial Green Screen Error at {time_stamp:.2f}s and frame: {currentFrame}")q
    #     error_counter = error_duration

    magenta_state = checkMagentaFrame(frame)
    if magenta_state == 1:
        # Create error text and error frame variables to display later
        error_text = f"Magenta Screen Error at {time_stamp:.2f}s and frame: {currentFrame}"
        print(f"Magenta Screen Error at {time_stamp:.2f}s and frame: {currentFrame}")
        error_frame = frame.copy()
        error_counter = error_duration

    black_state = checkBlackFrame(frame,lmask)
    if black_state == 1:
        # Create error text and error frame variables to display later
        error_text = f"Dropout Error at {time_stamp:.2f}s and frame: {currentFrame}"
        print(f"Dropout Error at {time_stamp:.2f}s and frame: {currentFrame}")
        error_frame = frame.copy()
        error_counter = error_duration

    # if checkHighlightsFrame(frame,lmask):
    #     error_text = f"Highlight Shimmer at {time_stamp:.2f}s and frame: {currentFrame}"q
    #     print(f"Highlight Shimmer at {time_stamp:.2f}s and frame: {currentFrame}")
    #     error_frame = frame.copy()
    #     error_counter = error_duration

    if prev_frame is not None and detect_frozen_frame(prev_frame, frame):
        #print(f"Frozen Frame at {time_stamp:.2f}s and frame: {currentFrame}")
        #error_text = f"Frozen Frame at {time_stamp:.2f}s and frame: {currentFrame}"
        frozen_frame_flags.append(1)
        #error_frame = frame.copy()
        #error_counter = error_duration
        frozen_frame_buffer.append(1)

    else:
        frozen_frame_flags.append(0)
        frozen_frame_buffer.append(0)

    # Only check the sum if the buffer has at least `window_size` elements
    if len(frozen_frame_buffer) >= window_size:
        #print(f"Window sum: {sum(frozen_frame_buffer)}")
        if sum(frozen_frame_buffer) > 4:
            print(f"Frozen Frame Error Detected at {time_stamp:.2f}s (More than 4 in the last {window_size} frames)")
            error_text = f"Frozen Frame Error at {time_stamp:.2f}s and frame: {currentFrame}"
            error_frame = frame.copy()
            error_counter = error_duration

        frozen_frame_buffer.pop(0)

    # Check pano
    pano_state_return = checkPanoEdge(frame,prev_frame,shrunk_mask)
    pano_state = pano_state_return[0]
    edge_frame = np.uint8(pano_state_return[1])
    edge_frame = cv2.merge((edge_frame, edge_frame, edge_frame))
    
    #uncomment to use edge-detection for Pano to 70 detection
    
    # if pano_state == 1:
    #     error_text = f"Pano-70 Error at {time_stamp:.2f}s and frame: {currentFrame}"
    #     print(f"Pano-70 Error at {time_stamp:.2f}s and frame: {currentFrame}")
    #     error_frame = frame.copy()
    #     error_counter = error_duration
    
    # Check pano autocorrelation
    repeated_return = repeated_region(frame)
    if repeated_return == 1:
        error_text = f"Pano-70 (repeated region) Error at {time_stamp:.2f}s and frame: {currentFrame}"
        print(f"Pano-70 (repeated region) Error at {time_stamp:.2f}s and frame: {currentFrame}")
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

    if not savededge_vid.isOpened():
        print("Error: VideoWriter failed to open.")

    if edge_frame is not None:
        savededge_vid.write(edge_frame)
    else:
        print("Warning: edge_frame is None at frame", currentFrame)

    #print(type(frame), frame.shape)
    #print(type(edge_frame), edge_frame.shape)

    # Label the input frame
    saved_vid.write(frame)
    savededge_vid.write(edge_frame)

    cv2.putText(frame, 'Input Video', top_left, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Combine the frames side by side
    combined_frame = np.hstack((frame, error_display))

    # Write to Output Video
    out.write(combined_frame)

    prev_frame = frame.copy()

    #print("Frame size:", frame.shape if frame is not None else "None")
    #print("Edge frame size:", edge_frame.shape if edge_frame is not None else "None")


    # Exit on 'q' key press
    if keyboard.is_pressed('q'):
        print("Ending Recording")
        break


video.release()
out.release()
saved_vid.release()
savededge_vid.release()
cv2.destroyAllWindows()

# Calculate actual FPS
end_time = time.time()
actual_duration = end_time - start_time
actual_fps = int(frame_count / actual_duration)
print(f"Saved Video at {output_video_path}")
print(f"Original FPS: {fps}, Calculated FPS: {actual_fps:.2f}")

if np.abs(actual_fps - fps) > 5:
    # Print warning:
    print(f'video at {output_video_path} experienced latency and will have FPS discrepancy > 5FPS')

print("--- %s seconds ---" % (time.time() - codeStart))

window_size = 10

# Sliding window sum
window_sums = [sum(frozen_frame_flags[i:i+window_size]) for i in range(len(frozen_frame_flags) - window_size + 1)]

# Plotting with window sum (window sum for visualizing concentration otherwise we would see a bunch of 1's in a row)
plt.figure(figsize=(10, 5))
plt.plot(window_sums, label='Window Sums')
#plt.axhline(y=np.mean(window_sums), color='r', linestyle='--', label='Mean')
plt.xlabel('Index')
plt.ylabel('Sum of Ones')
plt.title('Concentration of Detections')
plt.legend()
plt.show()
