import cv2
import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt
from source_update.greenVectorizedSolution import checkGreenFrame_numba as checkGreenFrame
from source_update.magentaScreen import checkMagentaFrame_numba as checkMagentaFrame
from source_update.dropoutScreen import checkBlackFrame_numba as checkBlackFrame
from source_update.highlights import checkHighlightsFrame
from source_update.lagff15 import detect_frozen_frame
from source_update.auto_mask import create_mask
from source_update.panoto70fcn import checkPanoEdge
from source_update.panoto70fcn import repeated_region_numpy_illustrative as repeated_region
from source_update.menuDetect import hasMenu

# Get video path:

current_dir = os.path.dirname(os.path.abspath(__file__))
#video_path = "C:/Users/korol/Documents/Arthrex Code/ece188a-arthrex/Consolidated/source_update/stitched_test_video.mp4"
video_path = "C:/Users/korol/Documents/Arthrex Code/ece188a-arthrex/Consolidated - Real Time - Arthroscope/Raw_Videos/RawVideo232.mp4"


codeStart = time.time()
frozen_frame_flags = []

video = cv2.VideoCapture(video_path)
if not video.isOpened():
    print("Video could not be opened")
    sys.exit('Load Video Properly')

fps = int(video.get(cv2.CAP_PROP_FPS))
frame_width = 640
frame_height = 480
frame_size = (frame_width, frame_height)
output_size = (2*frame_width, frame_height)

# Create an output stream to hold the prototype video
out = cv2.VideoWriter('result_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, output_size)

# Create Masks
_, initial_frame = video.read()
initial_frame = cv2.resize(initial_frame, (640, 480), interpolation=cv2.INTER_LINEAR)
lmask = create_mask(initial_frame).astype(np.uint8)
video.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Create variables for the error visualization
prev_frame = None
black_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
error_frame = black_frame.copy()
error_text = "No Error"
# This makes the error pop up on the screen for a full second (so you can actually see it)
error_duration = fps
error_counter = 0

window_size = 10
frozen_frame_buffer = []

# Compile JIT-enabled codes before starting
_ = checkBlackFrame(initial_frame, lmask)
_ = checkGreenFrame(initial_frame)
_ = checkMagentaFrame(initial_frame)

while video.isOpened():
    current_frame_index = int(video.get(cv2.CAP_PROP_POS_FRAMES))
    time_stamp = current_frame_index / fps
    ret, frame = video.read()

    if not ret:
        break
    
    if frame.shape[1] != 640 or frame.shape[0] != 480:
        frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)

    # Check Errors
    #print(hasMenu(frame))
    green_state = checkGreenFrame(frame)
    #print(frame.shape)q
    if green_state == 1:
        # Create error text and error frame variables to display later
        error_text = f"Green Screen Error at {time_stamp:.2f}s"
        print(f"Green Screen Error at {time_stamp:.2f}s")
        error_frame = frame.copy()
        error_counter = error_duration
        
    elif green_state == 2:
        # Create error text and error frame variables to qdisplay later
        error_text = f"Corner Green Screen Error at {time_stamp:.2f}s"
        print(f"Corner Green Screen Error at {time_stamp:.2f}s")
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
    #     error_text = f"Highlight Shimmer at {time_stamp:.2f}s and frame: {currentFrame}"q
    #     print(f"Highlight Shimmer at {time_stamp:.2f}s and frame: {currentFrame}")
    #     error_frame = frame.copy()
    #     error_counter = error_duration

    if prev_frame is not None and detect_frozen_frame(prev_frame, frame):
        #print(f"Frozen Frame at {time_stamp:.2f}s")
        #error_text = f"Frozen Frame at {time_stamp:.2f}s"
        #frozen_frame_flags.append(1)
        #error_frame = frame.copy()
        #error_counter = error_duration
        frozen_frame_buffer.append(1)

    else:
        #frozen_frame_flags.append(0)
        frozen_frame_buffer.append(0)

    # Only check the sum if the buffer has at least `window_size` elements

    if len(frozen_frame_buffer) >= window_size:
        #print(f"Window sum: {sum(frozen_frame_buffer)}")
        if sum(frozen_frame_buffer) > 4:
            print(f"Frozen Frame Error Detected at {time_stamp:.2f}s (More than 4 in the last {window_size} frames)")
            error_text = f"Frozen Frame Error at {time_stamp:.2f}s"
            error_frame = frame.copy()
            error_counter = error_duration

        frozen_frame_buffer.pop(0)

    # Check pano
    # pano_state_return = checkPanoEdge(frame,prev_frame,shrunk_mask)
    # pano_state = pano_state_return[0]
    # edge_frame = np.uint8(pano_state_return[1])
    # edge_frame = cv2.merge((edge_frame, edge_frame, edge_frame))
    
    #uncomment to use edge-detection for Pano to 70 detection
    
    # if pano_state == 1:
    #     error_text = f"Pano-70 Error at {time_stamp:.2f}s and frame: {currentFrame}"
    #     print(f"Pano-70 Error at {time_stamp:.2f}s and frame: {currentFrame}")
    #     error_frame = frame.copy()
    #     error_counter = error_duration
    
    # Check pano autocorrelation
    repeated_return = repeated_region(frame)
    if repeated_return == 1:
        error_text = f"Pano-70 (repeated region) Error at {time_stamp:.2f}s"
        print(f"Pano-70 (repeated region) Error at {time_stamp:.2f}s")
        error_frame = frame.copy()
        error_counter = error_duration
        
    
    prev_frame = frame.copy()


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

    cv2.putText(frame, 'Input Video', top_left, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Combine the frames side by side
    combined_frame = np.hstack((frame, error_display))

    # Write to Output Video
    out.write(combined_frame)

video.release()
out.release()
cv2.destroyAllWindows()

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
