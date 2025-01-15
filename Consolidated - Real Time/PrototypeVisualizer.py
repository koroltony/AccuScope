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
from source.panoto70fcn import checkPano

input_video_path = 'result_video.mp4'
output_video_path = 'corrected_video.mp4'

codeStart = time.time()
frozen_frame_flags = []

# Initialize video stream
video = cv2.VideoCapture(0)

if not video.isOpened():
    print("Camera could not be opened")
    exit()

fps = int(video.get(cv2.CAP_PROP_FPS))
print(fps)
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
#frame_size = (frame_width, frame_height)
output_size = (2*frame_width, frame_height)

# Create an output stream to hold the prototype video
out = cv2.VideoWriter(input_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, output_size)

# Masks don't make sense for laptop webcam so I will just make one artificially for now

# _, initial_frame = video.read()
# lmask, smask = create_mask(initial_frame)
# video.set(cv2.CAP_PROP_POS_FRAMES, 0)

lmask = np.ones((frame_height, frame_width), dtype=np.uint8)
smask = np.zeros((frame_height, frame_width), dtype=np.uint8)

# Create variables for the error visualization
prev_frame = None
black_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
error_frame = black_frame.copy()
error_text = "No Error"
# This makes the error pop up on the screen for a full second (so you can actually see it)
error_duration = fps
error_counter = 0

print('Press "q" to exit recording')

frame_count = 0
start_time = time.time()

while True:

    ret, frame = video.read()
    if not ret:
        break

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

    black_state = checkBlackFrame(frame)
    if black_state == 1:
        # Create error text and error frame variables to display later
        error_text = f"Dropout Error at {time_stamp:.2f}s"
        print(f"Dropout Error at {time_stamp:.2f}s")
        error_frame = frame.copy()
        error_counter = error_duration

    if checkHighlightsFrame(frame):
        error_text = f"Highlight Shimmer at {time_stamp:.2f}s"
        print(f"Highlight Shimmer at {time_stamp:.2f}s")
        error_frame = frame.copy()
        error_counter = error_duration

    if prev_frame is not None and detect_frozen_frame(prev_frame, frame):
        error_text = f"Frozen Frame at {time_stamp:.2f}s"
        print(f"Frozen Frame at {time_stamp:.2f}s")
        frozen_frame_flags.append(1)
        error_frame = frame.copy()
        error_counter = error_duration

    else:
        frozen_frame_flags.append(0)

    # only check pano if we did not already detect a dropout (because pano flags dropout)
    if black_state != 1:
        pano_state = checkPano(frame, smask, lmask)
        if pano_state == 1:
            error_text = f"Non Minimap Pano-70 Error at {time_stamp:.2f}s"
            print(f"Non Minimap Pano-70 Error at {time_stamp:.2f}s")
            error_frame = frame.copy()
            error_counter = error_duration
        elif pano_state == 2:
            error_text = f"Minimap Pano-70 Error at {time_stamp:.2f}s"
            print(f"Minimap Pano-70 Error at {time_stamp:.2f}s")
            error_frame = frame.copy()
            error_counter = error_duration
        elif pano_state == 3:
            error_text = f"Minimap and Main Screen Pano-70 Error at {time_stamp:.2f}s"
            print(f"Minimap and Main Screen Pano-70 Error at {time_stamp:.2f}s")
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
cv2.destroyAllWindows()

# Calculate actual FPS
end_time = time.time()
actual_duration = end_time - start_time  # Total duration in seconds
actual_fps = int(frame_count / actual_duration)
print(f"Original FPS: {fps}, Calculated FPS: {actual_fps:.2f}")

# Rewrite video with actual FPS
print("Rewriting video to get correct FPS after processing")
cap = cv2.VideoCapture(input_video_path)
out_corrected = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), actual_fps, output_size)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    out_corrected.write(frame)

# Release all resources
cap.release()
out_corrected.release()

print(f"Video rewritten with FPS: {actual_fps:.2f}. Saved as '{output_video_path}'")

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
