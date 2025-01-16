import cv2
import numpy as np
import time
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

# Get video path:

current_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(current_dir, "source", "Pano to 70 glitch.mp4")


codeStart = time.time()
frozen_frame_flags = []

video = cv2.VideoCapture(video_path)
if not video.isOpened():
    print("Video could not be opened")
    sys.exit('Load Video Properly')

fps = int(video.get(cv2.CAP_PROP_FPS))
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_size = (frame_width, frame_height)
output_size = (frame_width * 2+12, frame_height+8)

# Create an output stream to hold the prototype video
out = cv2.VideoWriter('result_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, output_size)

# Create Masks
_, initial_frame = video.read()
lmask, smask = create_mask(initial_frame)
video.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Create variables for the error visualization
prev_frame = None
black_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
error_frame = black_frame.copy()
error_text = "No Error"
# This makes the error pop up on the screen for a full second (so you can actually see it)
error_duration = fps
error_counter = 0

while video.isOpened():
    current_frame_index = int(video.get(cv2.CAP_PROP_POS_FRAMES))
    time_stamp = current_frame_index / fps
    ret, frame = video.read()
    if not ret:
        break

    # Check Errors
    green_state = checkGreenFrame(frame)
    if green_state == 1:
        # Create error text and error frame variables to display later
        error_text = f"Non Minimap Green Screen Error at {time_stamp:.2f}s"
        print(f"Non Minimap Green Screen Error at {time_stamp:.2f}s")
        error_frame = frame.copy()
        error_counter = error_duration
    elif green_state == 2:
        error_text = f"Minimap Green Screen Error at {time_stamp:.2f}s"
        error_frame = frame.copy()
        print(f"Minimap Green Screen Error at {time_stamp:.2f}s")
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
        pano_state = checkPanoEdge(frame,lmask)
        if pano_state == 1:
            error_text = f"Pano-70 Error at {time_stamp:.2f}s"
            print(f"Non Minimap Pano-70 Error at {time_stamp:.2f}s")
            error_frame = frame.copy()
            error_counter = error_duration


    # Create the error frames to be shown side by side with video
    if error_counter > 0:
        error_display = error_frame.copy()
        cv2.putText(error_display, error_text, (1200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(error_display, 'Error Stream', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        error_counter -= 1
    else:
        error_display = black_frame.copy()
        error_text = "No Error"
        cv2.putText(error_display, error_text, (900, 600), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(error_display, 'Error Stream', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Put the frames next to each other
    cv2.putText(frame, 'Input Video', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Create borders for frames:

    frame_disp = cv2.copyMakeBorder(
                 frame,
                 2,
                 6,
                 3,
                 3,
                 cv2.BORDER_CONSTANT,
                 value=(225,225,225)
              )

    error_display_disp = cv2.copyMakeBorder(
                 error_display,
                 2,
                 6,
                 3,
                 3,
                 cv2.BORDER_CONSTANT,
                 value=(225,225,225)
              )

    combined_frame = np.hstack((frame_disp, error_display_disp))

    # Write to Output Video
    out.write(combined_frame)

    prev_frame = frame.copy()

video.release()
out.release()
cv2.destroyAllWindows()

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
