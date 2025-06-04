import cv2
import numpy as np
#import matplotlib.pyplot as plt
import time

start_time = time.time()

def detect_frozen_frame(frame1, frame2, threshold=1):
    """Detects if two frames are similar enough to be considered frozen."""
    diff = cv2.absdiff(frame1, frame2)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    diff_pixels = np.count_nonzero(gray_diff > threshold)
    # print(diff_pixels)
    
    # returns yes if the number of diff pixels is less than the number of frame pixels * x where x = 0.01 or 1%
    return diff_pixels < frame1.shape[0] * frame1.shape[1] * 0.001 


# ----------------- Obsolete test code ----------------------------------------

# Function to identify frozen frame intervals and convert to seconds
def detect_frozen_intervals(window_sums, window_threshold, window_size, fps):
    intervals = []
    in_interval = False
    start = None

    for i, sum_value in enumerate(window_sums):
        if sum_value >= window_threshold:
            if not in_interval:
                start = i
                in_interval = True
        else:
            if in_interval:
                end = i + window_size - 1
                intervals.append((round((start / fps), 4), round((end / fps), 4)))
                in_interval = False

    # Check if the last interval goes till the end
    if in_interval:
        end = len(window_sums) + window_size - 1
        intervals.append((round((start / fps), 4), round((end / fps), 4)))

    return intervals

def main():

    frame_flagged = 135

    for i in range(frame_flagged - 2, frame_flagged + 2):
        prevFrame = cv2.imread(f"C:/Users/korol/Documents/Arthrex Code/ece188a-arthrex/Consolidated - Real Time - Arthroscope/Real_Time_Frames/frame{i}.jpg")
        currentFrame = cv2.imread(f"C:/Users/korol/Documents/Arthrex Code/ece188a-arthrex/Consolidated - Real Time - Arthroscope/Real_Time_Frames/frame{i+1}.jpg")

        if detect_frozen_frame(prevFrame, currentFrame):
            print(f'Error Detected between frames {i} and {i+1}')


    # cap = cv2.VideoCapture('C:/Users/16262/Desktop/arthrex/green flash and lag 3.mp4')
    # currentFrame = 1
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # time_interval = 1/fps  # Time interval in seconds

    # ret, prev_frame = cap.read()
    # frozen_frame_flags = []

    # start_time = time.time()
    # #resized_frame = cv2.resize(current_frame, (960, 540))

    # while ret:
    #     currentFrame = cap.get(cv2.CAP_PROP_POS_FRAMES)

    #     ret, current_frame = cap.read()
    #     if not ret:
    #         break
    #     resized_frame = cv2.resize(current_frame, (960, 540))

    #     current_time = time.time()

    #     if current_time - start_time >= time_interval:
    #         start_time = current_time
    #         if detect_frozen_frame(prev_frame, current_frame):
    #             frozen_frame_flags.append(1)  # Append 1 when frozen frame is detected
    #             print("Frozen frame detected! ", round((currentFrame/fps), 4), 'seconds')
    #         else:
    #             frozen_frame_flags.append(0)  # Append 0 when frozen frame is detected
    #         prev_frame = current_frame

    #     cv2.imshow('Frame', resized_frame)
    #     #cv2.imshow('Frame', current_frame)
    #     if cv2.waitKey(1) == ord('q'): # press q in the cv2 window to break all processes
    #         break

    # cap.release()
    # cv2.destroyAllWindows

    # """
    # # Plot the count of frozen frames over time
    # plt.plot(frozen_frame_flags)
    # plt.plot(frozen_frame_flags, 'o')
    # """

    # # Parameters
    # window_size = 5

    # # Sliding window sum
    # window_sums = [sum(frozen_frame_flags[i:i+window_size]) for i in range(len(frozen_frame_flags) - window_size + 1)]

    # # Plotting with window sum
    # plt.figure(figsize=(10, 5))
    # plt.plot(window_sums, label='Window Sums')
    # #plt.axhline(y=np.mean(window_sums), color='r', linestyle='--', label='Mean')
    # plt.xlabel('Frame Index')
    # plt.ylabel('Sums')
    # plt.title(f'Concentration of Frozen Frame Detections with Window Size = {window_size}')
    # plt.legend()
    # plt.show()

    # window_threshold = window_size * 0.5  # Define a threshold for detection

    # # Detect intervals
    # frozen_intervals = detect_frozen_intervals(window_sums, window_threshold, window_size, fps)

    # # Print intervals in seconds
    # for start, end in frozen_intervals:
    #     #print(f"Frozen frame from {start:.2f} seconds (frame {round(start*fps)}) to {end:.2f} seconds (frame {round(end*fps)})")
    #     print(f"Frozen frame from {start:.2f} seconds to {end:.2f} seconds")

if __name__ == "__main__":
    main()
    #print(time.time() - start_time) # total amount of time script takes