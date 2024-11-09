import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time()

def detect_frozen_frame(frame1, frame2, threshold=1):
    """Detects if two frames are similar enough to be considered frozen."""
    diff = cv2.absdiff(frame1, frame2)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    diff_pixels = np.count_nonzero(gray_diff > threshold)
    return diff_pixels < frame1.shape[0] * frame1.shape[1] * 0.001 # returns yes if the number of diff pixels is less than the number of frame pixels * x where x = 0.001 or 0.1%

def main():

    cap = cv2.VideoCapture('green flash lag 3.mp4')
    currentFrame = 1
    fps = cap.get(cv2.CAP_PROP_FPS)  
    time_interval = 1/fps  # Time interval in seconds

    ret, prev_frame = cap.read()
    frozen_frame_flags = []  

    start_time = time.time()
    #resized_frame = cv2.resize(current_frame, (960, 540))

    while ret:
        currentFrame = cap.get(cv2.CAP_PROP_POS_FRAMES)

        ret, current_frame = cap.read()
        if not ret:
            break
        resized_frame = cv2.resize(current_frame, (960, 540))

        current_time = time.time()

        if current_time - start_time >= time_interval:
            start_time = current_time
            if detect_frozen_frame(prev_frame, current_frame):
                frozen_frame_flags.append(1)  # Append 1 when frozen frame is detected
                print("Frozen frame detected! ", round((currentFrame/fps), 4), 'seconds')
            else: 
                frozen_frame_flags.append(0)  # Append 0 when frozen frame is detected
            prev_frame = current_frame

        cv2.imshow('Frame', resized_frame)
        #cv2.imshow('Frame', current_frame)
        if cv2.waitKey(1) == ord('q'): # press q in the cv2 window to break all processes
            break

    cap.release()
    cv2.destroyAllWindows

    """
    # Plot the count of frozen frames over time
    plt.plot(frozen_frame_flags)
    plt.plot(frozen_frame_flags, 'o')
    """

    # Parameters
    window_size = 10

    # Sliding window sum
    window_sums = [sum(frozen_frame_flags[i:i+window_size]) for i in range(len(frozen_frame_flags) - window_size + 1)]

    # Plotting with window sum
    plt.figure(figsize=(10, 5))
    plt.plot(window_sums, label='Window Sums')
    #plt.axhline(y=np.mean(window_sums), color='r', linestyle='--', label='Mean')
    plt.xlabel('Index')
    plt.ylabel('Sum of Ones')
    plt.title('Concentration of Ones in Binary Array')
    plt.legend()
    plt.show()

    """
    plt.xlabel('Frame')
    plt.ylabel('Frozen Frame Detection')
    plt.title('Frozen Frame Detection Over Time')
    plt.ylim([0, 1.5])
    plt.show()
    """

if __name__ == "__main__":
    main()
    print(time.time() - start_time) # total amount of time script takes