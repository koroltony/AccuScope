import numpy as np
import cv2
import time
from numba import njit, prange

# --- Original Implementation (OpenCV based) ---
def detect_frozen_frame_opencv(frame1, frame2, threshold=1):
    """Detects if two frames are similar enough to be considered frozen."""
    diff = cv2.absdiff(frame1, frame2)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    diff_pixels = np.count_nonzero(gray_diff > threshold)
    
    # Return true if less than 1% of pixels differ
    return diff_pixels < frame1.shape[0] * frame1.shape[1] * 0.01

# --- Optimized Implementation (Numba based) ---
@njit(parallel=True)
def detect_frozen_frame_numba(gray_frame1, gray_frame2, threshold=1):

    diff_pixels = 0
    rows, cols = gray_frame1.shape
    
    # Manually compute absolute difference
    for i in prange(rows):
        for j in range(cols):
            # Calculate the absolute difference
            diff = abs(gray_frame1[i, j] - gray_frame2[i, j])
            if diff > threshold:
                diff_pixels += 1
    
    # Return true if less than 1% of pixels differ
    return diff_pixels < gray_frame1.shape[0] * gray_frame1.shape[1] * 0.01

# --- Benchmark for time complexity test ---
def benchmark_frozen_frame_detection(num_frames):
    
    # Create fake frames
    frames = np.random.randint(0, 256, (num_frames, 480, 640, 3), dtype=np.uint8)
    
    # Go to grayscale manually to precompile
    gray_frames = np.zeros((num_frames, 480, 640), dtype=np.uint8)
    for i in range(num_frames):
        gray_frames[i] = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
    
    # Timine openCV implementation
    start_time_opencv = time.time()
    for i in range(1, num_frames):
        frame1 = frames[i-1]
        frame2 = frames[i]
        detect_frozen_frame_opencv(frame1, frame2, threshold=1)
    end_time_opencv = time.time()
    opencv_time = end_time_opencv - start_time_opencv
    print(f"OpenCV-based implementation took {opencv_time:.4f} seconds for {num_frames-1} frames.")
    
    # Time Numba implementation
    start_time_numba = time.time()
    for i in range(1, num_frames):
        gray_frame1 = gray_frames[i-1]
        gray_frame2 = gray_frames[i]
        detect_frozen_frame_numba(gray_frame1, gray_frame2, threshold=1)
    end_time_numba = time.time()
    numba_time = end_time_numba - start_time_numba
    print(f"Numba-based implementation took {numba_time:.4f} seconds for {num_frames-1} frames.")
    
    # Compare the results to see if they are the same
    consistent = True
    for i in range(1, num_frames):
        frame1 = frames[i-1]
        frame2 = frames[i]
        result_opencv = detect_frozen_frame_opencv(frame1, frame2, threshold=1)
        gray_frame1 = gray_frames[i-1]
        gray_frame2 = gray_frames[i]
        result_numba = detect_frozen_frame_numba(gray_frame1, gray_frame2, threshold=1)
        if result_opencv != result_numba:
            consistent = False
            print(f"Mismatch at frame {i}: OpenCV={result_opencv}, Numba={result_numba}")
            break
    
    if consistent:
        print("Implementations are consistent and correct")
    else:
        print("There was a mismatch in the results.")

# Run the benchmark
num_frames = 100
benchmark_frozen_frame_detection(num_frames)
