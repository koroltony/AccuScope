import numpy as np
import cv2
import time
from numba import njit

# Generate fake frames and a mask (480x640 with 3 channels)
num_frames = 1000
height, width = 480, 640

frames = np.random.randint(0, 30, (num_frames, height, width, 3), dtype=np.uint8)  # simulate darkish frames
mask = np.random.choice([0, 255], size=(height, width), p=[0.1, 0.9]).astype(np.uint8)  # mostly valid mask

# Original OpenCV+NumPy version
def checkBlackFrame_numpy(frame, mask):
    masked = cv2.bitwise_and(frame, frame, mask=mask)
    return int(~np.any((masked[:, :, 2] > 20) & (masked[:, :, 1] > 20) & (masked[:, :, 0] > 20)))

# Numba version
@njit
def checkBlackFrame_numba(frame, mask):
    height, width = frame.shape[:2]
    for i in range(height):
        for j in range(width):
            if mask[i, j] > 0:
                r = frame[i, j, 2]
                g = frame[i, j, 1]
                b = frame[i, j, 0]
                if r > 20 and g > 20 and b > 20:
                    return 0
    return 1

# Warm-up
_ = checkBlackFrame_numba(frames[0], mask)

numpy_results = []
numba_results = []

# Benchmark NumPy version
start = time.perf_counter()
for i in range(num_frames):
    result = checkBlackFrame_numpy(frames[i], mask)
    numpy_results.append(result)
numpy_time = time.perf_counter() - start

# Benchmark Numba version
start = time.perf_counter()
for i in range(num_frames):
    result = checkBlackFrame_numba(frames[i], mask)
    numba_results.append(result)
numba_time = time.perf_counter() - start

# Comparison
epsilon = 1e-9
mismatches = np.sum(np.array(numpy_results) != np.array(numba_results))

print(f"NumPy total time:   {numpy_time:.6f} seconds")
print(f"Numba total time:   {numba_time:.6f} seconds")
print(f"NumPy per frame:    {numpy_time / num_frames:.6f} seconds")
print(f"Numba per frame:    {numba_time / num_frames:.6f} seconds")
print(f"Speedup factor:     {numpy_time / (numba_time + epsilon):.2f}x")
print(f"Are all results equal? {'Yes' if mismatches == 0 else f'No, mismatches: {mismatches}'}")
