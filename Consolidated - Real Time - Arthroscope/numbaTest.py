import numpy as np
import cv2
import time
from numba import njit

# Generate fake frames and a mask (480x640 with 3 channels)
num_frames = 1000
height, width = 480, 640

# Frames and masks for simulation

frames = np.random.randint(0, 30, (num_frames, height, width, 3), dtype=np.uint8)
mask = np.random.choice([0, 255], size=(height, width), p=[0.1, 0.9]).astype(np.uint8)

# --- NumPy with mask ---
def checkBlackFrame_numpy(frame, mask):
    masked = cv2.bitwise_and(frame, frame, mask=mask)
    return int(~np.any((masked[:, :, 2] > 20) & (masked[:, :, 1] > 20) & (masked[:, :, 0] > 20)))

# --- Numba with mask ---
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

# --- NumPy without mask ---
def checkBlackFrame_numpy_nomask(frame):
    return int(~np.any((frame[:, :, 2] > 20) & (frame[:, :, 1] > 20) & (frame[:, :, 0] > 20)))

# --- Numba without mask ---
@njit
def checkBlackFrame_numba_nomask(frame):
    height, width = frame.shape[:2]
    for i in range(height):
        for j in range(width):
            r = frame[i, j, 2]
            g = frame[i, j, 1]
            b = frame[i, j, 0]
            if r > 20 and g > 20 and b > 20:
                return 0
    return 1

# Warm-up
_ = checkBlackFrame_numba(frames[0], mask)
_ = checkBlackFrame_numba_nomask(frames[0])

# --- Benchmarks ---

numpy_results = []
numba_results = []
numpy_nomask_results = []
numba_nomask_results = []

# NumPy with mask
start = time.perf_counter()
for i in range(num_frames):
    result = checkBlackFrame_numpy(frames[i], mask)
    numpy_results.append(result)
numpy_time = time.perf_counter() - start

# Numba with mask
start = time.perf_counter()
for i in range(num_frames):
    result = checkBlackFrame_numba(frames[i], mask)
    numba_results.append(result)
numba_time = time.perf_counter() - start

# NumPy without mask
start = time.perf_counter()
for i in range(num_frames):
    result = checkBlackFrame_numpy_nomask(frames[i])
    numpy_nomask_results.append(result)
numpy_nomask_time = time.perf_counter() - start

# Numba without mask
start = time.perf_counter()
for i in range(num_frames):
    result = checkBlackFrame_numba_nomask(frames[i])
    numba_nomask_results.append(result)
numba_nomask_time = time.perf_counter() - start

# --- Results Comparison ---
epsilon = 1e-9
mismatches_mask = np.sum(np.array(numpy_results) != np.array(numba_results))
mismatches_nomask = np.sum(np.array(numpy_nomask_results) != np.array(numba_nomask_results))

print("\n--- Masked Version ---")
print(f"NumPy (mask) total time:   {numpy_time:.6f} seconds")
print(f"Numba (mask) total time:   {numba_time:.6f} seconds")
print(f"NumPy (mask) per frame:    {numpy_time / num_frames:.6f} seconds")
print(f"Numba (mask) per frame:    {numba_time / num_frames:.6f} seconds")
print(f"Speedup (NumPy/Numba):     {numpy_time / (numba_time + epsilon):.2f}x")
print(f"Results match?             {'Yes' if mismatches_mask == 0 else f'No, mismatches: {mismatches_mask}'}")

print("\n--- No Mask Version ---")
print(f"NumPy (no mask) total time:   {numpy_nomask_time:.6f} seconds")
print(f"Numba (no mask) total time:   {numba_nomask_time:.6f} seconds")
print(f"NumPy (no mask) per frame:    {numpy_nomask_time / num_frames:.6f} seconds")
print(f"Numba (no mask) per frame:    {numba_nomask_time / num_frames:.6f} seconds")
print(f"Speedup (NumPy/Numba):        {numpy_nomask_time / (numba_nomask_time + epsilon):.2f}x")
print(f"Results match?                {'Yes' if mismatches_nomask == 0 else f'No, mismatches: {mismatches_nomask}'}")

print("\n--- Mask vs No Mask (Numba) ---")
print(f"Speedup from masking (Numba): {numba_nomask_time / (numba_time + epsilon):.2f}x")
print(f"Speedup from masking (NumPy): {numpy_nomask_time / (numpy_time + epsilon):.2f}x")
