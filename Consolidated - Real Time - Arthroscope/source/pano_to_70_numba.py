import numpy as np
import cv2
from numba import njit
import time
from scipy.signal import correlate, find_peaks

# Generate test data
num_frames = 5000
height, width = 480, 640
frames = np.random.randint(0, 255, (num_frames, height, width, 3), dtype=np.uint8)

# -------------------- NumPy Implementation -----------------------------------

def repeated_region_numpy(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    center = img.shape[1] // 2
    region_avg = np.mean(img[20:400, center - 5:center + 5], axis=1)
    norm = (region_avg - np.mean(region_avg)) / (np.std(region_avg) or 1)
    autocorr = np.correlate(norm, norm, mode='full')[len(norm)-1:]
    autocorr /= np.max(autocorr) or 1
    autocorr_log = np.log1p(np.abs(autocorr))
    peaks = np.where((autocorr_log[1:-1] > autocorr_log[:-2]) & 
                     (autocorr_log[1:-1] > autocorr_log[2:]) & 
                     (autocorr_log[1:-1] > 0.3))[0]
    return int(len(peaks) >= 3)

# -------------------- Scipy Implementation -----------------------------------

def repeated_region_scipy(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    center = img.shape[1] // 2
    region_avg = np.mean(img[20:400, center - 5:center + 5], axis=1)
    norm = (region_avg - np.mean(region_avg)) / (np.std(region_avg) or 1)
    autocorr = correlate(norm, norm, mode='full',method='fft')
    autocorr /= np.max(autocorr) or 1
    autocorr_log = np.log1p(np.abs(autocorr))
    peaks,_ = find_peaks(autocorr_log, prominence=0.3)
    return int(len(peaks) >= 3)

# --------------------- Numba Implementation  ---------------------------------

@njit
def autocorr_and_log(vec):
    n = vec.shape[0]
    mean_val = np.mean(vec)
    std_val = np.std(vec) or 1.0
    norm_vec = (vec - mean_val) / std_val
    autocorr = np.zeros(n)
    for lag in range(n):
        for i in range(n - lag):
            autocorr[lag] += norm_vec[i] * norm_vec[i + lag]
    maxval = np.max(autocorr)
    if maxval > 0:
        autocorr /= maxval
    for i in range(n):
        autocorr[i] = np.log1p(np.abs(autocorr[i]))
    return autocorr

@njit
def find_peaks_numba(signal, threshold=0.3, laplacian_threshold=0.005):
    count = 0
    n = len(signal)
    
    for i in range(2, n - 2):
        center = signal[i]
        
        if center <= threshold:
            continue

        neighbors_avg = (signal[i - 2] + signal[i - 1] + signal[i + 1] + signal[i + 2]) / 4.0
        laplacian = center - neighbors_avg
        
        if laplacian > laplacian_threshold:
            count += 1
            
    return count

def repeated_region_numba(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    center = img.shape[1] // 2
    region_avg = np.mean(img[20:400, center - 5:center + 5], axis=1).astype(np.float32)
    autocorr_log = autocorr_and_log(region_avg)
    peak_count = find_peaks_numba(autocorr_log)
    
    # if int(peak_count >= 3):
    #     plt.figure()
    #     plt.plot(autocorr_log)
    return int(peak_count >= 3)

# -----------------------------------------------------------------------------

# Benchmark function to time and evaluate different implementations

def benchmark(name, func):
    start = time.perf_counter()
    results = [func(frame) for frame in frames]
    duration = time.perf_counter() - start
    return name, results, duration

# Pre-compile and run tests

_ = repeated_region_numba(frames[0])

results_numpy = benchmark("NumPy", repeated_region_numpy)
results_scipy = benchmark("SciPy", repeated_region_scipy)
results_numba = benchmark("Numba", repeated_region_numba)

# Print results of the benchmark test
mismatches1 = np.sum(np.array(results_numpy[1]) != np.array(results_numba[1]))
mismatches2 = np.sum(np.array(results_numpy[1]) != np.array(results_scipy[1]))

print(f"NumPy total time: {results_numpy[2]:.4f} s")
print(f"Numba total time: {results_numba[2]:.4f} s")
print(f"Scipy total time: {results_scipy[2]:.4f} s")
print(f"Numba Speedup factor:   {results_numpy[2] / (results_numba[2] + 1e-9):.2f}x")
print(f"Scipy Speedup factor:   {results_numpy[2] / (results_scipy[2] + 1e-9):.2f}x")
print(f"Numpy and Numba Output match:     {'Yes' if mismatches1 == 0 else f'No, {mismatches1} mismatches'}")
print(f"Numpy and Scipy Output match:     {'Yes' if mismatches2 == 0 else f'No, {mismatches2} mismatches'}")
