import numpy as np
import cv2
from numba import njit
import time
from scipy.signal import correlate, find_peaks

# Generate test data
num_frames = 1000
height, width = 480, 640
frames = np.random.randint(0, 255, (num_frames, height, width, 3), dtype=np.uint8)

# -------------------- NumPy Implementation -----------------------------------

def find_peaks_numpy(signal, prominence=0.2):
    peaks = []
    for i in range(1, len(signal) - 1):
        if signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
            # Look left
            left_min = signal[i]
            for j in range(i - 1, -1, -1):
                if signal[j] > signal[i]:
                    break
                left_min = min(left_min, signal[j])
            # Look right
            right_min = signal[i]
            for j in range(i + 1, len(signal)):
                if signal[j] > signal[i]:
                    break
                right_min = min(right_min, signal[j])
            # Prominence check
            if signal[i] - max(left_min, right_min) >= prominence:
                peaks.append(i)
    return np.array(peaks)

# NumPy version
def repeated_region_numpy(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    center = img.shape[1] // 2
    region_avg = np.mean(img[20:400, center - 5:center + 5], axis=1)
    norm = (region_avg - np.mean(region_avg)) / (np.std(region_avg) or 1)
    autocorr = np.correlate(norm, norm, mode='full')[len(norm)-1:]
    autocorr /= np.max(autocorr) or 1
    autocorr_log = np.log1p(np.abs(autocorr))
    # Compare surrounding values to identify peaks in autocorrelation spectrum
    peaks = find_peaks_numpy(autocorr_log, prominence=0.2)
    
    # if int(len(peaks)) >= 3:
    #     plt.figure()
    #     plt.plot(autocorr_log)
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
def find_peaks_numba(signal, threshold=0.2, laplacian_threshold=0.01, window=2):
    count = 0
    n = len(signal)

    for i in range(window, n - window):
        center = signal[i]
        if center <= threshold:
            continue

        neighbors_sum = 0.0
        for offset in range(1, window + 1):
            neighbors_sum += signal[i - offset] + signal[i + offset]

        neighbors_avg = neighbors_sum / (2 * window)
        laplacian = center - neighbors_avg

        if laplacian > laplacian_threshold:
            count += 1

    return count

def repeated_region_numba(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    center = img.shape[1] // 2
    region_avg = np.mean(img[20:400, center - 5:center + 5], axis=1).astype(np.float32)
    autocorr_log = autocorr_and_log(region_avg)
    peak_count = find_peaks_numba(autocorr_log,window=5)
    
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
