import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import signal, ndimage
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from PIL import Image

def centerAndPad(small, large):
    heightS, widthS = small.shape
    heightL, widthL = large.shape
    zeroground = np.zeros((heightL, widthL), dtype=small.dtype)
    xoffset = (widthL - widthS) // 2
    yoffset = (heightL - heightS) // 2
    zeroground[yoffset:yoffset + heightS, xoffset:xoffset + widthS] = small
    return zeroground

def correlate2D(ref, target):
  # frequency domain conversion
  ref_fft = fft2(ref)
  target_fft = fft2(target)
  cross_corr_fft = ref_fft * np.conj(target_fft)

  # time domain conversion
  cross_corr = np.abs(ifft2(cross_corr_fft))

  # find max index of cross correlation
  return np.argmax(cross_corr)

target = cv2.imread('OpticalCharacterRecognition/menuImages/whiteBalance.jpg', cv2.IMREAD_GRAYSCALE)
frame = cv2.imread('OpticalCharacterRecognition/testImages/frame2.jpg', cv2.IMREAD_GRAYSCALE)

preprocessedTarget = centerAndPad(target, frame)
nothingMatrix = np.zeros(frame.shape)

#plt.imshow(target, cmap='gray')
#plt.imshow(frame, cmap='gray')
#plt.imshow(preprocessedTarget, cmap='gray')

randomNoise = np.random.randint(0, 256, frame.shape, dtype=np.uint8)
#plt.imshow(randomNoise, cmap='gray')

print(f'Correlation with "White Balance":           {correlate2D(frame, preprocessedTarget)}')
print(f'Correlation with All Black:                 {correlate2D(frame, nothingMatrix)}')
print(f'Correlation with Uniform Noise:             {correlate2D(frame, randomNoise)}')

#Run a Monte Carlo Simulation
#Distribution of the Mean from any Distribution in Normal
N = 10
averageCorrelation = np.zeros(N)
for trial in range(N):
   randomNoise = np.random.randint(0, 256, frame.shape, dtype=np.uint8)
   averageCorrelation[trial] = (correlate2D(frame, randomNoise))
print(f'Mean correlation for noise in {N} trials:   {np.mean(averageCorrelation)}')
plt.show()

#   1006790.0 For 1 Simulation
#   1484528.9 For 10 Simulations
#   1078870.6 For 100 Simulations
#   1020085.578 For 1000 Simulations