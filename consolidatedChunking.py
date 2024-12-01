import cv2
import os
import numpy as np
import time
import matplotlib.pyplot as plt
import subprocess
import sys
from chunkVectorSearch import chunkVectorSearch, capture_frames

'''
from Highlights.highlights import checkHighlightsFrame
from Frozen.lagff15 import detect_frozen_frame
'''
from HelperScripts.auto_mask import create_mask
from panoto70.panoChunkFcn import chunkPano

codeStart = time.time()

#OpenCV Declaration
video = cv2.VideoCapture("panoto70/Pano to 70 glitch.mp4")
totalFrames = video.get(cv2.CAP_PROP_FRAME_COUNT)
fps = video.get(cv2.CAP_PROP_FPS)
time_interval = 1/fps
tempTime = time.time()

# ---------- Create Mask From First Video Frame -----------------------

# First, read the starting frame
frame_read, frame = video.read()

frame = cv2.resize(frame,(1920,1080))

# Next, create masks for the main image and minimap: (lmask is main, smask is minimap)
lmask, smask = create_mask(frame)

# Reset the video frame grabber to start at frame 0
video.set(cv2.CAP_PROP_POS_FRAMES, 0)

# ----------------------------------------------------------------------

if not video.isOpened():
    print("Video could not be opened")

gmain_timestamps = []
gminimap_timestamps = []

mmain_timestamps = []
mminimap_timestamps = []

b_timestamps = []

pano_main_errors = []
pano_minimap_errors = []

while video.isOpened():
    # Check Pano-70
    chunk = capture_frames(video, 40)

    # Ensure chunk is valid
    if chunk.size == 0:
        break

    # Get current frame position
    currentFrame = video.get(cv2.CAP_PROP_POS_FRAMES)

    # Get offsets for main video and minimap
    gOffsetsl, gOffsetss, mOffsetsl, mOffsetss, bOffsets = chunkVectorSearch(chunk, smask, lmask)

    # Calculate timestamps
    if gOffsetsl.size > 0:
        timeStampsl = (currentFrame + gOffsetsl) / fps
        gmain_timestamps.extend(timeStampsl)

    if gOffsetss.size > 0:
        timeStampss = (currentFrame + gOffsetss) / fps
        gminimap_timestamps.extend(timeStampss)

    if mOffsetsl.size > 0:
        timeStampsl = (currentFrame + mOffsetsl) / fps
        mmain_timestamps.extend(timeStampsl)

    if mOffsetss.size > 0:
        timeStampss = (currentFrame + mOffsetss) / fps
        mminimap_timestamps.extend(timeStampss)

    if bOffsets.size > 0:
        timeStamps = (currentFrame + bOffsets) / fps
        b_timestamps.extend(timeStamps)

   # Pano-to-70 errors
    pano_minimap_timestamps, pano_main_timestamps = chunkPano(chunk, smask, lmask, fps, currentFrame)
    pano_main_errors.extend(pano_main_timestamps)
    pano_minimap_errors.extend(pano_minimap_timestamps)

# Print results
print("\nGreen Main Video Errors (seconds):", [round(t, 4) for t in gmain_timestamps])
print("\nGreen Minimap Errors (seconds):", [round(t, 4) for t in gminimap_timestamps])

print("\nMagenta Main Video Errors (seconds):", [round(t, 4) for t in mmain_timestamps])
print("\nMagenta Minimap Errors (seconds):", [round(t, 4) for t in mminimap_timestamps])

print("\nDropout Video Errors (seconds):", [round(t, 4) for t in b_timestamps])

print("\nPano-to-70 Main Screen Errors (seconds):", pano_main_errors)
print("\nPano-to-70 Minimap Errors (seconds):", pano_minimap_errors)

print("\n--- %s seconds ---" % (time.time() - codeStart))