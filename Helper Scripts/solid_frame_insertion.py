import cv2
import numpy as np

# Open the video file
# Change filepath to reflect your stored video path
input_path = 'C:/Users/korol/Documents/Capstone TK/short_shimmer.mp4'
output_path = 'C:/Users/korol/Documents/Capstone TK/random_frame_drop.mp4'
cap = cv2.VideoCapture(input_path)

# Get video props to know how big to make the inserted frames:

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Create color frame snippets that are 0.05 seconds long

color_frame_count = int(np.round(fps * 0.05))

# Uncomment to test ground truth:

#-------------------------------

color_frame_count = 1

#-------------------------------

# Choose random places to put color frames

location_array = np.random.randint(0,total_frames-color_frame_count,size = 5)

# Uncomment to test ground truth:

#--------------------------------

location_array = np.array([10,20,30])

#--------------------------------

# Create the VideoWriter for the output video

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Create a color frame (0-255)

color = [0,255,0]

color_frame = np.zeros((height, width, 3), dtype=np.uint8)

for i,val in enumerate(color):
    color_frame[:,:,i] = color[i]

# write frames
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Write frames whenever we get to a flagged location in the stream
    if frame_count in location_array:
        for _ in range(color_frame_count):
            out.write(color_frame)
    else:
        out.write(frame)

    frame_count += 1

cap.release()
out.release()
print("Video saved")
