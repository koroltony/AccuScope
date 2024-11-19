import cv2
import numpy as np

# Open the video file
input_path = 'C:/Users/korol/Documents/Capstone TK/short_shimmer.mp4'
output_path = 'C:/Users/korol/Documents/Capstone TK/output_with_magenta_frames.mp4'
cap = cv2.VideoCapture(input_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Define the midpoint and chunk length for adding magenta lines
# Choose random places to put black frames

magenta_frame_count = fps * 0.05

location_array = np.random.randint(0,total_frames-magenta_frame_count,size = 5)



# Create the VideoWriter for the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Magenta color in BGR
magenta = (255, 0, 255)
line_thickness = 3

# Calculate line positions for 20 evenly spaced horizontal lines
line_positions = np.linspace(0, height, 20, dtype=int)

# Process and write frames
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Add magenta lines to frames in the specified chunk at the midpoint
    if frame_count in location_array:
        for y in line_positions:
            cv2.line(frame, (0, y), (width, y), magenta, line_thickness)

    # Write the frame to the output video
    out.write(frame)
    frame_count += 1

# Release resources
cap.release()
out.release()
print("Video saved successfully with magenta lines!")