import cv2
import numpy as np

# Function to add magenta lines to a frame
def add_magenta_lines(frame, line_positions, line_thickness=3):
    magenta = (255, 0, 255)  # Magenta color in BGR
    for y in line_positions:
        cv2.line(frame, (0, y), (frame.shape[1], y), magenta, line_thickness)
    return frame

# Function to stitch videos with black/green frames and magenta line errors
def create_test_video(video_files, output_path, insert_frame_color, magenta_frame_indices, fps=30):
    # Get properties from the first video
    first_video = cv2.VideoCapture(video_files[0])
    width = int(first_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(first_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first_video.release()

    frame_size = (width, height)
    black_frame = np.zeros((height, width, 3), dtype=np.uint8)
    green_frame = np.full((height, width, 3), (0, 255, 0), dtype=np.uint8)

    # Define VideoWriter
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)

    # Prepare magenta line positions
    line_positions = np.linspace(0, height, 20, dtype=int)

    # Track current frame index
    current_frame_index = 0

    # Process each video
    for video_file in video_files:
        cap = cv2.VideoCapture(video_file)

        if not cap.isOpened():
            print(f"Failed to open video file: {video_file}")
            continue

        while True:
            ret, frame = cap.read()
            if not ret:
                print(current_frame_index)
                print(f"Finished processing video file: {video_file}")
                break
            frame = cv2.resize(frame, (width, height))

            # Add magenta lines to specified frames
            if current_frame_index in magenta_frame_indices:
                frame = add_magenta_lines(frame, line_positions)

            out.write(frame)
            current_frame_index += 1

        cap.release()

        # Insert the specified frame color between videos
        transition_frame = green_frame if insert_frame_color == "green" else black_frame
        out.write(transition_frame)
        current_frame_index += 1

    out.release()

# Usage example
video_files = [
    "C:/Users/korol/Documents/Capstone TK/Pano to 70 glitch.mp4",
    "C:/Users/korol/Documents/Capstone TK/green flash and lag.mp4",
    "C:/Users/korol/Documents/Capstone TK/HDR shimmer coracoid.mp4"
]
output_path = "C:/Users/korol/Documents/Capstone TK/stitched_test_video.mp4"

# Specify the absolute frame indices where magenta lines should appear
magenta_frame_indices = [30, 150, 300, 450, 600]

create_test_video(
    video_files=video_files,
    output_path=output_path,
    insert_frame_color="black",  # Change to "green" for green screen inserts
    magenta_frame_indices=magenta_frame_indices
)

print("Test video created successfully!")
