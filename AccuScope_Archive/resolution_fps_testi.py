import cv2
import os

# Define the folder to save videos
output_folder = "testVideos"
os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist

# Get a list of all existing files in the folder
existing_files = os.listdir(output_folder)

# Find the highest existing video index
video_indices = [
    int(f.split("savedVideo")[1].split(".")[0]) for f in existing_files if f.startswith("savedVideo") and f.endswith(".mp4")
]
next_index = max(video_indices, default=0) + 1

# Define the output video path with the new name
output_video_path = os.path.join(output_folder, f"savedVideo{next_index}.mp4")

# Initialize the webcam
cap = cv2.VideoCapture(0)  # 0 is the default device ID for the primary webcam

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set the desired resolution
desired_width = 1920  # Set the desired width
desired_height = 1080  # Set the desired height
cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

# Set the frame rate
cap.set(cv2.CAP_PROP_FPS, 60)

# Get the actual width and height of the video frames
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
saved_vid = cv2.VideoWriter(output_video_path, fourcc, 60.0, (frame_width, frame_height))

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Write the frame to the output file
    saved_vid.write(frame)

    # Display the frame
    cv2.imshow('Webcam Video', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources
cap.release()
saved_vid.release()
cv2.destroyAllWindows()
