import cv2

#Start Capture from Capture Card
video = cv2.VideoCapture(1)

#Video Path
output_video_path = 'outputTest.mp4'

fps = int(video.get(cv2.CAP_PROP_FPS))
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_size = (frame_width, frame_height)

#Create Output Stream
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, output_size)

if not video.isOpened():
    print("Stream could not be opened")

while video.isOpened():
    frameRead, frame = video.read()
    if not frameRead:
        break
    
    #Show Video
    cv2.imshow("Arthroscope Feed", frame)

    #Save Video
    out.write(frame)

    #Press q to break
    if cv2.waitKey(25) & 0xFF == ord('q'): # Press 'q' to exit
        break