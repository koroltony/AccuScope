import cv2
import numpy as np
import time
# TODO:

'''
1) See about isolating the minimap and circular part since that's all we care about
2) See if Zion's implementation is any better than mine and evaluate what's next
3) Look into overlap to make sure we have full frame coverage at edges of chunks
4) look into chunk size and why it might be useful to control
5) Figure out if chunking is even necessary at all (maybe not in this case, but maybe for lag and stuff)
6) Doing chunks means we can't compare with a previous chunk. See if can do that efficiently
'''

start_time = time.time()

#Pass in a VideoCapture Object, starting time and ending time and function will return every frame

def capture_frames(video,chunk_size=40):

    # Preallocate array to hold the frames
    frames = np.zeros((chunk_size, 1080, 1920, 3), dtype=np.uint8)

    # Count variable to keep track of frames opened in the chunk:

    frame_count = 0


    # Check if Video Was Opened Successfully
    if not video.isOpened():
        print("Video could not be opened")

    # Save only the part of the video in the current chunk

    for i in range(chunk_size):

        # Read the frame

        frame_read, frame = video.read()

        # If there are frames left to read, put them in the frames array
        if frame_read:
            frame = cv2.resize(frame, (1920, 1080))
            frames[i] = frame
            frame_count += 1

        # If no more frames are left, pad the end with the same frame repeated
        else:
            if frame_count > 0:
                frames[i:] = frames[frame_count - 1]
            break

    return frames

# Frame difference function
def frame_difference(frames):

    # Array to hold numerical difference
    numDiff = np.zeros(len(frames)-1)

    for j in range (1,len(frames)):
        # Subtract frames
        diff = cv2.subtract(frames[j],frames[j-1])

        # add to array of numerical differences
        numDiff[j-1] = np.sum(diff)


    return(numDiff)

# chunk processing function:

def chunk_process(path,chunk_size=40):

    # create video capture:

    video = cv2.VideoCapture(path)

    fps = video.get(cv2.CAP_PROP_FPS)
    num_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)

    if not video.isOpened():
        print("Video could not be opened")
        return

    # Create variables to store differences, index, the flagged frame, and the chunk "index"

    max_diff = 0
    max_time = -1
    max_frame = None
    frame_offset = 0

    # Create variables to store the local maximums in each chunk in case they are useful too
    # Not allocating array because there shouldn't be many of these and also don't know exactly how many

    local_max_diffs = []
    local_max_times = []
    local_max_frames = []

    while video.isOpened() and frame_offset <= num_frames:

        # Get the chunk of frames:

        frames = capture_frames(video,chunk_size)

        # Get the difference:

        num_diffs = frame_difference(frames)

        # Find max_diff in the given chunk:

        local_max_diff = np.max(num_diffs)
        local_max_index = np.argmax(num_diffs) + frame_offset

        local_max_diffs.append(local_max_diff)
        local_max_times.append(local_max_index/fps )
        local_max_frames.append(frames[np.argmax(num_diffs) + 1])

        # set the total maximum values after each chunk

        if local_max_diff > max_diff:
            max_diff = local_max_diff
            max_time = local_max_index/fps
            max_frame = frames[np.argmax(num_diffs) + 1]
            diff_frame = max_frame - frames[np.argmax(num_diffs)]

        # iterate the frame offset in between chunks

        print(f' Frames Processed: {frame_offset}')
        frame_offset += len(frames)

    print("Finished Processing")
    video.release()
    cv2.destroyAllWindows()

    # if max_frame is not None:
    #     cv2.imshow('Frame with Max Difference', diff_frame)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    return([max_diff,max_time,max_frame,local_max_diffs,local_max_times,local_max_frames])

# Test

video_path = 'C:/Users/korol/Documents/Capstone TK/green flash and lag.mp4'

[diff,ttime,frame,ldiffs,ltimes,lframes] = chunk_process(video_path,chunk_size=10)

print(f'The most defective frame happens at {ttime} seconds')

# for i in range(np.shape(lframes)[0]):
#     cv2.imshow(f'Local Max at: {ltimes[i]}', lframes[i])
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

print(time.time()-start_time)
