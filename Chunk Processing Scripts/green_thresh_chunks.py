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

# green detect function
def green_thresh(frames):

    # see which frames are green using a logical mask
    green_mask = (frames[:, :, :, 1] > 100) & (frames[:, :, :, 0] < 10) & (frames[:, :, :, 2] < 10)

    # Find indices of these frames
    flagged_inds = np.where(np.any(green_mask, axis=(1, 2)))[0]

    return(flagged_inds)

# chunk processing function:

def chunk_process(path,chunk_size=40):

    # create video capture:

    video = cv2.VideoCapture(path)

    fps = video.get(cv2.CAP_PROP_FPS)
    num_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)

    if not video.isOpened():
        print("Video could not be opened")
        return

    # Create variables to store green indices, frames, and chunk index

    green_ind_list = []
    green_time_list = []
    green_frames = []
    frame_offset = 0

    while video.isOpened() and frame_offset <= num_frames:

        # Get the chunk of frames:

        frames = capture_frames(video,chunk_size)

        # find if it contains green and return indices

        green_inds = green_thresh(frames) + frame_offset
        green_times = green_inds/fps

        if green_inds.size > 0:
            green_ind_list.extend(green_inds)
            green_time_list.extend(green_times)
            # Green frames

            green_frames.extend(frames[green_inds-frame_offset])

        else:
            print(f' No green frames in chunk: {frame_offset}')
            frame_offset += len(frames)
            continue

        # iterate the frame offset in between chunks

        print(f' Green Frame at: {frame_offset}')
        frame_offset += len(frames)

    print("Finished Processing")
    video.release()
    cv2.destroyAllWindows()

    return([green_time_list,green_frames])

# Test

video_path = 'C:/Users/korol/Documents/Capstone TK/green flash and lag.mp4'

[times,gframes] = chunk_process(video_path,chunk_size=40)

# for i in range(np.shape(gframes)[0]):
#     cv2.imshow(f'Green frames in chunk: {times[i]}', gframes[i])
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

print(f'Times with green flash: {times}')
print(time.time()-start_time)
