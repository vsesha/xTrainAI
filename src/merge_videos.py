import cv2
from vidgear.gears import VideoGear, WriteGear
from datetime import datetime

# Paths to your video files
video1_path = '/Users/vasudevanseshadri/AllWorks/AI_Test/xTrainAI/inputVideos/Video_1.mov'
video2_path = '/Users/vasudevanseshadri/AllWorks/AI_Test/xTrainAI/inputVideos/Video_2.mov'

# Open video files with VideoGear
video1 = VideoGear(source=video1_path).start()
video2 = VideoGear(source=video2_path).start()


# Use OpenCV's VideoCapture to get frame rates
video1_capture = cv2.VideoCapture(video1_path)
video2_capture = cv2.VideoCapture(video2_path)

frame_rate1 = int(video1_capture.get(cv2.CAP_PROP_FPS))
frame_rate2 = int(video2_capture.get(cv2.CAP_PROP_FPS))

video1_capture.release()
video2_capture.release()

# Calculate frames to skip
skip_frames1 = 2 * frame_rate1
skip_frames2 = 3 * frame_rate2

# Generate timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Define video writer with output parameters and timestamp in filename
output_path = f'output_video_{timestamp}.mp4'
output_params = {"-fourcc": "mp4v", "-fps": 30, "-input_framerate": 30}
writer = WriteGear(output=output_path, compression_mode=True, logging=True, **output_params)

frame_counter1 = 0
frame_counter2 = 0

# Process video frames
while True:
    frame1 = video1.read()
    frame2 = video2.read()
    
    # Skip initial frames for video1 and video2
    if frame_counter1 < skip_frames1:
        frame_counter1 += 1
        continue
    if frame_counter2 < skip_frames2:
        frame_counter2 += 1
        continue
    
    # Check if frames are valid
    if frame1 is None or frame2 is None:
        break
    
    # Resize second frame to match the dimensions of the first frame
    frame2_resized = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))
    
    # Superimpose the frames by blending them
    superimposed_frame = cv2.addWeighted(frame1, 0.5, frame2_resized, 0.5, 0)
    
    # Write each frame multiple times to create slow-motion effect
    for _ in range(3):  # Change the number to adjust the slow-motion effect
        writer.write(superimposed_frame)

    frame_counter1 += 1
    frame_counter2 += 1

# Release resources
video1.stop()
video2.stop()
writer.close()