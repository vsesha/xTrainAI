import ssl
import certifi
import cv2
from vidgear.gears import VideoGear, WriteGear
from datetime import datetime
import torch
from huggingface_hub import hf_hub_download


import sys
sys.path.append('/Users/vasudevanseshadri/AllWorks/AI_Test/xTrainAI/src/yolov9')


# Set the default SSL context to use the certifi certificates
#ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())


# Download YOLOv9 weights
#hf_hub_download("merve/yolov9", filename="yolov9-c.pt", local_dir="./")

# Load YOLOv9 model from local directory
#repo_or_dir = '/Users/vasudevanseshadri/AllWorks/AI_Test/xTrainAI/src/yolov9'
#repo_or_dir = '../yolov9'
#model_path = 'yolov9-c.pt'

#model = torch.hub.load(repo_or_dir, 'custom', path=model_path)


model_path = '/Users/vasudevanseshadri/AllWorks/AI_Test/xTrainAI/src/yolov9/yolov9-c.pt'
model = torch.load(model_path, map_location=torch.device('cpu'))



#model = torch.load(model_path)

# Load YOLOv9 model from the YOLOv9 directory
#model = YOLO('yolov9-c.pt')


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

# Calculate the maximum number of frames to process
max_frames = 25 * min(frame_rate1, frame_rate2)

# Generate timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Define video writer with output parameters and timestamp in filename
output_path = f'output_video_{timestamp}.mp4'
output_params = {"-fourcc": "mp4v", "-fps": 30, "-input_framerate": 30}
writer = WriteGear(output=output_path, compression_mode=True, logging=True, **output_params)

frame_counter1 = 0
frame_counter2 = 0
processed_frames = 0

# Function to draw trails with specified color
def draw_trails(frame, points, color):
    for i in range(1, len(points)):
        if points[i - 1] is None or points[i] is None:
            continue
        cv2.line(frame, points[i - 1], points[i], color, 2)

# Lists to store points for drawing trails
points1 = []
points2 = []

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
    
    # Check if frames are valid or if processed_frames exceed max_frames
    if frame1 is None or frame2 is None or processed_frames >= max_frames:
        break
    
    # Slow down the videos by repeating frames
    for _ in range(3):  # Adjust this value for the desired slow-motion effect
        # Detect objects using YOLOv9
        results1 = model(frame1)
        results2 = model(frame2)
        
        # Process results for video1
        for result in results1.xyxy[0]:
            label = int(result[5])
            if label == "person":
                cv2.putText(frame1, "coach_player", (int(result[0]), int(result[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                cv2.rectangle(frame1, (int(result[0]), int(result[1])), (int(result[2])), (int(result[3])), (0,255,0), 2)
            elif label == "ball":
                cv2.putText(frame1, "coaching_ball", (int(result[0]), int(result[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
                cv2.rectangle(frame1, (int(result[0]), int(result[1])), (int(result[2])), (int(result[3])), (0,0,255), 2)
                points1.append((int(result[0] + (result[2] - result[0]) / 2), int(result[1] + (result[3] - result[1]) / 2)))
        
        # Process results for video2
        for result in results2.xyxy[0]:
            label = int(result[5])
            if label == "person":
                cv2.putText(frame2, "trainee_1", (int(result[0]), int(result[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                cv2.rectangle(frame2, (int(result[0]), int(result[1])), (int(result[2])), (int(result[3])), (0,255,0), 2)
            elif label == "ball":
                cv2.putText(frame2, "trainee_ball", (int(result[0]), int(result[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
                cv2.rectangle(frame2, (int(result[0]), int(result[1])), (int(result[2])), (int(result[3])), (0,0,255), 2)
                points2.append((int(result[0] + (result[2] - result[0]) / 2), int(result[1] + (result[3] - result[1]) / 2)))
        
        # Draw trails
        draw_trails(frame1, points1, (0, 0, 255))  # Red color for video1
        draw_trails(frame2, points2, (0, 100, 0))  # Dark green color for video2
        
        # Resize second frame to match the dimensions of the first frame
        frame2_resized = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))
        
        # Superimpose the frames by blending them
        superimposed_frame = cv2.addWeighted(frame1, 0.5, frame2_resized, 0.5, 0)
        
        # Write the superimposed frame to output video
        writer.write(superimposed_frame)
        
        processed_frames += 1

    frame_counter1 += 1
    frame_counter2 += 1

# Release resources
video1.stop()
video2.stop()
writer.close()
