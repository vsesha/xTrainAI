#import torch
#model_path = "/Users/vasudevanseshadri/AllWorks/AI_Test/xTrainAI/src/yolov9/yolov9-c.pt"
#model = torch.load(model_path, map_location=torch.device('cpu'))
#model.eval()

#import sys
#sys.path.append('/Users/vasudevanseshadri/AllWorks/AI_Test/xTrainAI/src/yolov9')

#import torch
#model_path = '/Users/vasudevanseshadri/AllWorks/AI_Test/xTrainAI/src/yolov9/yolov9-c.pt'
#model = torch.load(model_path, map_location=torch.device('cpu'))
#        results1 = model(frame1)
#        results2 = model(frame2)



#import sys
#import torch
#from models.yolo import Model  # Ensure this import is correct for your YOLO version

# Add YOLOv9 directory to path
#sys.path.append('/Users/vasudevanseshadri/AllWorks/AI_Test/xTrainAI/src/yolov9')

# Define the model architecture
#model = Model(cfg='path/to/yolov9.yaml')  # Use the correct config file for your model

# Load weights into the model
#model.load_state_dict(torch.load('/Users/vasudevanseshadri/AllWorks/AI_Test/xTrainAI/src/yolov9/yolov9-c.pt', map_location=torch.device('cpu')))

# Set the model to evaluation mode
#model.eval()

import sys
import torch

checkpoint = torch.load('/Users/vasudevanseshadri/AllWorks/AI_Test/xTrainAI/src/yolov9/yolov9-c.pt', map_location='cpu')
print(type(checkpoint))