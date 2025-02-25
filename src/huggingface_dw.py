from huggingface_hub import hf_hub_download

# Download the weights
hf_hub_download("merve/yolov9", filename="yolov9-c.pt", local_dir="./")
