import os
import requests
import shutil
from ultralytics import YOLO

def download_yolov8_model():
    model_path = "../models/yolov8m.pt"
    models_dir = os.path.dirname(model_path)
    
    # Create models directory if it doesn't exist
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    # Attempt to load the model (this triggers automatic download if not cached)
    try:
        print("Attempting to load YOLOv8m.pt via Ultralytics...")
        model = YOLO("yolov8m.pt")  # This downloads to cache if not present
        # Move the cached model to our models directory
        cache_path = os.path.expanduser("~/.cache/ultralytics/models/yolov8m.pt")
        if os.path.exists(cache_path):
            shutil.copy(cache_path, model_path)
            print(f"YOLOv8m.pt downloaded and saved to {model_path}")
        else:
            print("Model not found in cache. Trying manual download...")
    except Exception as e:
        print(f"Automatic download failed: {e}. Falling back to manual download...")
    
    # Manual download URL (as of March 16, 2025, based on Ultralytics releases)
    url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt"
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(model_path, "wb") as f:
                response.raw.decode_content = True
                shutil.copyfileobj(response.raw, f)
            print(f"YOLOv8m.pt manually downloaded and saved to {model_path}")
        else:
            print(f"Failed to download from {url}. Status code: {response.status_code}")
    except Exception as e:
        print(f"Manual download failed: {e}. Please download yolov8m.pt manually from https://github.com/ultralytics/ultralytics/releases and place it in Stampede-Detection/models/")

if __name__ == "__main__":
    download_yolov8_model()