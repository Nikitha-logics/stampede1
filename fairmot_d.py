import os
import requests
import shutil

def download_fairmot_model():
    model_path = "../models/fairmot.pth"
    models_dir = os.path.dirname(model_path)
    
    # Create models directory if it doesn't exist
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    # Known Google Drive link for fairmot_dla34.pth (rename to fairmot.pth)
    url = "https://drive.google.com/uc?export=download&id=1m0cO3s3Xv3j0fA_9pArS6_KZ7nU0nq1d"  # Example ID; check FairMOT README for the latest
    try:
        print("Attempting to download fairmot.pth from Google Drive...")
        response = requests.get(url, stream=True, allow_redirects=True)
        if response.status_code == 200:
            with open(model_path, "wb") as f:
                response.raw.decode_content = True
                shutil.copyfileobj(response.raw, f)
            print(f"fairmot.pth downloaded and saved to {model_path}")
        else:
            print(f"Download failed with status code {response.status_code}. Please check the URL.")
    except Exception as e:
        print(f"Download failed: {e}. Please follow manual instructions below.")
    
    # Manual instructions if download fails
    if not os.path.exists(model_path):
        print("Manual Download Required:")
        print("1. Clone the FairMOT repository: git clone https://github.com/ifzhang/FairMOT.git")
        print("2. Follow the FairMOT README to install dependencies (e.g., pip install -r requirements.txt, compile DCNv2).")
        print("3. Download fairmot_dla34.pth from the FairMOT GitHub README (e.g., Google Drive link: https://drive.google.com/drive/folders/1j1dWjK0Wj4b7vH4zLtWn8tKX7a_8X8g).")
        print("4. Rename it to fairmot.pth and place it in Stampede-Detection/models/.")

if __name__ == "__main__":
    download_fairmot_model()