import torch
import cv2
import numpy as np
import sys
import os
from simple_tracker import SimpleTracker

# Add FairMOT to the system path
sys.path.append(r"C:\Users\Nikhi\OneDrive\Desktop\stampede_env\FairMOT")  # Replace with the actual path to your FairMOT directory
from fairmot.tracker import Tracker  # FairMOT's tracker class
from fairmot.opts import Opts  # FairMOT's options class

class PersonTracker:
    def __init__(self, model_path=r"C:\Users\Nikhi\OneDrive\Desktop\stampede_env\models\fairmot.pth"):
        # Initialize FairMOT options
        self.opt = Opts().init()  # Load default options
        self.opt.load_model = model_path  # Specify the model path
        self.opt.device = torch.device("cpu")  # Force CPU usage
        self.opt.gpus = []  # Disable GPU usage
        
        # Initialize the FairMOT tracker
        self.tracker = Tracker(self.opt)
        self.tracks = {}

    def track(self, frame, detections):
        # Convert YOLO detections to FairMOT input format: [x1, y1, x2, y2, conf]
        dets = np.array([[x1, y1, x2, y2, conf] for x1, y1, x2, y2, conf in detections], dtype=np.float32)

        # Pre-process the frame for FairMOT
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (1088, 608))  # FairMOT default input size
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]
        img = torch.from_numpy(img).unsqueeze(0).to(self.opt.device)  # Add batch dimension

        # Run FairMOT tracking
        self.tracker.update(img, dets)  # Update tracker with new detections
        tracked_objects = self.tracker.tracks  # Get tracked objects

        # Update tracks with new IDs and bounding boxes
        self.tracks.clear()
        for track in tracked_objects:
            track_id = track.track_id
            bbox = track.tlbr  # [top-left-x, top-left-y, bottom-right-x, bottom-right-y]
            self.tracks[track_id] = bbox

        return self.tracks

if __name__ == "__main__":
    from detect_people import PersonDetector
    
    detector = PersonDetector()
    tracker = PersonTracker()
    cap = cv2.VideoCapture("../data/crowd_video.mp4")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect people
        person_boxes = detector.detect(frame)
        
        # Track people
        tracks = tracker.track(frame, person_boxes)
        
        # Draw tracks
        for track_id, bbox in tracks.items():
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        cv2.imshow("FairMOT Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()