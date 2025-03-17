import cv2
import numpy as np
from detect_people import PersonDetector
from track_people import PersonTracker
from heatmap import DensityHeatmap
from optical_flow import OpticalFlowAnalyzer
from risk_prediction import RiskPredictor

def main():
    # Initialize components
    cap = cv2.VideoCapture("../data/crowd_video.mp4")
    detector = PersonDetector()
    tracker = PersonTracker()
    heatmap = DensityHeatmap(cap.read()[1].shape)
    optical_flow = OpticalFlowAnalyzer()
    risk_predictor = RiskPredictor()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Step 1: Detect people with YOLOv8
        person_boxes = detector.detect(frame)

        # Step 2: Track people with FairMOT
        tracks = tracker.track(frame, person_boxes)

        # Step 3: Generate density heatmap
        heatmap.update(person_boxes)
        frame_with_heatmap = heatmap.apply_to_frame(frame)

        # Step 4: Detect sudden movements with Optical Flow
        sudden_movement, flow_visual = optical_flow.compute_flow(frame)

        # Step 5: Calculate density (normalized number of people)
        density = len(person_boxes) / (frame.shape[0] * frame.shape[1]) * 1000  # Example normalization

        # Step 6: Predict stampede risk with LSTM
        risk_score = risk_predictor.predict(density, sudden_movement)

        # Draw bounding boxes and track IDs
        for track_id, bbox in tracks.items():
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame_with_heatmap, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame_with_heatmap, f"ID: {track_id}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Overlay risk score and sudden movement warning
        if sudden_movement:
            cv2.putText(frame_with_heatmap, "Sudden Movement Detected!", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame_with_heatmap, f"Risk Score: {risk_score:.2f}", (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Display the result
        cv2.imshow("Stampede Detection", frame_with_heatmap)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()