import cv2
import numpy as np

class OpticalFlowAnalyzer:
    def __init__(self):
        self.prev_gray = None

    def compute_flow(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return None, frame
        
        # Compute dense optical flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        self.prev_gray = gray
        
        # Compute magnitude and angle of flow
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Visualize flow as a color map
        hsv = np.zeros_like(frame)
        hsv[..., 1] = 255
        hsv[..., 0] = angle * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        flow_visual = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Detect sudden movements (high magnitude indicates rapid motion)
        sudden_movement = np.mean(magnitude) > 5  # Threshold for sudden movement
        return sudden_movement, flow_visual

if __name__ == "__main__":
    cap = cv2.VideoCapture("../data/crowd_video.mp4")
    optical_flow = OpticalFlowAnalyzer()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        sudden_movement, flow_visual = optical_flow.compute_flow(frame)
        
        if sudden_movement:
            cv2.putText(flow_visual, "Sudden Movement Detected!", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow("Optical Flow", flow_visual)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()