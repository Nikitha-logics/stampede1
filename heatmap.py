import cv2
import numpy as np
import matplotlib.pyplot as plt

class DensityHeatmap:
    def __init__(self, frame_shape):
        self.height, self.width = frame_shape[:2]
        self.heatmap = np.zeros((self.height, self.width), dtype=np.float32)

    def update(self, person_boxes):
        # Reset heatmap
        self.heatmap.fill(0)
        
        # Add Gaussian blobs for each person
        for box in person_boxes:
            x1, y1, x2, y2, _ = box
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            sigma = 50  # Spread of the Gaussian
            for i in range(self.height):
                for j in range(self.width):
                    self.heatmap[i, j] += np.exp(-((i - center_y) ** 2 + (j - center_x) ** 2) / (2 * sigma ** 2))
        
        # Normalize heatmap
        self.heatmap = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX)
        self.heatmap = self.heatmap.astype(np.uint8)

    def apply_to_frame(self, frame):
        # Convert heatmap to color map
        heatmap_color = cv2.applyColorMap(self.heatmap, cv2.COLORMAP_JET)
        # Overlay heatmap on the frame
        overlay = cv2.addWeighted(frame, 0.5, heatmap_color, 0.5, 0)
        return overlay

if __name__ == "__main__":
    from detect_people import PersonDetector
    
    detector = PersonDetector()
    cap = cv2.VideoCapture("../data/crowd_video.mp4")
    ret, frame = cap.read()
    heatmap = DensityHeatmap(frame.shape)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        person_boxes = detector.detect(frame)
        heatmap.update(person_boxes)
        frame_with_heatmap = heatmap.apply_to_frame(frame)
        
        cv2.imshow("Density Heatmap", frame_with_heatmap)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()