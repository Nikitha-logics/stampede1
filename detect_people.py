import cv2
from ultralytics import YOLO

class PersonDetector:
    def __init__(self, model_path="../models/yolov8m.pt"):
        self.model = YOLO(model_path)
        self.class_id_person = 0  # YOLOv8 class ID for "person"

    def detect(self, frame):
        # Run YOLOv8 inference on the frame
        results = self.model(frame)
        
        # Extract person detections
        person_boxes = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                if int(box.cls) == self.class_id_person:  # Filter for "person" class
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf)
                    person_boxes.append([x1, y1, x2, y2, conf])
        return person_boxes

if __name__ == "__main__":
    # Test the detector
    detector = PersonDetector()
    cap = cv2.VideoCapture("../data/crowd_video.mp4")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        person_boxes = detector.detect(frame)
        for box in person_boxes:
            x1, y1, x2, y2, conf = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Conf: {conf:.2f}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow("YOLOv8 Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()