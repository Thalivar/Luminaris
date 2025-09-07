import cv2
import numpy as np
from ultralytics import YOLO
from config.settings import Config

class YOLODetector:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")
        self.securityClasses = {
            0: "person",
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck"
            # Add more class ID's here later on
        }

    def detectObjects(self, frame):
        results = self.model(frame, verbose = False)
        detections = []

        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    confidence = float(box.conf[0])
                    classID = int(box.cls[0])

                    if confidence >= Config.minDetectionConfidence:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        className = self.model.names[classID]
                        shouldInclude = False

                        if Config.detectAllClasses:
                            shouldInclude = True
                        else:
                            shouldInclude = classID in self.securityClasses
                        
                        if shouldInclude:
                            detection = {
                                "className": className,
                                "confidence": confidence,
                                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                                "classID": classID
                            }
                            detections.append(detection)

                        if len(detection) >= Config.maxDetectionsPerFrame:
                            break

        return detections

    def drawDetections(self, frame, detections):
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            className = detection["className"]
            confidence = detection["confidence"]

            if className == "person":
                color = (0, 255, 0) # Green
            elif className in ["laptop", "computer", "keyboard", "mouse"]:
                color = (255, 165, 0) # Orange
            elif className in ["cup", "bottle", "bowl"]:
                color = (0, 255, 255) # Yellow
            elif className in ["chair", "couch", "bed", "desk"]:
                color = (255, 0, 255) # Magenta
            elif className in ["book", "cell phone", "remote"]:
                color = (128, 0, 128) # Purple
            else:
                color = (0, 0, 255) # Red
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{className}: {confidence:.2f}"
            labelSize = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

            cv2.rectangle(frame, (x1, y1 - labelSize[1] - 10), (x1 + labelSize[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return frame
