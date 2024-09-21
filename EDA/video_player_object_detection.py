#Bounding Box
import cv2
import numpy as np
from ultralytics import YOLO

# Load the trained weights file
model = YOLO("/home/jinjuuk/dev_ws/pt_files/segmentation_n.pt")

# Open the camera
cap = cv2.VideoCapture(0)  # Usually 0

# Set camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Could not open video device.")
    exit()

# Initialize thresholds
confidence_threshold = 40  # Initial confidence threshold (in percent)
iou_threshold = 50  # Initial IoU threshold (in percent)

# Callback function for trackbars
def nothing(x):
    pass

# Create a window for display
cv2.namedWindow('YOLOv8 Real-Time Object Detection')

# Create trackbars for adjusting confidence and IoU thresholds
cv2.createTrackbar('Confidence Threshold', 'YOLOv8 Real-Time Object Detection', confidence_threshold, 100, nothing)
cv2.createTrackbar('IoU Threshold', 'YOLOv8 Real-Time Object Detection', iou_threshold, 100, nothing)

# Real-time object detection
while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    # Convert OpenCV frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get the current positions of the trackbars
    confidence_threshold = cv2.getTrackbarPos('Confidence Threshold', 'YOLOv8 Real-Time Object Detection') / 100
    iou_threshold = cv2.getTrackbarPos('IoU Threshold', 'YOLOv8 Real-Time Object Detection') / 100

    # Perform object detection with custom IoU threshold
    results = model.predict(source=frame_rgb, conf=confidence_threshold, iou=iou_threshold)
    
    # Draw bounding boxes and labels
    for result in results[0].boxes:
        conf = result.conf[0].item()  # Convert tensor to float
        if conf >= confidence_threshold:  # Filter by confidence score
            x1, y1, x2, y2 = map(int, result.xyxy[0])  # Bounding box coordinates
            cls = int(result.cls[0])  # Class ID
            label = f'{model.names[cls]} {conf:.2f}'

            # Draw bounding box and label on frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display the frame with detections
    cv2.imshow('YOLOv8 Real-Time Object Detection', frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()










