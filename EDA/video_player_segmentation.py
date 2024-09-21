# # Segmentation
# import cv2
# import numpy as np
# from ultralytics import YOLO
# import time

# # Load the YOLOv8 segmentation model
# model = YOLO("/home/jinjuuk/dev_ws/pt_files/yolov5s.pt")

# # Open the camera
# cap = cv2.VideoCapture(0)  # 0 for the default camera

# # Set camera resolution
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# if not cap.isOpened():
#     print("Error: Could not open video device.")
#     exit()

# # Initialize thresholds
# confidence_threshold = 40  # Initial confidence threshold (in percent)
# iou_threshold = 50  # Initial IoU threshold (in percent)

# # Callback function for trackbars
# def nothing(x):
#     pass

# # Create a window for display
# cv2.namedWindow('YOLOv8 Real-Time Object Detection')

# # Create trackbars for adjusting confidence and IoU thresholds
# cv2.createTrackbar('Confidence Threshold', 'YOLOv8 Real-Time Object Detection', confidence_threshold, 100, nothing)
# cv2.createTrackbar('IoU Threshold', 'YOLOv8 Real-Time Object Detection', iou_threshold, 100, nothing)

# while cap.isOpened():
#     success, frame = cap.read()

#     if success:
#         # Get the current positions of the trackbars
#         confidence_threshold = cv2.getTrackbarPos('Confidence Threshold', 'YOLOv8 Real-Time Object Detection') / 100
#         iou_threshold = cv2.getTrackbarPos('IoU Threshold', 'YOLOv8 Real-Time Object Detection') / 100

#         # Start timer
#         start = time.perf_counter()
        
#         # Perform prediction
#         results = model(frame, conf=confidence_threshold, iou=iou_threshold)
        
#         # End timer
#         end = time.perf_counter()
#         total_time = end - start
#         fps = 1 / total_time

#         # Annotate frame with segmentation results
#         annotated_frame = results[0].plot()

#         # Display FPS on the frame
#         cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#         # Calculate and print center points of bounding boxes and class names
#         for box in results[0].boxes:
#             x1, y1, x2, y2 = box.xyxy[0]
#             center_x = (x1 + x2) / 2
#             center_y = (y1 + y2) / 2
#             class_id = int(box.cls[0])
#             class_name = model.names[class_id]
#             print(f"Class: {class_name}, Center point: ({center_x:.2f}, {center_y:.2f}, Class_Id: {class_id}")



#         # Display the frame with detections
#         cv2.imshow('YOLOv8 Real-Time Object Detection', annotated_frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


# Segmentation
import cv2
import numpy as np
import torch
import time

# Load the YOLOv5 segmentation model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/jinjuuk/dev_ws/pt_files/yolov5s.pt')

# Open the camera
cap = cv2.VideoCapture(0)  # 0 for the default camera

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
cv2.namedWindow('YOLOv5 Real-Time Object Detection')

# Create trackbars for adjusting confidence and IoU thresholds
cv2.createTrackbar('Confidence Threshold', 'YOLOv5 Real-Time Object Detection', confidence_threshold, 100, nothing)
cv2.createTrackbar('IoU Threshold', 'YOLOv5 Real-Time Object Detection', iou_threshold, 100, nothing)

while cap.isOpened():
    success, frame = cap.read()

    if success:
        # Get the current positions of the trackbars
        confidence_threshold = cv2.getTrackbarPos('Confidence Threshold', 'YOLOv5 Real-Time Object Detection') / 100.0
        iou_threshold = cv2.getTrackbarPos('IoU Threshold', 'YOLOv5 Real-Time Object Detection') / 100.0

        # Start timer
        start = time.perf_counter()
        
        # Perform prediction
        results = model(frame)  # conf와 iou는 내부에서 처리됩니다
        
        # End timer
        end = time.perf_counter()
        total_time = end - start
        fps = 1 / total_time

        # Filter results for 'person' class only
        person_class_id = None
        for key, value in model.names.items():
            if value == 'person':
                person_class_id = key
                break

        # Check if person class ID was found
        if person_class_id is not None:
            filtered_results = [box for box in results.xyxy[0] if int(box[5]) == person_class_id]

            # Annotate frame with segmentation results
            for box in filtered_results:
                x1, y1, x2, y2, conf, class_id = box
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                class_name = model.names[int(class_id)]

                # Draw bounding box and label
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f'{class_name} {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Print center point and class name
                print(f"Class: {class_name}, Center point: ({center_x:.2f}, {center_y:.2f}), Class_Id: {int(class_id)}")

            # Display FPS on the frame
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame with detections
        cv2.imshow('YOLOv5 Real-Time Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
