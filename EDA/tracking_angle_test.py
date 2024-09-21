# import cv2
# import numpy as np
# from ultralytics import YOLO
# import time

# # Load the YOLOv8 segmentation model
# model = YOLO("/home/jinjuuk/dev_ws/pt_files/newjeans.pt")

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

# def find_object_angle(mask, center):
#     """
#     Find the angle of the object using its mask and center point.
#     """
#     y, x = np.where(mask > 0)
#     points = np.column_stack((x, y))
    
#     # Calculate distances from center to all points
#     distances = np.sqrt(np.sum((points - center) ** 2, axis=1))
    
#     # Find the farthest point
#     farthest_index = np.argmax(distances)
#     farthest_point = points[farthest_index]
    
#     # Calculate angle
#     dx = farthest_point[0] - center[0]
#     dy = farthest_point[1] - center[1]
#     angle = np.degrees(np.arctan2(dy, dx))
    
#     return angle, farthest_point

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

#         # Calculate and print center points of bounding boxes, class names, and angles
#         for box, mask in zip(results[0].boxes, results[0].masks):
#             x1, y1, x2, y2 = box.xyxy[0]
#             center_x = int((x1 + x2) / 2)
#             center_y = int((y1 + y2) / 2)
#             center = (center_x, center_y)
            
#             class_id = int(box.cls[0])
#             class_name = model.names[class_id]
            
#             # Find object angle
#             angle, farthest_point = find_object_angle(mask.data[0].cpu().numpy(), center)
            
#             print(f"Class: {class_name}, Center point: ({center_x:.2f}, {center_y:.2f}), Class_Id: {class_id}, Angle: {angle:.2f}")
            
#             # Draw center line
#             cv2.line(annotated_frame, center, (center_x, 0), (0, 255, 0), 2)
            
#             # Draw line to farthest point
#             cv2.line(annotated_frame, center, tuple(farthest_point), (0, 0, 255), 2)
            
#             # Put angle text
#             cv2.putText(annotated_frame, f"Angle: {angle:.2f}", (center_x, center_y - 20),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

#         # Display the frame with detections
#         cv2.imshow('YOLOv8 Real-Time Object Detection', annotated_frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# cap.release()
# cv2.destroyAllWindows()













# import cv2
# import numpy as np
# from ultralytics import YOLO
# import time

# # Load the YOLOv8 segmentation model
# model = YOLO("/home/jinjuuk/dev_ws/pt_files/newjeans.pt")

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

# def find_object_angle(mask, center):
#     """
#     Find the angle of the object using its mask and center point.
#     """
#     y, x = np.where(mask > 0)
#     points = np.column_stack((x, y))
    
#     # Calculate distances from center to all points
#     distances = np.sqrt(np.sum((points - center) ** 2, axis=1))
    
#     # Find the farthest point
#     farthest_index = np.argmax(distances)
#     farthest_point = points[farthest_index]
    
#     # Calculate angle
#     dx = farthest_point[0] - center[0]
#     dy = farthest_point[1] - center[1]
#     angle = np.degrees(np.arctan2(dy, dx))
    
#     return angle, farthest_point


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

#         # Check if masks are available (i.e., objects were detected)
#         if results[0].masks is not None:
#             # Calculate and print center points of bounding boxes, class names, and angles
#             for box, mask in zip(results[0].boxes, results[0].masks):
#                 x1, y1, x2, y2 = box.xyxy[0]
#                 center_x = int((x1 + x2) / 2)
#                 center_y = int((y1 + y2) / 2)
#                 center = (center_x, center_y)
                
#                 class_id = int(box.cls[0])
#                 class_name = model.names[class_id]
                
#                 # Find object angle
#                 angle, farthest_point = find_object_angle(mask.data[0].cpu().numpy(), center)
                
#                 print(f"Class: {class_name}, Center point: ({center_x:.2f}, {center_y:.2f}), Class_Id: {class_id}, Angle: {angle:.2f}")
                
#                 # Draw center line
#                 cv2.line(annotated_frame, center, (center_x, 0), (0, 255, 0), 2)
                
#                 # Draw line to farthest point
#                 cv2.line(annotated_frame, center, tuple(farthest_point), (0, 0, 255), 2)
                
#                 # Put angle text
#                 cv2.putText(annotated_frame, f"Angle: {angle:.2f}", (center_x, center_y - 20),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
#         else:
#             print("No objects detected in this frame")

#         # Display the frame with detections
#         cv2.imshow('YOLOv8 Real-Time Object Detection', annotated_frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# cap.release()
# cv2.destroyAllWindows()


import cv2
import numpy as np
from ultralytics import YOLO
import time

# Load the YOLOv8 segmentation model
model = YOLO("/home/jinjuuk/dev_ws/pt_files/newjeans.pt")

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
cv2.namedWindow('YOLOv8 Real-Time Object Detection')

# Create trackbars for adjusting confidence and IoU thresholds
cv2.createTrackbar('Confidence Threshold', 'YOLOv8 Real-Time Object Detection', confidence_threshold, 100, nothing)
cv2.createTrackbar('IoU Threshold', 'YOLOv8 Real-Time Object Detection', iou_threshold, 100, nothing)

def find_object_angle(mask, center):
    """
    Find the angle of the object using its mask and center point.
    """
    y, x = np.where(mask > 0)
    points = np.column_stack((x, y))
    
    # Calculate distances from center to all points
    distances = np.sqrt(np.sum((points - center) ** 2, axis=1))
    
    # Find the farthest point
    farthest_index = np.argmax(distances)
    farthest_point = points[farthest_index]
    
    # Calculate angle
    dx = farthest_point[0] - center[0]
    dy = farthest_point[1] - center[1]
    angle = np.degrees(np.arctan2(dy, dx))
    
    return angle, farthest_point


while cap.isOpened():
    success, frame = cap.read()
    if success:
        # Get the current positions of the trackbars
        confidence_threshold = cv2.getTrackbarPos('Confidence Threshold', 'YOLOv8 Real-Time Object Detection') / 100
        iou_threshold = cv2.getTrackbarPos('IoU Threshold', 'YOLOv8 Real-Time Object Detection') / 100

        # Start timer
        start = time.perf_counter()

        # Perform prediction
        results = model(frame, conf=confidence_threshold, iou=iou_threshold)

        # End timer
        end = time.perf_counter()
        total_time = end - start
        fps = 1 / total_time

        # Annotate frame with segmentation results
        annotated_frame = results[0].plot()

        # Display FPS on the frame
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Check if masks are available (i.e., objects were detected)
        if results[0].masks is not None:
            # Calculate and print center points of bounding boxes, class names, and angles
            for box, mask in zip(results[0].boxes, results[0].masks):
                x1, y1, x2, y2 = box.xyxy[0]
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                center = (center_x, center_y)
                
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                
                # Find object angle
                angle, farthest_point = find_object_angle(mask.data[0].cpu().numpy(), center)
                
                print(f"Class: {class_name}, Center point: ({center_x:.2f}, {center_y:.2f}), Class_Id: {class_id}, Angle: {angle:.2f}")
                
                # Draw center line within the bounding box
                cv2.line(annotated_frame, (int(x1), center_y), (int(x2), center_y), (0, 255, 0), 2)
                
                # Draw line to farthest point
                cv2.line(annotated_frame, center, tuple(farthest_point), (0, 0, 255), 2)
                
                # Put angle text
                cv2.putText(annotated_frame, f"Angle: {angle:.2f}", (center_x, center_y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        else:
            print("No objects detected in this frame")

        # Display the frame with detections
        cv2.imshow('YOLOv8 Real-Time Object Detection', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
