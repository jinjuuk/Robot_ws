# import cv2
# import numpy as np
# from ultralytics import YOLO
# import time
# import math

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

# def calculate_nearest_boundary_point(mask, center_x, center_y):
#     min_dist = float('inf')
#     nearest_point = (center_x, center_y)
    
#     # Iterate through all pixels in the mask
#     for y in range(mask.shape[0]):
#         for x in range(mask.shape[1]):
#             if mask[y, x] == 1:  # Check if the pixel is part of the boundary
#                 dist = math.sqrt((center_x - x) ** 2 + (center_y - y) ** 2)
#                 if dist < min_dist:
#                     min_dist = dist
#                     nearest_point = (x, y)
    
#     return nearest_point

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

#         # Process the segmentation masks and boxes
#         masks = results[0].masks  # Assumes that the results contain masks
#         boxes = results[0].boxes  # Get the bounding boxes

#         for i, mask in enumerate(masks):
#             class_id = int(boxes.cls[i])  # Get the class ID for the corresponding mask
#             class_name = model.names[class_id]
            
#             if class_name == "side_cup":  # Check if the detected object is a side_cup
#                 mask = mask.cpu().numpy().astype(np.uint8)  # Convert mask to numpy array if necessary and binarize it
                
#                 # Calculate the center of the mask
#                 M = cv2.moments(mask)
#                 if M["m00"] != 0:
#                     cX = int(M["m10"] / M["m00"])
#                     cY = int(M["m01"] / M["m00"])
#                 else:
#                     cX, cY = 0, 0
                
#                 # Find the nearest boundary pixel
#                 nearest_point = calculate_nearest_boundary_point(mask, cX, cY)
                
#                 # Draw a line from the center to the nearest boundary pixel
#                 cv2.line(annotated_frame, (cX, cY), nearest_point, (0, 255, 0), 2)
                
#                 # Optionally display the nearest point and the center
#                 cv2.circle(annotated_frame, (cX, cY), 5, (255, 0, 0), -1)
#                 cv2.circle(annotated_frame, nearest_point, 5, (0, 0, 255), -1)

#         # Display the frame with detections
#         cv2.imshow('YOLOv8 Real-Time Object Detection', annotated_frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()




# import cv2
# import numpy as np
# from ultralytics import YOLO
# import time
# import math

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

# def calculate_nearest_boundary_point(mask, center_x, center_y):
#     min_dist = float('inf')
#     nearest_point = (center_x, center_y)
    
#     # Iterate through all pixels in the mask
#     for y in range(mask.shape[0]):
#         for x in range(mask.shape[1]):
#             if mask[y, x] == 1:  # Check if the pixel is part of the boundary
#                 dist = math.sqrt((center_x - x) ** 2 + (center_y - y) ** 2)
#                 if dist < min_dist:
#                     min_dist = dist
#                     nearest_point = (x, y)
    
#     return nearest_point

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

#         # Process the segmentation masks and boxes
#         masks = results[0].masks  # Assumes that the results contain masks
#         boxes = results[0].boxes  # Get the bounding boxes

#         if masks is not None and boxes is not None:  # Check if masks and boxes are not None
#             for i, mask in enumerate(masks):
#                 class_id = int(boxes.cls[i])  # Get the class ID for the corresponding mask
#                 class_name = model.names[class_id]

#                 if class_name == "side_cup":  # Check if the detected object is a side_cup
#                     mask = mask.cpu().numpy().astype(np.uint8)  # Convert mask to numpy array if necessary and binarize it

#                     # Calculate the center of the mask
#                     M = cv2.moments(mask)
#                     if M["m00"] != 0:
#                         cX = int(M["m10"] / M["m00"])
#                         cY = int(M["m01"] / M["m00"])
#                     else:
#                         cX, cY = 0, 0

#                     # Find the nearest boundary pixel
#                     nearest_point = calculate_nearest_boundary_point(mask, cX, cY)

#                     # Draw a line from the center to the nearest boundary pixel
#                     cv2.line(annotated_frame, (cX, cY), nearest_point, (0, 255, 0), 2)

#                     # Optionally display the nearest point and the center
#                     cv2.circle(annotated_frame, (cX, cY), 5, (255, 0, 0), -1)
#                     cv2.circle(annotated_frame, nearest_point, 5, (0, 0, 255), -1)

#         # Display the frame with detections
#         cv2.imshow('YOLOv8 Real-Time Object Detection', annotated_frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


# import cv2
# import numpy as np
# from ultralytics import YOLO
# import time
# import math

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

# def calculate_nearest_boundary_point(mask, center_x, center_y):
#     min_dist = float('inf')
#     nearest_point = (center_x, center_y)
    
#     # Iterate through all pixels in the mask
#     for y in range(mask.shape[0]):
#         for x in range(mask.shape[1]):
#             # Check if the pixel is part of the boundary
#             if mask[y, x] == 1:
#                 # Check 8-connected neighborhood to see if it's a boundary pixel
#                 if (y > 0 and mask[y - 1, x] == 0) or (y < mask.shape[0] - 1 and mask[y + 1, x] == 0) or \
#                    (x > 0 and mask[y, x - 1] == 0) or (x < mask.shape[1] - 1 and mask[y, x + 1] == 0) or \
#                    (y > 0 and x > 0 and mask[y - 1, x - 1] == 0) or (y > 0 and x < mask.shape[1] - 1 and mask[y - 1, x + 1] == 0) or \
#                    (y < mask.shape[0] - 1 and x > 0 and mask[y + 1, x - 1] == 0) or (y < mask.shape[0] - 1 and x < mask.shape[1] - 1 and mask[y + 1, x + 1] == 0):
#                     dist = math.sqrt((center_x - x) ** 2 + (center_y - y) ** 2)
#                     if dist < min_dist:
#                         min_dist = dist
#                         nearest_point = (x, y)
    
#     return nearest_point

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

#         # Process the segmentation masks and boxes
#         masks = results[0].masks  # Assumes that the results contain masks
#         boxes = results[0].boxes  # Get the bounding boxes

#         if masks is not None and boxes is not None:  # Check if masks and boxes are not None
#             for i, mask in enumerate(masks):
#                 class_id = int(boxes.cls[i])  # Get the class ID for the corresponding mask
#                 class_name = model.names[class_id]

#                 if class_name == "side_cup":  # Check if the detected object is a side_cup
#                     mask = mask.cpu().numpy().astype(np.uint8)  # Convert mask to numpy array if necessary and binarize it

#                     # Calculate the center of the mask
#                     M = cv2.moments(mask)
#                     if M["m00"] != 0:
#                         cX = int(M["m10"] / M["m00"])
#                         cY = int(M["m01"] / M["m00"])
#                     else:
#                         cX, cY = 0, 0

#                     # Find the nearest boundary pixel
#                     nearest_point = calculate_nearest_boundary_point(mask, cX, cY)

#                     # Draw a line from the center to the nearest boundary pixel
#                     cv2.line(annotated_frame, (cX, cY), nearest_point, (0, 255, 0), 2)

#                     # Optionally display the nearest point and the center
#                     cv2.circle(annotated_frame, (cX, cY), 5, (255, 0, 0), -1)
#                     cv2.circle(annotated_frame, nearest_point, 5, (0, 0, 255), -1)

#         # Display the frame with detections
#         cv2.imshow('YOLOv8 Real-Time Object Detection', annotated_frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()




import cv2
import numpy as np
from ultralytics import YOLO
import time
import math

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

def calculate_nearest_boundary_point(mask, center_x, center_y):
    min_dist = float('inf')
    nearest_point = (center_x, center_y)
    
    # Iterate through all pixels in the mask
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            # Check if the pixel is part of the mask
            if mask[y, x] > 0:  # Assuming mask pixels are binary (0 or 1)
                # Check 4-connected neighborhood to see if it's a boundary pixel
                if (y > 0 and mask[y - 1, x] == 0) or (y < mask.shape[0] - 1 and mask[y + 1, x] == 0) or \
                   (x > 0 and mask[y, x - 1] == 0) or (x < mask.shape[1] - 1 and mask[y, x + 1] == 0):
                    dist = math.sqrt((center_x - x) ** 2 + (center_y - y) ** 2)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_point = (x, y)
    
    return nearest_point

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

        # Process the segmentation masks and boxes
        masks = results[0].masks  # Assumes that the results contain masks
        boxes = results[0].boxes  # Get the bounding boxes

        if masks is not None and boxes is not None:  # Check if masks and boxes are not None
            for i, mask in enumerate(masks):
                class_id = int(boxes.cls[i])  # Get the class ID for the corresponding mask
                class_name = model.names[class_id]

                if class_name == "side_cup":  # Check if the detected object is a side_cup
                    mask = mask.cpu().numpy().astype(np.uint8)  # Convert mask to numpy array if necessary and binarize it

                    # Calculate the center of the mask
                    M = cv2.moments(mask)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                    else:
                        cX, cY = 0, 0

                    # Find the nearest boundary pixel
                    nearest_point = calculate_nearest_boundary_point(mask, cX, cY)
                    print(f"Center: ({cX}, {cY}), Nearest Boundary Point: {nearest_point}")

                    # Draw a line from the center to the nearest boundary pixel
                    cv2.line(annotated_frame, (cX, cY), nearest_point, (0, 255, 0), 2)

                    # Optionally display the nearest point and the center
                    cv2.circle(annotated_frame, (cX, cY), 5, (255, 0, 0), -1)
                    cv2.circle(annotated_frame, nearest_point, 5, (0, 0, 255), -1)

        # Display the frame with detections
        cv2.imshow('YOLOv8 Real-Time Object Detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
