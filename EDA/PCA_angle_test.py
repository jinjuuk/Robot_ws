# import cv2
# import numpy as np
# from ultralytics import YOLO
# import time

# # Load the YOLOv8 segmentation model
# print("Loading YOLOv8 model...")
# model = YOLO("/home/jinjuuk/dev_ws/pt_files/newjeans.pt")
# print("Model loaded.")

# # Open the camera
# print("Opening camera...")
# cap = cv2.VideoCapture(0)  # 0 for the default camera

# # Set camera resolution
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# if not cap.isOpened():
#     print("Error: Could not open video device.")
#     exit()

# print("Camera opened.")

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

# print("Trackbars created.")

# while cap.isOpened():
#     success, frame = cap.read()

#     if success:
#         # Get the current positions of the trackbars
#         confidence_threshold = cv2.getTrackbarPos('Confidence Threshold', 'YOLOv8 Real-Time Object Detection') / 100
#         iou_threshold = cv2.getTrackbarPos('IoU Threshold', 'YOLOv8 Real-Time Object Detection') / 100

#         print(f"Confidence Threshold: {confidence_threshold}, IoU Threshold: {iou_threshold}")

#         # Start timer
#         start = time.perf_counter()
        
#         # Perform prediction
#         results = model(frame, conf=confidence_threshold, iou=iou_threshold)
        
#         # End timer
#         end = time.perf_counter()
#         total_time = end - start
#         fps = 1 / total_time

#         print(f"Prediction completed in {total_time:.4f} seconds, FPS: {fps:.2f}")

#         # Annotate frame with segmentation results
#         annotated_frame = results[0].plot()

#         # Display FPS on the frame
#         cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#         # Calculate and print center points of bounding boxes and class names
#         for i, box in enumerate(results[0].boxes):
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             center_x = (x1 + x2) // 2
#             center_y = (y1 + y2) // 2
#             class_id = int(box.cls[0])
#             class_name = model.names[class_id]
#             print(f"Class: {class_name}, Center point: ({center_x}, {center_y}), Class_Id: {class_id}")

#             # Extract the mask for the current object and move it to the CPU
#             mask = results[0].masks.data[i].cpu().numpy()

#             # Verify mask is extracted correctly
#             print(f"Mask for {class_name}: {mask.shape}")

#             # Ensure the mask is in the correct shape (H, W)
#             if mask.ndim == 3:
#                 mask = mask[0]

#             # Find the closest edge pixel from the center
#             mask_indices = np.argwhere(mask > 0)
#             print(f"Mask indices shape: {mask_indices.shape}")

#             if mask_indices.size == 0:
#                 print(f"No mask indices found for {class_name}")
#                 continue

#             distances = np.sqrt((mask_indices[:, 1] - center_x) ** 2 + (mask_indices[:, 0] - center_y) ** 2)
#             closest_pixel_index = np.argmin(distances)
#             closest_pixel = mask_indices[closest_pixel_index][::-1]  # Switch (row, col) to (x, y)

#             print(f"Drawing line to closest edge: {closest_pixel}")

#             # Draw line from center to the closest pixel
#             cv2.line(annotated_frame, (center_x, center_y), tuple(closest_pixel), (255, 0, 0), 2)

#             # Draw center point and closest pixel point for debugging
#             cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 255, 0), -1)
#             cv2.circle(annotated_frame, tuple(closest_pixel), 5, (0, 0, 255), -1)

#         # Display the frame with detections
#         cv2.imshow('YOLOv8 Real-Time Object Detection', annotated_frame)

#     else:
#         print("Failed to capture frame.")

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# print("Releasing camera and closing windows.")
# cap.release()
# cv2.destroyAllWindows()









# import cv2
# import numpy as np
# from ultralytics import YOLO
# import time

# # Load the YOLOv8 segmentation model
# print("Loading YOLOv8 model...")
# model = YOLO("/home/jinjuuk/dev_ws/pt_files/newjeans.pt")
# print("Model loaded.")

# # Open the camera
# print("Opening camera...")
# cap = cv2.VideoCapture(0)  # 0 for the default camera

# # Set camera resolution
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# if not cap.isOpened():
#     print("Error: Could not open video device.")
#     exit()

# print("Camera opened.")

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

# print("Trackbars created.")

# while cap.isOpened():
#     success, frame = cap.read()

#     if success:
#         # Get the current positions of the trackbars
#         confidence_threshold = cv2.getTrackbarPos('Confidence Threshold', 'YOLOv8 Real-Time Object Detection') / 100
#         iou_threshold = cv2.getTrackbarPos('IoU Threshold', 'YOLOv8 Real-Time Object Detection') / 100

#         print(f"Confidence Threshold: {confidence_threshold}, IoU Threshold: {iou_threshold}")

#         # Start timer
#         start = time.perf_counter()
        
#         # Perform prediction
#         results = model(frame, conf=confidence_threshold, iou=iou_threshold)
        
#         # End timer
#         end = time.perf_counter()
#         total_time = end - start
#         fps = 1 / total_time

#         print(f"Prediction completed in {total_time:.4f} seconds, FPS: {fps:.2f}")

#         # Annotate frame with segmentation results
#         annotated_frame = results[0].plot()

#         # Display FPS on the frame
#         cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#         # Calculate and print center points of bounding boxes and class names
#         for i, box in enumerate(results[0].boxes):
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             center_x = (x1 + x2) // 2
#             center_y = (y1 + y2) // 2
#             class_id = int(box.cls[0])
#             class_name = model.names[class_id]
#             print(f"Class: {class_name}, Center point: ({center_x}, {center_y}), Class_Id: {class_id}")

#             # Extract the mask for the current object and move it to the CPU
#             mask = results[0].masks.data[i].cpu().numpy()

#             # Verify mask is extracted correctly
#             print(f"Mask for {class_name}: {mask.shape}")

#             # Ensure the mask is in the correct shape (H, W)
#             if mask.ndim == 3:
#                 mask = mask[0]

#             # Find the closest edge pixel from the center
#             mask_indices = np.argwhere(mask > 0)
#             print(f"Mask indices shape: {mask_indices.shape}")

#             if mask_indices.size == 0:
#                 print(f"No mask indices found for {class_name}")
#                 continue

#             distances = np.sqrt((mask_indices[:, 1] - center_x) ** 2 + (mask_indices[:, 0] - center_y) ** 2)
#             closest_pixel_index = np.argmin(distances)
#             closest_pixel = mask_indices[closest_pixel_index][::-1]  # Switch (row, col) to (x, y)

#             print(f"Drawing line to closest edge: {closest_pixel}")

#             # Draw line from center to the closest pixel
#             cv2.line(annotated_frame, (center_x, center_y), tuple(closest_pixel), (255, 0, 0), 2)

#             # Draw center point and closest pixel point for debugging
#             cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 255, 0), -1)
#             cv2.circle(annotated_frame, tuple(closest_pixel), 5, (0, 0, 255), -1)

#         # Display the frame with detections
#         cv2.imshow('YOLOv8 Real-Time Object Detection', annotated_frame)

#     else:
#         print("Failed to capture frame.")

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# print("Releasing camera and closing windows.")
# cap.release()
# cv2.destroyAllWindows()





# import cv2
# import numpy as np
# from ultralytics import YOLO
# import time

# # Load the YOLOv8 segmentation model
# print("Loading YOLOv8 model...")
# model = YOLO("/home/jinjuuk/dev_ws/pt_files/newjeans.pt")
# print("Model loaded.")

# # Open the camera
# print("Opening camera...")
# cap = cv2.VideoCapture(0)  # 0 for the default camera

# # Set camera resolution
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# if not cap.isOpened():
#     print("Error: Could not open video device.")
#     exit()

# print("Camera opened.")

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

# print("Trackbars created.")

# while cap.isOpened():
#     success, frame = cap.read()

#     if success:
#         # Get the current positions of the trackbars
#         confidence_threshold = cv2.getTrackbarPos('Confidence Threshold', 'YOLOv8 Real-Time Object Detection') / 100
#         iou_threshold = cv2.getTrackbarPos('IoU Threshold', 'YOLOv8 Real-Time Object Detection') / 100

#         print(f"Confidence Threshold: {confidence_threshold}, IoU Threshold: {iou_threshold}")

#         # Start timer
#         start = time.perf_counter()
        
#         # Perform prediction
#         results = model(frame, conf=confidence_threshold, iou=iou_threshold)
        
#         # End timer
#         end = time.perf_counter()
#         total_time = end - start
#         fps = 1 / total_time

#         print(f"Prediction completed in {total_time:.4f} seconds, FPS: {fps:.2f}")

#         # Annotate frame with segmentation results
#         annotated_frame = results[0].plot()

#         # Display FPS on the frame
#         cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#         # Calculate and print center points of bounding boxes and class names
#         for i, box in enumerate(results[0].boxes):
#             x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
#             center_x = (x1 + x2) // 2
#             center_y = (y1 + y2) // 2
#             class_id = int(box.cls[0].cpu().numpy())
#             class_name = model.names[class_id]
#             print(f"Class: {class_name}, Center point: ({center_x}, {center_y}), Class_Id: {class_id}")

#             # Draw line from center point to the bottom edge (y축으로 선 그리기)
#             cv2.line(annotated_frame, (center_x, center_y), (center_x, y2), (255, 0, 0), 2)

#             # Extract the mask for the current object and move it to the CPU
#             mask = results[0].masks.data[i].cpu().numpy()

#             # Verify mask is extracted correctly
#             print(f"Mask for {class_name}: {mask.shape}")

#             # Ensure the mask is in the correct shape (H, W)
#             if mask.ndim == 3:
#                 mask = mask[0]

#             # Find the farthest edge pixel from the center
#             mask_indices = np.argwhere(mask > 0)
#             print(f"Mask indices shape: {mask_indices.shape}")

#             if mask_indices.size == 0:
#                 print(f"No mask indices found for {class_name}")
#                 continue

#             distances = np.sqrt((mask_indices[:, 1] - center_x) ** 2 + (mask_indices[:, 0] - center_y) ** 2)
#             farthest_pixel_index = np.argmax(distances)
#             farthest_pixel = mask_indices[farthest_pixel_index][::-1]  # Switch (row, col) to (x, y)

#             print(f"Drawing line to farthest edge: {farthest_pixel}")

#             # Draw center point and farthest pixel point for debugging
#             cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 255, 0), -1)
#             cv2.circle(annotated_frame, tuple(farthest_pixel), 5, (0, 0, 255), -1)

#             # Draw line from center to the farthest pixel (객체의 방향이 바뀌면 외각 픽셀의 점도 바뀌기)
#             cv2.line(annotated_frame, (center_x, center_y), tuple(farthest_pixel), (0, 0, 255), 2)

#         # Display the frame with detections
#         cv2.imshow('YOLOv8 Real-Time Object Detection', annotated_frame)

#     else:
#         print("Failed to capture frame.")

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# print("Releasing camera and closing windows.")
# cap.release()
# cv2.destroyAllWindows()





# import cv2
# import numpy as np
# from ultralytics import YOLO
# import time

# # Load the YOLOv8 segmentation model
# print("Loading YOLOv8 model...")
# model = YOLO("/home/jinjuuk/dev_ws/pt_files/newjeans.pt")
# print("Model loaded.")

# # Open the camera
# print("Opening camera...")
# cap = cv2.VideoCapture(0)  # 0 for the default camera

# # Set camera resolution
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# if not cap.isOpened():
#     print("Error: Could not open video device.")
#     exit()

# print("Camera opened.")

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

# print("Trackbars created.")

# while cap.isOpened():
#     success, frame = cap.read()

#     if success:
#         # Get the current positions of the trackbars
#         confidence_threshold = cv2.getTrackbarPos('Confidence Threshold', 'YOLOv8 Real-Time Object Detection') / 100
#         iou_threshold = cv2.getTrackbarPos('IoU Threshold', 'YOLOv8 Real-Time Object Detection') / 100

#         print(f"Confidence Threshold: {confidence_threshold}, IoU Threshold: {iou_threshold}")

#         # Start timer
#         start = time.perf_counter()
        
#         # Perform prediction
#         results = model(frame, conf=confidence_threshold, iou=iou_threshold)
        
#         # End timer
#         end = time.perf_counter()
#         total_time = end - start
#         fps = 1 / total_time

#         print(f"Prediction completed in {total_time:.4f} seconds, FPS: {fps:.2f}")

#         # Annotate frame with segmentation results
#         annotated_frame = results[0].plot()

#         # Display FPS on the frame
#         cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#         # Calculate and print center points of bounding boxes and class names
#         for i, box in enumerate(results[0].boxes):
#             x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
#             center_x = (x1 + x2) // 2
#             center_y = (y1 + y2) // 2
#             class_id = int(box.cls[0].cpu().numpy())
#             class_name = model.names[class_id]
#             print(f"Class: {class_name}, Center point: ({center_x}, {center_y}), Class_Id: {class_id}")

#             # Draw line from center point to the bottom edge (y축으로 선 그리기)
#             cv2.line(annotated_frame, (center_x, center_y), (center_x, y2), (255, 0, 0), 2)

#             # Extract the mask for the current object and move it to the CPU
#             mask = results[0].masks.data[i].cpu().numpy()

#             # Verify mask is extracted correctly
#             print(f"Mask for {class_name}: {mask.shape}")

#             # Ensure the mask is in the correct shape (H, W)
#             if mask.ndim == 3:
#                 mask = mask[0]

#             # Find the farthest edge pixel from the center along the y-axis
#             mask_indices = np.argwhere(mask > 0)
#             print(f"Mask indices shape: {mask_indices.shape}")

#             if mask_indices.size == 0:
#                 print(f"No mask indices found for {class_name}")
#                 continue

#             # Find the farthest point along the y-axis
#             distances = np.abs(mask_indices[:, 0] - center_y)
#             farthest_pixel_index = np.argmax(distances)
#             farthest_pixel = mask_indices[farthest_pixel_index][::-1]  # Switch (row, col) to (x, y)

#             print(f"Drawing line to farthest edge along y-axis: {farthest_pixel}")

#             # Draw center point and farthest pixel point for debugging
#             cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 255, 0), -1)
#             cv2.circle(annotated_frame, tuple(farthest_pixel), 5, (0, 0, 255), -1)

#             # Draw line from center to the farthest pixel along the y-axis (객체의 방향이 바뀌면 외각 픽셀의 점도 바뀌기)
#             cv2.line(annotated_frame, (center_x, center_y), tuple(farthest_pixel), (0, 0, 255), 2)

#         # Display the frame with detections
#         cv2.imshow('YOLOv8 Real-Time Object Detection', annotated_frame)

#     else:
#         print("Failed to capture frame.")

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# print("Releasing camera and closing windows.")
# cap.release()
# cv2.destroyAllWindows()




# import cv2
# import numpy as np
# from ultralytics import YOLO
# import time

# # Load the YOLOv8 segmentation model
# print("Loading YOLOv8 model...")
# model = YOLO("/home/jinjuuk/dev_ws/pt_files/newjeans.pt")
# print("Model loaded.")

# # Open the camera
# print("Opening camera...")
# cap = cv2.VideoCapture(0)  # 0 for the default camera

# # Set camera resolution
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# if not cap.isOpened():
#     print("Error: Could not open video device.")
#     exit()

# print("Camera opened.")

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

# print("Trackbars created.")

# while cap.isOpened():
#     success, frame = cap.read()

#     if success:
#         # Get the current positions of the trackbars
#         confidence_threshold = cv2.getTrackbarPos('Confidence Threshold', 'YOLOv8 Real-Time Object Detection') / 100
#         iou_threshold = cv2.getTrackbarPos('IoU Threshold', 'YOLOv8 Real-Time Object Detection') / 100

#         print(f"Confidence Threshold: {confidence_threshold}, IoU Threshold: {iou_threshold}")

#         # Start timer
#         start = time.perf_counter()
        
#         # Perform prediction
#         results = model(frame, conf=confidence_threshold, iou=iou_threshold)
        
#         # End timer
#         end = time.perf_counter()
#         total_time = end - start
#         fps = 1 / total_time

#         print(f"Prediction completed in {total_time:.4f} seconds, FPS: {fps:.2f}")

#         # Annotate frame with segmentation results
#         annotated_frame = results[0].plot()

#         # Display FPS on the frame
#         cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#         # Calculate and print center points of bounding boxes and class names
#         for i, box in enumerate(results[0].boxes):
#             x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
#             center_x = (x1 + x2) // 2
#             center_y = (y1 + y2) // 2
#             class_id = int(box.cls[0].cpu().numpy())
#             class_name = model.names[class_id]
#             print(f"Class: {class_name}, Center point: ({center_x}, {center_y}), Class_Id: {class_id}")

#             # Draw line from center point to the bottom edge (y축으로 선 그리기)
#             cv2.line(annotated_frame, (center_x, y1), (center_x, y2), (255, 0, 0), 2)

#             # Extract the mask for the current object and move it to the CPU
#             mask = results[0].masks.data[i].cpu().numpy()

#             # Verify mask is extracted correctly
#             print(f"Mask for {class_name}: {mask.shape}")

#             # Ensure the mask is in the correct shape (H, W)
#             if mask.ndim == 3:
#                 mask = mask[0]

#             # Find the farthest edge pixel from the center along the negative y-axis
#             mask_indices = np.argwhere(mask > 0)
#             print(f"Mask indices shape: {mask_indices.shape}")

#             if mask_indices.size == 0:
#                 print(f"No mask indices found for {class_name}")
#                 continue

#             # Filter out points below the center_y to get points only in the negative y direction
#             negative_y_indices = mask_indices[mask_indices[:, 0] < center_y]

#             if negative_y_indices.size == 0:
#                 print(f"No negative y direction mask indices found for {class_name}")
#                 continue

#             distances = np.abs(negative_y_indices[:, 0] - center_y)
#             farthest_pixel_index = np.argmax(distances)
#             farthest_pixel = negative_y_indices[farthest_pixel_index][::-1]  # Switch (row, col) to (x, y)

#             print(f"Drawing line to farthest edge along negative y-axis: {farthest_pixel}")

#             # Draw center point and farthest pixel point for debugging
#             cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 255, 0), -1)
#             cv2.circle(annotated_frame, tuple(farthest_pixel), 5, (0, 0, 255), -1)

#             # Draw line from center to the farthest pixel along the negative y-axis (객체의 방향이 바뀌면 외각 픽셀의 점도 바뀌기)
#             cv2.line(annotated_frame, (center_x, center_y), tuple(farthest_pixel), (0, 0, 255), 2)

#         # Display the frame with detections
#         cv2.imshow('YOLOv8 Real-Time Object Detection', annotated_frame)

#     else:
#         print("Failed to capture frame.")

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# print("Releasing camera and closing windows.")
# cap.release()
# cv2.destroyAllWindows()




# import cv2
# import numpy as np
# from ultralytics import YOLO
# import time

# # Load the YOLOv8 segmentation model
# print("Loading YOLOv8 model...")
# model = YOLO("/home/jinjuuk/dev_ws/pt_files/newjeans.pt")
# print("Model loaded.")

# # Open the camera
# print("Opening camera...")
# cap = cv2.VideoCapture(0)  # 0 for the default camera

# # Set camera resolution
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# if not cap.isOpened():
#     print("Error: Could not open video device.")
#     exit()

# print("Camera opened.")

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

# print("Trackbars created.")

# while cap.isOpened():
#     success, frame = cap.read()

#     if success:
#         # Get the current positions of the trackbars
#         confidence_threshold = cv2.getTrackbarPos('Confidence Threshold', 'YOLOv8 Real-Time Object Detection') / 100
#         iou_threshold = cv2.getTrackbarPos('IoU Threshold', 'YOLOv8 Real-Time Object Detection') / 100

#         print(f"Confidence Threshold: {confidence_threshold}, IoU Threshold: {iou_threshold}")

#         # Start timer
#         start = time.perf_counter()
        
#         # Perform prediction
#         results = model(frame, conf=confidence_threshold, iou=iou_threshold)
        
#         # End timer
#         end = time.perf_counter()
#         total_time = end - start
#         fps = 1 / total_time

#         print(f"Prediction completed in {total_time:.4f} seconds, FPS: {fps:.2f}")

#         # Annotate frame with segmentation results
#         annotated_frame = results[0].plot()

#         # Display FPS on the frame
#         cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#         # Calculate and print center points of bounding boxes and class names
#         for i, box in enumerate(results[0].boxes):
#             x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
#             center_x = (x1 + x2) // 2
#             center_y = (y1 + y2) // 2
#             class_id = int(box.cls[0].cpu().numpy())
#             class_name = model.names[class_id]
#             print(f"Class: {class_name}, Center point: ({center_x}, {center_y}), Class_Id: {class_id}")

#             # Draw line from top to bottom through the center (y축으로 bbox의 중앙지점을 통과하는 선 그리기)
#             cv2.line(annotated_frame, (center_x, y1), (center_x, y2), (255, 0, 0), 2)

#             # Extract the mask for the current object and move it to the CPU
#             mask = results[0].masks.data[i].cpu().numpy()

#             # Verify mask is extracted correctly
#             print(f"Mask for {class_name}: {mask.shape}")

#             # Ensure the mask is in the correct shape (H, W)
#             if mask.ndim == 3:
#                 mask = mask[0]

#             # Find the farthest edge pixel from the center along the negative y-axis
#             mask_indices = np.argwhere(mask > 0)
#             print(f"Mask indices shape: {mask_indices.shape}")

#             if mask_indices.size == 0:
#                 print(f"No mask indices found for {class_name}")
#                 continue

#             # Filter out points below the center_y to get points only in the negative y direction
#             negative_y_indices = mask_indices[mask_indices[:, 0] < center_y]

#             if negative_y_indices.size == 0:
#                 print(f"No negative y direction mask indices found for {class_name}")
#                 continue

#             # Find the farthest point along the negative y-axis
#             distances = np.abs(negative_y_indices[:, 0] - center_y)
#             farthest_pixel_index = np.argmax(distances)
#             farthest_pixel = negative_y_indices[farthest_pixel_index][::-1]  # Switch (row, col) to (x, y)

#             print(f"Drawing line to farthest edge along negative y-axis: {farthest_pixel}")

#             # Draw center point and farthest pixel point for debugging
#             cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 255, 0), -1)
#             cv2.circle(annotated_frame, tuple(farthest_pixel), 5, (0, 0, 255), -1)

#             # Draw line from center to the farthest pixel along the negative y-axis (객체의 방향이 바뀌면 외각 픽셀의 점도 바뀌기)
#             cv2.line(annotated_frame, (center_x, center_y), tuple(farthest_pixel), (0, 0, 255), 2)

#         # Display the frame with detections
#         cv2.imshow('YOLOv8 Real-Time Object Detection', annotated_frame)

#     else:
#         print("Failed to capture frame.")

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# print("Releasing camera and closing windows.")
# cap.release()
# cv2.destroyAllWindows()




# import cv2
# import numpy as np
# from ultralytics import YOLO
# import time

# def draw_arrow(image, p1, p2, color, thickness=2, tip_length=0.2):
#     p1 = tuple(map(int, p1))
#     p2 = tuple(map(int, p2))
#     cv2.arrowedLine(image, p1, p2, color, thickness, tipLength=tip_length)

# # Load the YOLOv8 segmentation model
# print("Loading YOLOv8 model...")
# model = YOLO("/home/jinjuuk/dev_ws/pt_files/newjeans.pt")
# print("Model loaded.")

# # Open the camera
# print("Opening camera...")
# cap = cv2.VideoCapture(0)  # 0 for the default camera

# # Set camera resolution
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# if not cap.isOpened():
#     print("Error: Could not open video device.")
#     exit()

# print("Camera opened.")

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

# print("Trackbars created.")

# while cap.isOpened():
#     success, frame = cap.read()

#     if success:
#         # Get the current positions of the trackbars
#         confidence_threshold = cv2.getTrackbarPos('Confidence Threshold', 'YOLOv8 Real-Time Object Detection') / 100
#         iou_threshold = cv2.getTrackbarPos('IoU Threshold', 'YOLOv8 Real-Time Object Detection') / 100

#         print(f"Confidence Threshold: {confidence_threshold}, IoU Threshold: {iou_threshold}")

#         # Start timer
#         start = time.perf_counter()
        
#         # Perform prediction
#         results = model(frame, conf=confidence_threshold, iou=iou_threshold)
        
#         # End timer
#         end = time.perf_counter()
#         total_time = end - start
#         fps = 1 / total_time

#         print(f"Prediction completed in {total_time:.4f} seconds, FPS: {fps:.2f}")

#         # Annotate frame with segmentation results
#         annotated_frame = results[0].plot()

#         # Display FPS on the frame
#         cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#         # Calculate and print center points of bounding boxes and class names
#         for i, box in enumerate(results[0].boxes):
#             x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
#             center_x = (x1 + x2) // 2
#             center_y = (y1 + y2) // 2
#             class_id = int(box.cls[0].cpu().numpy())
#             class_name = model.names[class_id]
#             print(f"Class: {class_name}, Center point: ({center_x}, {center_y}), Class_Id: {class_id}")

#             # Draw line from top to bottom through the center (y축으로 bbox의 중앙지점을 통과하는 선 그리기)
#             cv2.line(annotated_frame, (center_x, y1), (center_x, y2), (255, 0, 0), 2)

#             # Extract the mask for the current object and move it to the CPU
#             mask = results[0].masks.data[i].cpu().numpy()

#             # Verify mask is extracted correctly
#             print(f"Mask for {class_name}: {mask.shape}")

#             # Ensure the mask is in the correct shape (H, W)
#             if mask.ndim == 3:
#                 mask = mask[0]

#             # Find the non-zero indices of the mask
#             mask_indices = np.argwhere(mask > 0)
#             print(f"Mask indices shape: {mask_indices.shape}")

#             if mask_indices.size == 0:
#                 print(f"No mask indices found for {class_name}")
#                 continue

#             # Calculate principal component
#             mean, eigen_vectors = cv2.PCACompute(mask_indices.astype(np.float32), mean=np.array([]))
#             angle = np.arctan2(eigen_vectors[0, 1], eigen_vectors[0, 0]) * 180 / np.pi

#             print(f"Object angle: {angle:.2f} degrees")

#             # Draw an arrow to represent the principal axis
#             arrow_length = 100
#             p1 = (center_x, center_y)
#             p2 = (center_x + int(arrow_length * eigen_vectors[0, 0]),
#                   center_y + int(arrow_length * eigen_vectors[0, 1]))
#             draw_arrow(annotated_frame, p1, p2, (0, 255, 255), 2)

#             # Draw center point for debugging
#             cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 255, 0), -1)

#         # Display the frame with detections
#         cv2.imshow('YOLOv8 Real-Time Object Detection', annotated_frame)

#     else:
#         print("Failed to capture frame.")

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# print("Releasing camera and closing windows.")
# cap.release()
# cv2.destroyAllWindows()



import cv2
import numpy as np
from ultralytics import YOLO
import time

def draw_arrow(image, p1, p2, color, thickness=2, tip_length=0.2):
    p1 = tuple(map(int, p1))
    p2 = tuple(map(int, p2))
    cv2.arrowedLine(image, p1, p2, color, thickness, tipLength=tip_length)

# Load the YOLOv8 segmentation model
print("Loading YOLOv8 model...")
model = YOLO("/home/jinjuuk/dev_ws/pt_files/newjeans.pt")
print("Model loaded.")

# Open the camera
print("Opening camera...")
cap = cv2.VideoCapture(0)  # 0 for the default camera

# Set camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Could not open video device.")
    exit()

print("Camera opened.")

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

print("Trackbars created.")

while cap.isOpened():
    success, frame = cap.read()

    if success:
        # Get the current positions of the trackbars
        confidence_threshold = cv2.getTrackbarPos('Confidence Threshold', 'YOLOv8 Real-Time Object Detection') / 100
        iou_threshold = cv2.getTrackbarPos('IoU Threshold', 'YOLOv8 Real-Time Object Detection') / 100

        print(f"Confidence Threshold: {confidence_threshold}, IoU Threshold: {iou_threshold}")

        # Start timer
        start = time.perf_counter()
        
        # Perform prediction
        results = model(frame, conf=confidence_threshold, iou=iou_threshold)
        
        # End timer
        end = time.perf_counter()
        total_time = end - start
        fps = 1 / total_time

        print(f"Prediction completed in {total_time:.4f} seconds, FPS: {fps:.2f}")

        # Annotate frame with segmentation results
        annotated_frame = results[0].plot()

        # Display FPS on the frame
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Calculate and print center points of bounding boxes and class names
        for i, box in enumerate(results[0].boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            class_id = int(box.cls[0].cpu().numpy())
            class_name = model.names[class_id]
            print(f"Class: {class_name}, Center point: ({center_x}, {center_y}), Class_Id: {class_id}")

            # Draw line from top to bottom through the center (y축으로 bbox의 중앙지점을 통과하는 선 그리기)
            cv2.line(annotated_frame, (center_x, y1), (center_x, y2), (255, 0, 0), 2)

            # Extract the mask for the current object and move it to the CPU
            mask = results[0].masks.data[i].cpu().numpy()

            # Verify mask is extracted correctly
            print(f"Mask for {class_name}: {mask.shape}")

            # Ensure the mask is in the correct shape (H, W)
            if mask.ndim == 3:
                mask = mask[0]

            # Use GrabCut to refine the mask
            rect = (x1, y1, x2 - x1, y2 - y1)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            grabcut_mask = np.zeros(mask.shape, np.uint8)
            grabcut_mask[mask > 0] = cv2.GC_FGD
            grabcut_mask[mask == 0] = cv2.GC_BGD
            cv2.grabCut(frame, grabcut_mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)

            # Create binary mask from grabcut result
            refined_mask = np.where((grabcut_mask == 2) | (grabcut_mask == 0), 0, 1).astype('uint8')

            # Find the non-zero indices of the refined mask
            refined_indices = np.argwhere(refined_mask > 0)
            print(f"Refined mask indices shape: {refined_indices.shape}")

            if refined_indices.size == 0:
                print(f"No refined mask indices found for {class_name}")
                continue

            # Calculate principal component
            mean, eigen_vectors = cv2.PCACompute(refined_indices.astype(np.float32), mean=np.array([]))
            angle = np.arctan2(eigen_vectors[0, 1], eigen_vectors[0, 0]) * 180 / np.pi

            print(f"Object angle: {angle:.2f} degrees")

            # Draw an arrow to represent the principal axis
            arrow_length = 100
            p1 = (center_x, center_y)
            p2 = (center_x + int(arrow_length * eigen_vectors[0, 0]),
                  center_y + int(arrow_length * eigen_vectors[0, 1]))
            draw_arrow(annotated_frame, p1, p2, (0, 255, 255), 2)

            # Draw center point for debugging
            cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 255, 0), -1)

            # Draw contour on the original frame
            contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(annotated_frame, contours, -1, (0, 255, 0), 2)

        # Display the frame with detections
        cv2.imshow('YOLOv8 Real-Time Object Detection', annotated_frame)

    else:
        print("Failed to capture frame.")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Releasing camera and closing windows.")
cap.release()
cv2.destroyAllWindows()
