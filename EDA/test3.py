# # # [[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[문제점: opencv 객체는 인식하는데, model로 인식한 객체는 잡지 않음.]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]

# import cv2
# import numpy as np
# import time
# import torch
# import cv2.aruco as aruco
# import mediapipe as mp
# from ultralytics import YOLO

# def detect_star_shape(image, canny_thresh1, canny_thresh2):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     edged = cv2.Canny(blurred, canny_thresh1, canny_thresh2)
#     contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     if len(contours) == 1 or len(contours) == 2:
#         contour = contours[0]
#         epsilon = 0.02 * cv2.arcLength(contour, True)
#         approx = cv2.approxPolyDP(contour, epsilon, True)
#         if len(approx) >= 10:  # Assuming a star has approximately 10 points
#             return True, approx, edged, len(contours)
#     return False, None, edged, len(contours)

# def nothing(x):
#     pass

# def draw_rounded_rectangle(image, top_left, width, height, radius, color, thickness):
#     x, y = top_left
#     top_right = (x + width, y)
#     bottom_right = (x + width, y + height)
#     bottom_left = (x, y + height)

#     # Draw straight lines
#     cv2.line(image, (x + radius, y), (x + width - radius, y), color, thickness)
#     cv2.line(image, (x, y + radius), (x, y + height), color, thickness)
#     cv2.line(image, (x + width, y + radius), (x + width, y + height), color, thickness)
#     cv2.line(image, (x, y + height), (x + width, y + height), color, thickness)

#     # Draw arcs for rounded corners
#     cv2.ellipse(image, (x + radius, y + radius), (radius, radius), 180, 0, 90, color, thickness)
#     cv2.ellipse(image, (x + width - radius, y + radius), (radius, radius), 270, 0, 90, color, thickness)

# cap = cv2.VideoCapture(0)

# # YOLOv8 모델 로드
# model = YOLO('/home/jinjuuk/dev_ws/pt_files/segmentation_s_batch16_freeze8.pt')
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# if torch.cuda.is_available():
#     model.to(device)
# print(f'Using device: {device}')

# # Mediapipe hand detection
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

# # ArUco marker detection
# aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
# parameters = aruco.DetectorParameters()
# last_seen = {}

# # Trackbars for thresholds and brightness
# cv2.namedWindow('Frame')
# cv2.createTrackbar('Threshold', 'Frame', 0, 20000, nothing)
# cv2.createTrackbar('Canny Thresh1', 'Frame', 50, 255, nothing)
# cv2.createTrackbar('Canny Thresh2', 'Frame', 112, 255, nothing)
# cv2.createTrackbar('Brightness', 'Frame', 50, 100, nothing)
# cv2.createTrackbar('Confidence', 'Frame', 50, 100, nothing)

# # Trackbars for trash detection
# cv2.createTrackbar('Diff_Thresh', 'Frame', 45, 255, nothing)
# cv2.createTrackbar('Hue Min', 'Frame', 0, 179, nothing)
# cv2.createTrackbar('Hue Max', 'Frame', 179, 179, nothing)
# cv2.createTrackbar('Sat Min', 'Frame', 0, 255, nothing)
# cv2.createTrackbar('Sat Max', 'Frame', 255, 255, nothing)
# cv2.createTrackbar('Val Min', 'Frame', 0, 255, nothing)
# cv2.createTrackbar('Val Max', 'Frame', 255, 255, nothing)

# # ROI definitions
# roi_x_large = 55
# roi_y_large = 0
# roi_width_large = 485
# roi_height_large = 315

# roi_x_medium = 270
# roi_y_medium = 0
# roi_width_medium = 270
# roi_height_medium = 60

# roi_x_small = 464
# roi_y_small = 118
# roi_width_small = 35
# roi_height_small = 35

# fgbg = cv2.createBackgroundSubtractorMOG2()

# # 초기 배경 이미지 변수
# initial_gray = None
# post_cleanup_gray = None
# detection_enabled = False
# cleanup_detection_enabled = False
# yolo_detection_enabled = False  # New variable to control YOLO detection

# # 카메라 매트릭스와 왜곡 계수 설정
# camera_matrix = np.array([[474.51901407, 0, 302.47811758],
#                           [0, 474.18970657, 250.66191453],
#                           [0, 0, 1]])
# dist_coeffs = np.array([[-0.06544764, -0.07654065, -0.00761827, -0.00279316, 0.08062307]])

# # 픽셀 단위에서 밀리미터 단위로 변환 비율
# pixel_to_mm_ratio = 1.55145

# # 로봇 좌표계의 원점 (픽셀 단위)
# robot_origin_x = 295
# robot_origin_y = 184

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to capture frame from camera.")
#         break

#     # Trackbar values
#     threshold = cv2.getTrackbarPos('Threshold', 'Frame')
#     canny_thresh1 = cv2.getTrackbarPos('Canny Thresh1', 'Frame')
#     canny_thresh2 = cv2.getTrackbarPos('Canny Thresh2', 'Frame')
#     brightness = cv2.getTrackbarPos('Brightness', 'Frame')
#     confidence_threshold = cv2.getTrackbarPos('Confidence', 'Frame') / 100.0

#     # Trackbar values for trash detection
#     diff_thresh = cv2.getTrackbarPos('Diff_Thresh', 'Frame')
#     h_min = cv2.getTrackbarPos('Hue Min', 'Frame')
#     h_max = cv2.getTrackbarPos('Hue Max', 'Frame')
#     s_min = cv2.getTrackbarPos('Sat Min', 'Frame')
#     s_max = cv2.getTrackbarPos('Sat Max', 'Frame')
#     v_min = cv2.getTrackbarPos('Val Min', 'Frame')
#     v_max = cv2.getTrackbarPos('Val Max', 'Frame')

#     frame = cv2.convertScaleAbs(frame, alpha=1, beta=(brightness - 50) * 2)

#     fgmask = fgbg.apply(frame)

#     # Intrusion detection
#     roi_large = fgmask[roi_y_large:roi_y_large + roi_height_large, roi_x_large:roi_x_large + roi_width_large]
#     intrusion_detected = np.sum(roi_large) > threshold

#     warning_detected = False  # Initialize warning detected flag
#     trash_detected = False  # Initialize trash detection flag

#     # Mediapipe hand detection
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     result = hands.process(frame_rgb)
#     if result.multi_hand_landmarks:
#         for hand_landmarks in result.multi_hand_landmarks:
#             for landmark in hand_landmarks.landmark:
#                 x = int(landmark.x * frame.shape[1])
#                 y = int(landmark.y * frame.shape[0])
#                 if roi_x_large < x < roi_x_large + roi_width_large and roi_y_large < y < roi_y_large + roi_height_large and intrusion_detected:
#                     warning_detected = True
#                     break
#             mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#     # ArUco marker detection
#     roi_medium = frame[roi_y_medium:roi_y_medium + roi_height_medium, roi_x_medium:roi_x_medium + roi_width_medium]
#     corners, ids, rejectedImgPoints = aruco.detectMarkers(roi_medium, aruco_dict, parameters=parameters)
#     current_time = time.time()

#     if ids is not None and len(ids) > 0:
#         aruco.drawDetectedMarkers(roi_medium, corners, ids)
#         for id in ids.flatten():
#             last_seen[id] = current_time
#         for id, last_time in list(last_seen.items()):
#             if current_time - last_time > 3:
#                 cv2.putText(frame, f"Action executed for marker ID {id}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#                 del last_seen[id]
#                 break

#     # Star shape detection
#     small_roi = frame[roi_y_small:roi_y_small + roi_height_small, roi_x_small:roi_x_small + roi_width_small]
#     star_detected, star_contour, edged, contour_count = detect_star_shape(small_roi, canny_thresh1, canny_thresh2)
#     if star_detected:
#         star_contour += [roi_x_small, roi_y_small]
#         cv2.drawContours(frame, [star_contour], -1, (0, 255, 0), 3)
#         cv2.putText(frame, "Star detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
#     else:
#         cv2.putText(frame, "Star not detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

#     # Trash detection
#     frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     frame_gray = cv2.GaussianBlur(frame_gray, (21, 21), 0)
#     hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#     # 초기값 설정 또는 해제
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('s'):
#         initial_gray = frame_gray
#         detection_enabled = True
#         yolo_detection_enabled = True  # Enable YOLO detection as well
#         print("Initial background set, detection enabled.")

#     # 검출 활성화된 경우에만 실행
#     if detection_enabled and initial_gray is not None and not cleanup_detection_enabled:
#         # 초기 이미지와 현재 이미지의 차이 계산
#         frame_delta = cv2.absdiff(initial_gray, frame_gray)
#         # 차이 이미지의 임계값 적용
#         _, diff_mask = cv2.threshold(frame_delta, diff_thresh, 255, cv2.THRESH_BINARY)
#         diff_mask = cv2.dilate(diff_mask, None, iterations=2)
#         # Canny 엣지 검출
#         edges = cv2.Canny(frame_gray, canny_thresh1, canny_thresh2)
#         edges = cv2.dilate(edges, None, iterations=1)
#         edges = cv2.bitwise_and(edges, edges, mask=diff_mask)
#         # HSV 색상 범위에 따른 마스크 생성
#         lower_bound = np.array([h_min, s_min, v_min])
#         upper_bound = np.array([h_max, s_max, v_max])
#         hsv_mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
#         # 탐지 영역 마스크 생성 (마커와 별 인식 영역 제외)
#         detection_mask = np.zeros(diff_mask.shape, dtype=np.uint8)
#         detection_mask[roi_y_large:roi_y_large + roi_height_large, roi_x_large:roi_x_large + roi_width_large] = 255
#         detection_mask[roi_y_medium:roi_y_medium + roi_height_medium, roi_x_medium:roi_x_medium + roi_width_medium] = 0
#         detection_mask[roi_y_small:roi_y_small + roi_height_small, roi_x_small:roi_x_small + roi_width_small] = 0
#         # 최종 마스크 적용
#         combined_mask = cv2.bitwise_or(diff_mask, edges)
#         combined_mask = cv2.bitwise_and(combined_mask, combined_mask, mask=detection_mask)
#         combined_mask = cv2.bitwise_and(combined_mask, hsv_mask)
#         # 윤곽선 검출
#         contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         detected_objects = []

#         for cnt in contours:
#             if cv2.contourArea(cnt) > 80:  # 최소 면적 기준
#                 x, y, w, h = cv2.boundingRect(cnt)
#                 # 탐지된 객체가 지정된 탐지 영역 내에 있는지 확인
#                 if (roi_x_large <= x <= roi_x_large + roi_width_large and
#                     roi_y_large <= y <= roi_y_large + roi_height_large):
#                     detected_objects.append((x, y, w, h))
#                     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                     trash_detected = True  # Set trash detected flag

#     # YOLOv8 object detection within the large ROI
#     if yolo_detection_enabled:
#         yolo_roi = frame[roi_y_large:roi_y_large + roi_height_large, roi_x_large:roi_x_large + roi_width_large]
#         results = model(yolo_roi)
#         for result in results:
#             boxes = result.boxes
#             for box in boxes:
#                 cls_id = int(box.cls)
#                 if cls_id == 0:
#                     continue
#                 label = model.names[cls_id]
#                 confidence = box.conf.item()
#                 if confidence >= confidence_threshold:
#                     bbox = box.xyxy[0].tolist()
#                     x1, y1, x2, y2 = map(int, bbox)
#                     x1 += roi_x_large
#                     y1 += roi_y_large
#                     x2 += roi_x_large
#                     y2 += roi_y_large
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                     if label == 'cup' or label == 'star':
#                         text = f'{label}'
#                         text_x = x1
#                         text_y = y1 - 10 if y1 - 10 > 10 else y1 + 10
#                         cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#                     trash_detected = True  # Set trash detected flag

#     # Draw rounded ROI rectangle
#     draw_rounded_rectangle(frame, (roi_x_large, roi_y_large), roi_width_large, roi_height_large, 20, (255, 0, 0), 2)
#     # Draw other ROI rectangles
#     cv2.rectangle(frame, (roi_x_medium, roi_y_medium), (roi_x_medium + roi_width_medium, roi_y_medium + roi_height_medium), (0, 255, 0), 2)
#     cv2.rectangle(frame, (roi_x_small, roi_y_small), (roi_x_small + roi_width_small, roi_y_small + roi_height_small), (0, 0, 255), 2)
#     cv2.putText(frame, f"Contours: {contour_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
#     # Display the edged image in the small ROI
#     frame[roi_y_small:roi_y_small + roi_height_small, roi_x_small:roi_x_small + roi_width_small] = cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR)

#     # Display "Warning Detected" message if a hand is detected
#     if warning_detected:
#         cv2.putText(frame, 'WARNING DETECTED!', (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

#     # Display "Trash Detected" message and perform coordinate transformation
#     if trash_detected:
#         cv2.putText(frame, 'TRASH DETECTED', (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
#         undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs)

#         for (x, y, w, h) in detected_objects:
#             center_x = x + w / 2
#             center_y = y + h / 2

#             # Transform pixel coordinates to robot coordinates
#             robot_coords_mm_x = (center_x - robot_origin_x) * pixel_to_mm_ratio * -1
#             robot_coords_mm_y = (center_y - robot_origin_y) * pixel_to_mm_ratio

#             print(f"Detected object coordinates in robot frame: ({robot_coords_mm_x:.2f} mm, {robot_coords_mm_y:.2f} mm)")

#             # Display the robot coordinates on the frame
#             cv2.putText(frame, f'Robot Coords: ({robot_coords_mm_x:.2f}, {robot_coords_mm_y:.2f})', (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#             # Add robot control code here

#     cv2.imshow('Frame', frame)
#     if key == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

# # =================================================================================================================================================


# import cv2
# import numpy as np
# import time
# import torch
# import cv2.aruco as aruco
# import mediapipe as mp
# from ultralytics import YOLO

# def detect_star_shape(image, canny_thresh1, canny_thresh2):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     edged = cv2.Canny(blurred, canny_thresh1, canny_thresh2)
#     contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     if len(contours) == 1 or len(contours) == 2:
#         contour = contours[0]
#         epsilon = 0.02 * cv2.arcLength(contour, True)
#         approx = cv2.approxPolyDP(contour, epsilon, True)
#         if len(approx) >= 10:
#             return True, approx, edged, len(contours)
#     return False, None, edged, len(contours)

# def nothing(x):
#     pass

# def draw_rounded_rectangle(image, top_left, width, height, radius, color, thickness):
#     x, y = top_left
#     top_right = (x + width, y)
#     bottom_right = (x + width, y + height)
#     bottom_left = (x, y + height)
#     cv2.line(image, (x + radius, y), (x + width - radius, y), color, thickness)
#     cv2.line(image, (x, y + radius), (x, y + height), color, thickness)
#     cv2.line(image, (x + width, y + radius), (x + width, y + height), color, thickness)
#     cv2.line(image, (x, y + height), (x + width, y + height), color, thickness)
#     cv2.ellipse(image, (x + radius, y + radius), (radius, radius), 180, 0, 90, color, thickness)
#     cv2.ellipse(image, (x + width - radius, y + radius), (radius, radius), 270, 0, 90, color, thickness)


# #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# #코드 병합
# # Camera parameters for 3D to 2D conversion
# camera_matrix = np.array([[474.51901407, 0, 302.47811758],
#                           [0, 474.18970657, 250.66191453],
#                           [0, 0, 1]])
# dist_coeffs = np.array([[-0.06544764, -0.07654065, -0.00761827, -0.00279316, 0.08062307]])
# pixel_to_mm_ratio = 1.55145
# robot_origin_x = 295
# robot_origin_y = 184

# def convert_to_robot_coordinates(bbox, label):
#     x1, y1, x2, y2 = map(int, bbox)
#     center_x = (x1 + x2) / 2
#     center_y = (y1 + y2) / 2
#     robot_coords_mm_x = (center_x - robot_origin_x) * pixel_to_mm_ratio * -1
#     robot_coords_mm_y = (center_y - robot_origin_y) * pixel_to_mm_ratio
#     print(f"{label} detected at robot coordinates: ({robot_coords_mm_x:.2f} mm, {robot_coords_mm_y:.2f} mm)")


# #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# cap = cv2.VideoCapture(0)
# model = YOLO('/home/jinjuuk/dev_ws/pt_files/segmentation_s_batch16_freeze8.pt')
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# if torch.cuda.is_available():
#     model.to(device)
# print(f'Using device: {device}')

# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
# aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
# parameters = aruco.DetectorParameters()
# last_seen = {}

# cv2.namedWindow('Frame')
# cv2.createTrackbar('Threshold', 'Frame', 0, 20000, nothing)
# cv2.createTrackbar('Canny Thresh1', 'Frame', 50, 255, nothing)
# cv2.createTrackbar('Canny Thresh2', 'Frame', 112, 255, nothing)
# cv2.createTrackbar('Brightness', 'Frame', 50, 100, nothing)
# cv2.createTrackbar('Confidence', 'Frame', 50, 100, nothing)
# cv2.createTrackbar('Diff_Thresh', 'Frame', 45, 255, nothing)
# cv2.createTrackbar('Hue Min', 'Frame', 0, 179, nothing)
# cv2.createTrackbar('Hue Max', 'Frame', 179, 179, nothing)
# cv2.createTrackbar('Sat Min', 'Frame', 0, 255, nothing)
# cv2.createTrackbar('Sat Max', 'Frame', 255, 255, nothing)
# cv2.createTrackbar('Val Min', 'Frame', 0, 255, nothing)
# cv2.createTrackbar('Val Max', 'Frame', 255, 255, nothing)

# roi_x_large = 55
# roi_y_large = 0
# roi_width_large = 485
# roi_height_large = 315
# roi_x_medium = 270
# roi_y_medium = 0
# roi_width_medium = 270
# roi_height_medium = 60
# roi_x_small = 464
# roi_y_small = 118
# roi_width_small = 35
# roi_height_small = 35

# fgbg = cv2.createBackgroundSubtractorMOG2()
# initial_gray = None
# post_cleanup_gray = None
# detection_enabled = False
# cleanup_detection_enabled = False
# yolo_detection_enabled = False

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to capture frame from camera.")
#         break

#     threshold = cv2.getTrackbarPos('Threshold', 'Frame')
#     canny_thresh1 = cv2.getTrackbarPos('Canny Thresh1', 'Frame')
#     canny_thresh2 = cv2.getTrackbarPos('Canny Thresh2', 'Frame')
#     brightness = cv2.getTrackbarPos('Brightness', 'Frame')
#     confidence_threshold = cv2.getTrackbarPos('Confidence', 'Frame') / 100.0

#     diff_thresh = cv2.getTrackbarPos('Diff_Thresh', 'Frame')
#     h_min = cv2.getTrackbarPos('Hue Min', 'Frame')
#     h_max = cv2.getTrackbarPos('Hue Max', 'Frame')
#     s_min = cv2.getTrackbarPos('Sat Min', 'Frame')
#     s_max = cv2.getTrackbarPos('Sat Max', 'Frame')
#     v_min = cv2.getTrackbarPos('Val Min', 'Frame')
#     v_max = cv2.getTrackbarPos('Val Max', 'Frame')

#     frame = cv2.convertScaleAbs(frame, alpha=1, beta=(brightness - 50) * 2)
#     fgmask = fgbg.apply(frame)

#     roi_large = fgmask[roi_y_large:roi_y_large + roi_height_large, roi_x_large:roi_x_large + roi_width_large]
#     intrusion_detected = np.sum(roi_large) > threshold

#     warning_detected = False
#     trash_detected = False

#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     result = hands.process(frame_rgb)
#     if result.multi_hand_landmarks:
#         for hand_landmarks in result.multi_hand_landmarks:
#             for landmark in hand_landmarks.landmark:
#                 x = int(landmark.x * frame.shape[1])
#                 y = int(landmark.y * frame.shape[0])
#                 if roi_x_large < x < roi_x_large + roi_width_large and roi_y_large < y < roi_y_large + roi_height_large and intrusion_detected:
#                     warning_detected = True
#                     break
#             mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#     roi_medium = frame[roi_y_medium:roi_y_medium + roi_height_medium, roi_x_medium:roi_x_medium + roi_width_medium]
#     corners, ids, rejectedImgPoints = aruco.detectMarkers(roi_medium, aruco_dict, parameters=parameters)
#     current_time = time.time()

#     if ids is not None and len(ids) > 0:
#         aruco.drawDetectedMarkers(roi_medium, corners, ids)
#         for id in ids.flatten():
#             last_seen[id] = current_time
#         for id, last_time in list(last_seen.items()):
#             if current_time - last_time > 3:
#                 cv2.putText(frame, f"Action executed for marker ID {id}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#                 del last_seen[id]
#                 break

#     small_roi = frame[roi_y_small:roi_y_small + roi_height_small, roi_x_small:roi_x_small + roi_width_small]
#     star_detected, star_contour, edged, contour_count = detect_star_shape(small_roi, canny_thresh1, canny_thresh2)
#     if star_detected:
#         star_contour += [roi_x_small, roi_y_small]
#         cv2.drawContours(frame, [star_contour], -1, (0, 255, 0), 3)
#         cv2.putText(frame, "Star detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
#     else:
#         cv2.putText(frame, "Star not detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

#     frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     frame_gray = cv2.GaussianBlur(frame_gray, (21, 21), 0)
#     hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('s'):
#         initial_gray = frame_gray
#         detection_enabled = True
#         yolo_detection_enabled = True
#         print("Initial background set, detection enabled.")

#     if detection_enabled and initial_gray is not None and not cleanup_detection_enabled:
#         frame_delta = cv2.absdiff(initial_gray, frame_gray)
#         _, diff_mask = cv2.threshold(frame_delta, diff_thresh, 255, cv2.THRESH_BINARY)
#         diff_mask = cv2.dilate(diff_mask, None, iterations=2)
#         edges = cv2.Canny(frame_gray, canny_thresh1, canny_thresh2)
#         edges = cv2.dilate(edges, None, iterations=1)
#         edges = cv2.bitwise_and(edges, edges, mask=diff_mask)
#         lower_bound = np.array([h_min, s_min, v_min])
#         upper_bound = np.array([h_max, s_max, v_max])
#         hsv_mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
#         detection_mask = np.zeros(diff_mask.shape, dtype=np.uint8)
#         detection_mask[roi_y_large:roi_y_large + roi_height_large, roi_x_large:roi_x_large + roi_width_large] = 255
#         detection_mask[roi_y_medium:roi_y_medium + roi_height_medium, roi_x_medium:roi_x_medium + roi_width_medium] = 0
#         detection_mask[roi_y_small:roi_y_small + roi_height_small, roi_x_small:roi_x_small + roi_width_small] = 0
#         combined_mask = cv2.bitwise_or(diff_mask, edges)
#         combined_mask = cv2.bitwise_and(combined_mask, combined_mask, mask=detection_mask)
#         combined_mask = cv2.bitwise_and(combined_mask, hsv_mask)
#         contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         detected_objects = []

#         for cnt in contours:
#             if cv2.contourArea(cnt) > 80:
#                 x, y, w, h = cv2.boundingRect(cnt)
#                 if (roi_x_large <= x <= roi_x_large + roi_width_large and
#                     roi_y_large <= y <= roi_y_large + roi_height_large):
#                     detected_objects.append((x, y, w, h))
#                     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                     trash_detected = True

# #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# #객체인식 된것만 로봇좌표로 변환
#     if yolo_detection_enabled:
#         yolo_roi = frame[roi_y_large:roi_y_large + roi_height_large, roi_x_large:roi_x_large + roi_width_large]
#         results = model(yolo_roi)
#         for result in results:
#             boxes = result.boxes
#             for box in boxes:
#                 cls_id = int(box.cls)
#                 if cls_id == 0:
#                     continue
#                 label = model.names[cls_id]
#                 confidence = box.conf.item()
#                 if confidence >= confidence_threshold:
#                     bbox = box.xyxy[0].tolist()
#                     x1, y1, x2, y2 = map(int, bbox)
#                     x1 += roi_x_large
#                     y1 += roi_y_large
#                     x2 += roi_x_large
#                     y2 += roi_y_large
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                     if label == 'cup' or label == 'star':
#                         text = f'{label}'
#                         text_x = x1
#                         text_y = y1 - 10 if y1 - 10 > 10 else y1 + 10
#                         cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#                     trash_detected = True
#                     convert_to_robot_coordinates(bbox, label)

#     draw_rounded_rectangle(frame, (roi_x_large, roi_y_large), roi_width_large, roi_height_large, 20, (255, 0, 0), 2)
#     cv2.rectangle(frame, (roi_x_medium, roi_y_medium), (roi_x_medium + roi_width_medium, roi_y_medium + roi_height_medium), (0, 255, 0), 2)
#     cv2.rectangle(frame, (roi_x_small, roi_y_small), (roi_x_small + roi_width_small, roi_y_small + roi_height_small), (0, 0, 255), 2)
#     cv2.putText(frame, f"Contours: {contour_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
#     frame[roi_y_small:roi_y_small + roi_height_small, roi_x_small:roi_x_small + roi_width_small] = cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR)

#     if warning_detected:
#         cv2.putText(frame, 'WARNING DETECTED!', (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

#     if trash_detected:
#         cv2.putText(frame, 'TRASH DETECTED', (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

#     cv2.imshow('Frame', frame)
#     if key == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

# #==================================================================================================================================================


import cv2
import numpy as np
import time
import torch
import cv2.aruco as aruco
import mediapipe as mp
from ultralytics import YOLO

def detect_star_shape(image, canny_thresh1, canny_thresh2):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, canny_thresh1, canny_thresh2)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 1 or len(contours) == 2:
        contour = contours[0]
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) >= 10:
            return True, approx, edged, len(contours)
    return False, None, edged, len(contours)

def nothing(x):
    pass

def draw_rounded_rectangle(image, top_left, width, height, radius, color, thickness):
    x, y = top_left
    top_right = (x + width, y)
    bottom_right = (x + width, y + height)
    bottom_left = (x, y + height)
    cv2.line(image, (x + radius, y), (x + width - radius, y), color, thickness)
    cv2.line(image, (x, y + radius), (x, y + height), color, thickness)
    cv2.line(image, (x + width, y + radius), (x + width, y + height), color, thickness)
    cv2.line(image, (x, y + height), (x + width, y + height), color, thickness)
    cv2.ellipse(image, (x + radius, y + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(image, (x + width - radius, y + radius), (radius, radius), 270, 0, 90, color, thickness)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Camera parameters for 3D to 2D conversion
camera_matrix = np.array([[474.51901407, 0, 302.47811758],
                          [0, 474.18970657, 250.66191453],
                          [0, 0, 1]])
dist_coeffs = np.array([[-0.06544764, -0.07654065, -0.00761827, -0.00279316, 0.08062307]])
pixel_to_mm_ratio = 1.55145
robot_origin_x = 295
robot_origin_y = 184

def convert_to_robot_coordinates(bbox, label):
    x1, y1, x2, y2 = map(int, bbox)
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    robot_coords_mm_x = (center_x - robot_origin_x) * pixel_to_mm_ratio * -1
    robot_coords_mm_y = (center_y - robot_origin_y) * pixel_to_mm_ratio
    print(f"{label} detected at robot coordinates: ({robot_coords_mm_x:.2f} mm, {robot_coords_mm_y:.2f} mm)")

    # 로봇 제어 코드 추가
    send_to_robot(robot_coords_mm_x, robot_coords_mm_y)

def send_to_robot(x, y):
    # 여기에 로봇 제어 코드를 추가하세요
    # 예를 들어, 로봇 API를 사용하여 특정 좌표로 이동하도록 할 수 있습니다.
    print(f"Sending robot to coordinates: x={x:.2f} mm, y={y:.2f} mm")
    # 로봇 제어 API 호출 예제 (이 부분을 실제 로봇 제어 코드로 대체하세요)
    # robot.move_to(x, y)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

cap = cv2.VideoCapture(0)
model = YOLO('/home/jinjuuk/dev_ws/pt_files/segmentation_s_batch16_freeze8.pt')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if torch.cuda.is_available():
    model.to(device)
print(f'Using device: {device}')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()
last_seen = {}

cv2.namedWindow('Frame')
cv2.createTrackbar('Threshold', 'Frame', 0, 20000, nothing)
cv2.createTrackbar('Canny Thresh1', 'Frame', 50, 255, nothing)
cv2.createTrackbar('Canny Thresh2', 'Frame', 112, 255, nothing)
cv2.createTrackbar('Brightness', 'Frame', 50, 100, nothing)
cv2.createTrackbar('Confidence', 'Frame', 50, 100, nothing)
cv2.createTrackbar('Diff_Thresh', 'Frame', 45, 255, nothing)
cv2.createTrackbar('Hue Min', 'Frame', 0, 179, nothing)
cv2.createTrackbar('Hue Max', 'Frame', 179, 179, nothing)
cv2.createTrackbar('Sat Min', 'Frame', 0, 255, nothing)
cv2.createTrackbar('Sat Max', 'Frame', 255, 255, nothing)
cv2.createTrackbar('Val Min', 'Frame', 0, 255, nothing)
cv2.createTrackbar('Val Max', 'Frame', 255, 255, nothing)

roi_x_large = 55
roi_y_large = 0
roi_width_large = 485
roi_height_large = 315
roi_x_medium = 270
roi_y_medium = 0
roi_width_medium = 270
roi_height_medium = 60
roi_x_small = 464
roi_y_small = 118
roi_width_small = 35
roi_height_small = 35

fgbg = cv2.createBackgroundSubtractorMOG2()
initial_gray = None
post_cleanup_gray = None
detection_enabled = False
cleanup_detection_enabled = False
yolo_detection_enabled = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from camera.")
        break

    threshold = cv2.getTrackbarPos('Threshold', 'Frame')
    canny_thresh1 = cv2.getTrackbarPos('Canny Thresh1', 'Frame')
    canny_thresh2 = cv2.getTrackbarPos('Canny Thresh2', 'Frame')
    brightness = cv2.getTrackbarPos('Brightness', 'Frame')
    confidence_threshold = cv2.getTrackbarPos('Confidence', 'Frame') / 100.0

    diff_thresh = cv2.getTrackbarPos('Diff_Thresh', 'Frame')
    h_min = cv2.getTrackbarPos('Hue Min', 'Frame')
    h_max = cv2.getTrackbarPos('Hue Max', 'Frame')
    s_min = cv2.getTrackbarPos('Sat Min', 'Frame')
    s_max = cv2.getTrackbarPos('Sat Max', 'Frame')
    v_min = cv2.getTrackbarPos('Val Min', 'Frame')
    v_max = cv2.getTrackbarPos('Val Max', 'Frame')

    frame = cv2.convertScaleAbs(frame, alpha=1, beta=(brightness - 50) * 2)
    fgmask = fgbg.apply(frame)

    roi_large = fgmask[roi_y_large:roi_y_large + roi_height_large, roi_x_large:roi_x_large + roi_width_large]
    intrusion_detected = np.sum(roi_large) > threshold

    warning_detected = False
    trash_detected = False

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                if roi_x_large < x < roi_x_large + roi_width_large and roi_y_large < y < roi_y_large + roi_height_large and intrusion_detected:
                    warning_detected = True
                    break
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    roi_medium = frame[roi_y_medium:roi_y_medium + roi_height_medium, roi_x_medium:roi_x_medium + roi_width_medium]
    corners, ids, rejectedImgPoints = aruco.detectMarkers(roi_medium, aruco_dict, parameters=parameters)
    current_time = time.time()

    if ids is not None and len(ids) > 0:
        aruco.drawDetectedMarkers(roi_medium, corners, ids)
        for id in ids.flatten():
            last_seen[id] = current_time
        for id, last_time in list(last_seen.items()):
            if current_time - last_time > 3:
                cv2.putText(frame, f"Action executed for marker ID {id}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                del last_seen[id]
                break

    small_roi = frame[roi_y_small:roi_y_small + roi_height_small, roi_x_small:roi_x_small + roi_width_small]
    star_detected, star_contour, edged, contour_count = detect_star_shape(small_roi, canny_thresh1, canny_thresh2)
    if star_detected:
        star_contour += [roi_x_small, roi_y_small]
        cv2.drawContours(frame, [star_contour], -1, (0, 255, 0), 3)
        cv2.putText(frame, "Star detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    else:
        cv2.putText(frame, "Star not detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.GaussianBlur(frame_gray, (21, 21), 0)
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        initial_gray = frame_gray
        detection_enabled = True
        yolo_detection_enabled = True
        print("Initial background set, detection enabled.")

    if detection_enabled and initial_gray is not None and not cleanup_detection_enabled:
        frame_delta = cv2.absdiff(initial_gray, frame_gray)
        _, diff_mask = cv2.threshold(frame_delta, diff_thresh, 255, cv2.THRESH_BINARY)
        diff_mask = cv2.dilate(diff_mask, None, iterations=2)
        edges = cv2.Canny(frame_gray, canny_thresh1, canny_thresh2)
        edges = cv2.dilate(edges, None, iterations=1)
        edges = cv2.bitwise_and(edges, edges, mask=diff_mask)
        lower_bound = np.array([h_min, s_min, v_min])
        upper_bound = np.array([h_max, s_max, v_max])
        hsv_mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
        detection_mask = np.zeros(diff_mask.shape, dtype=np.uint8)
        detection_mask[roi_y_large:roi_y_large + roi_height_large, roi_x_large:roi_x_large + roi_width_large] = 255
        detection_mask[roi_y_medium:roi_y_medium + roi_height_medium, roi_x_medium:roi_x_medium + roi_width_medium] = 0
        detection_mask[roi_y_small:roi_y_small + roi_height_small, roi_x_small:roi_x_small + roi_width_small] = 0
        combined_mask = cv2.bitwise_or(diff_mask, edges)
        combined_mask = cv2.bitwise_and(combined_mask, combined_mask, mask=detection_mask)
        combined_mask = cv2.bitwise_and(combined_mask, hsv_mask)
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected_objects = []

        for cnt in contours:
            if cv2.contourArea(cnt) > 80:
                x, y, w, h = cv2.boundingRect(cnt)
                if (roi_x_large <= x <= roi_x_large + roi_width_large and
                    roi_y_large <= y <= roi_y_large + roi_height_large):
                    detected_objects.append((x, y, w, h))
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    trash_detected = True

    if yolo_detection_enabled:
        yolo_roi = frame[roi_y_large:roi_y_large + roi_height_large, roi_x_large:roi_x_large + roi_width_large]
        results = model(yolo_roi)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls)
                if cls_id == 0:
                    continue
                label = model.names[cls_id]
                confidence = box.conf.item()
                if confidence >= confidence_threshold:
                    bbox = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = map(int, bbox)
                    x1 += roi_x_large
                    y1 += roi_y_large
                    x2 += roi_x_large
                    y2 += roi_y_large
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    if label == 'cup' or label == 'star':
                        text = f'{label}'
                        text_x = x1
                        text_y = y1 - 10 if y1 - 10 > 10 else y1 + 10
                        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    trash_detected = True
                    convert_to_robot_coordinates(bbox, label) # 추가

    draw_rounded_rectangle(frame, (roi_x_large, roi_y_large), roi_width_large, roi_height_large, 20, (255, 0, 0), 2)
    cv2.rectangle(frame, (roi_x_medium, roi_y_medium), (roi_x_medium + roi_width_medium, roi_y_medium + roi_height_medium), (0, 255, 0), 2)
    cv2.rectangle(frame, (roi_x_small, roi_y_small), (roi_x_small + roi_width_small, roi_y_small + roi_height_small), (0, 0, 255), 2)
    cv2.putText(frame, f"Contours: {contour_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    frame[roi_y_small:roi_y_small + roi_height_small, roi_x_small:roi_x_small + roi_width_small] = cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR)

    if warning_detected:
        cv2.putText(frame, 'WARNING DETECTED!', (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    if trash_detected:
        cv2.putText(frame, 'TRASH DETECTED', (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow('Frame', frame)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
