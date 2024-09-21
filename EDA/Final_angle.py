import cv2
import numpy as np
import time
import torch
import cv2.aruco as aruco
import mediapipe as mp
from ultralytics import YOLO
from skimage.metrics import structural_similarity as compare_ssim



def detect_star_shape(image, canny_thresh1, canny_thresh2):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, canny_thresh1, canny_thresh2)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 1 or len(contours) == 2:
        contour = contours[0]
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) >= 10:  # Assuming a star has approximately 10 points
            return True, approx, edged, len(contours)
    return False, None, edged, len(contours)

def nothing(x):
    pass

def draw_rounded_rectangle(image, top_left, width, height, radius, color, thickness):
    x, y = top_left
    top_right = (x + width, y)
    bottom_right = (x + width, y + height)
    bottom_left = (x, y + height)

    # Draw straight lines
    cv2.line(image, (x + radius, y), (x + width - radius, y), color, thickness)
    cv2.line(image, (x, y + radius), (x, y + height), color, thickness)
    cv2.line(image, (x + width, y + radius), (x + width, y + height), color, thickness)
    cv2.line(image, (x, y + height), (x + width, y + height), color, thickness)

    # Draw arcs for rounded corners
    cv2.ellipse(image, (x + radius, y + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(image, (x + width - radius, y + radius), (radius, radius), 270, 0, 90, color, thickness)

def calculate_ssim(imageA, imageB):
    score, diff = compare_ssim(imageA, imageB, full=True)
    return score

cap = cv2.VideoCapture(1)

# YOLOv8 model
model = YOLO('newjeans.pt')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    model.to(device)
print(f'Using device: {device}')

# Mediapipe hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

# ArUco marker detection
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()
last_seen = {}

# Trackbars for thresholds and brightness
cv2.namedWindow('Frame')
cv2.createTrackbar('Threshold', 'Frame', 0, 20000, nothing)
cv2.createTrackbar('Canny Thresh1', 'Frame', 50, 255, nothing)
cv2.createTrackbar('Canny Thresh2', 'Frame', 112, 255, nothing)
cv2.createTrackbar('Brightness', 'Frame', 50, 100, nothing)
cv2.createTrackbar('Confidence', 'Frame', 50, 100, nothing)

# Trackbars for trash detection
cv2.createTrackbar('Diff_Thresh', 'Frame', 45, 255, nothing)
cv2.createTrackbar('Hue Min', 'Frame', 0, 179, nothing)
cv2.createTrackbar('Hue Max', 'Frame', 179, 179, nothing)
cv2.createTrackbar('Sat Min', 'Frame', 0, 255, nothing)
cv2.createTrackbar('Sat Max', 'Frame', 255, 255, nothing)
cv2.createTrackbar('Val Min', 'Frame', 0, 255, nothing)
cv2.createTrackbar('Val Max', 'Frame', 255, 255, nothing)

# ROI definitions
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

# 초기 배경 이미지 변수
initial_gray = None
detection_enabled = False
yolo_detection_enabled = False
cleanup_start_time = None
cleanup_delay = 10  # Delay in seconds before starting cleanup

# 추가된 변수
object_coords = {}
control_mode = False

# 캘리브레이션 데이터
camera_matrix = np.array([[474.51901407, 0, 302.47811758],
                          [0, 474.18970657, 250.66191453],
                          [0, 0, 1]])
dist_coeffs = np.array([[-0.06544764, -0.07654065, -0.00761827, -0.00279316, 0.08062307]])
pixel_to_mm_ratio = 1.55145
robot_origin_x = 295
robot_origin_y = 184



def find_similar_object(center_x, center_y, label, object_coords, threshold):
    for obj_id, coords in object_coords.items():
        if obj_id.startswith(label) and coords:
            avg_x = sum([coord[0] for coord in coords]) / len(coords)
            avg_y = sum([coord[1] for coord in coords]) / len(coords)
            distance = np.sqrt((avg_x - center_x) ** 2 + (avg_y - center_y) ** 2)
            if distance < threshold:
                return obj_id
    return None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from camera.")
        break
    

    # 왜곡 보정
    undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs)


    # Trackbar values
    threshold = cv2.getTrackbarPos('Threshold', 'Frame')
    canny_thresh1 = cv2.getTrackbarPos('Canny Thresh1', 'Frame')
    canny_thresh2 = cv2.getTrackbarPos('Canny Thresh2', 'Frame')
    brightness = cv2.getTrackbarPos('Brightness', 'Frame')
    confidence_threshold = cv2.getTrackbarPos('Confidence', 'Frame') / 100.0
    
    # Trackbar values for trash detection
    diff_thresh = cv2.getTrackbarPos('Diff_Thresh', 'Frame')
    h_min = cv2.getTrackbarPos('Hue Min', 'Frame')
    h_max = cv2.getTrackbarPos('Hue Max', 'Frame')
    s_min = cv2.getTrackbarPos('Sat Min', 'Frame')
    s_max = cv2.getTrackbarPos('Sat Max', 'Frame')
    v_min = cv2.getTrackbarPos('Val Min', 'Frame')
    v_max = cv2.getTrackbarPos('Val Max', 'Frame')
    
    undistorted_frame = cv2.convertScaleAbs(undistorted_frame, alpha=1, beta=(brightness - 50) * 2)
    
    fgmask = fgbg.apply(undistorted_frame)
    
    # Intrusion detection
    roi_large = fgmask[roi_y_large:roi_y_large + roi_height_large, roi_x_large:roi_x_large + roi_width_large]
    intrusion_detected = np.sum(roi_large) > threshold
    
    warning_detected = False
    trash_detected = False

    # Mediapipe hand detection
    frame_rgb = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                x = int(landmark.x * undistorted_frame.shape[1])
                y = int(landmark.y * undistorted_frame.shape[0])
                if roi_x_large < x < roi_x_large + roi_width_large and roi_y_large < y < roi_y_large + roi_height_large and intrusion_detected:
                    warning_detected = True
                    break
            mp.solutions.drawing_utils.draw_landmarks(undistorted_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    # ArUco marker detection
    roi_medium = undistorted_frame[roi_y_medium:roi_y_medium + roi_height_medium, roi_x_medium:roi_x_medium + roi_width_medium]
    corners, ids, rejectedImgPoints = aruco.detectMarkers(roi_medium, aruco_dict, parameters=parameters)
    current_time = time.time()
    if ids is not None and len(ids) > 0:
        aruco.drawDetectedMarkers(roi_medium, corners, ids)
        for id in ids.flatten():
            last_seen[id] = current_time
        for id, last_time in list(last_seen.items()):
            if current_time - last_time > 3:
                cv2.putText(undistorted_frame, f"Action executed for marker ID {id}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                del last_seen[id]
                break
    
    # Star shape detection
    small_roi = undistorted_frame[roi_y_small:roi_y_small + roi_height_small, roi_x_small:roi_x_small + roi_width_small]
    star_detected, star_contour, edged, contour_count = detect_star_shape(small_roi, canny_thresh1, canny_thresh2)
    if star_detected:
        star_contour += [roi_x_small, roi_y_small]
        cv2.drawContours(undistorted_frame, [star_contour], -1, (0, 255, 0), 3)
        cv2.putText(undistorted_frame, "Star detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    else:
        cv2.putText(undistorted_frame, "Star not detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    # Trash detection
    frame_gray = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.GaussianBlur(frame_gray, (21, 21), 0)
    hsv_frame = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2HSV)
    
    # 초기값 설정 또는 해제
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        initial_gray = frame_gray
        detection_enabled = True
        yolo_detection_enabled = True
        cleanup_start_time = time.time()
        print("Initial background set, detection enabled.")
        
    # 일정 시간이 지난 후에만 검출 활성화
    if detection_enabled and initial_gray is not None:
        # 초기 이미지와 현재 이미지의 차이 계산
        elapsed_time = time.time() - cleanup_start_time
        if elapsed_time > cleanup_delay:
            frame_delta = cv2.absdiff(initial_gray, frame_gray)
            
            # 차이 이미지의 임계값 적용
            _, diff_mask = cv2.threshold(frame_delta, diff_thresh, 255, cv2.THRESH_BINARY)
            diff_mask = cv2.dilate(diff_mask, None, iterations=2)
            
            # Canny 엣지 검출
            edges = cv2.Canny(frame_gray, canny_thresh1, canny_thresh2)
            edges = cv2.dilate(edges, None, iterations=1)
            edges = cv2.bitwise_and(edges, edges, mask=diff_mask)
            
            # HSV 색상 범위에 따른 마스크 생성
            lower_bound = np.array([h_min, s_min, v_min])
            upper_bound = np.array([h_max, s_max, v_max])
            hsv_mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
            
            # 탐지 영역 마스크 생성 (마커와 별 인식 영역 제외)
            detection_mask = np.zeros(diff_mask.shape, dtype=np.uint8)
            detection_mask[roi_y_large:roi_y_large + roi_height_large, roi_x_large:roi_x_large + roi_width_large] = 255
            detection_mask[roi_y_medium:roi_y_medium + roi_height_medium, roi_x_medium:roi_x_medium + roi_width_medium] = 0
            detection_mask[roi_y_small:roi_y_small + roi_height_small, roi_x_small:roi_x_small + roi_width_small] = 0
            
            # 최종 마스크 적용
            combined_mask = cv2.bitwise_or(diff_mask, edges)
            combined_mask = cv2.bitwise_and(combined_mask, combined_mask, mask=detection_mask)
            combined_mask = cv2.bitwise_and(combined_mask, hsv_mask)
            
            # 윤곽선 검출
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
          
            
            for cnt in contours:
                if cv2.contourArea(cnt) > 80:
                    x, y, w, h = cv2.boundingRect(cnt)
                    if (roi_x_large <= x <= roi_x_large + roi_width_large and
                        roi_y_large <= y <= roi_y_large + roi_height_large):
                        cv2.rectangle(undistorted_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        trash_detected = True
    
    
    draw_rounded_rectangle(undistorted_frame, (roi_x_large, roi_y_large), roi_width_large, roi_height_large, 20, (255, 0, 0), 2)
    cv2.rectangle(undistorted_frame, (roi_x_medium, roi_y_medium), (roi_x_medium + roi_width_medium, roi_y_medium + roi_height_medium), (0, 255, 0), 2)
    cv2.rectangle(undistorted_frame, (roi_x_small, roi_y_small), (roi_x_small + roi_width_small, roi_y_small + roi_height_small), (0, 0, 255), 2)
    cv2.putText(undistorted_frame, f"Contours: {contour_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    
    undistorted_frame[roi_y_small:roi_y_small + roi_height_small, roi_x_small:roi_x_small + roi_width_small] = cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR)
    
    if warning_detected:
        cv2.putText(undistorted_frame, 'WARNING DETECTED!', (10, undistorted_frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    if detection_enabled and initial_gray is not None and (time.time() - cleanup_start_time > cleanup_delay):
        ssim_score = calculate_ssim(initial_gray, frame_gray)
        print(f"SSIM score: {ssim_score}")
        if ssim_score > 0.95 :
            
            if trash_detected:
                cv2.putText(undistorted_frame, 'TRASH DETECTED', (10, undistorted_frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)          
            
            if not control_mode:
                # # 왜곡 보정
                # undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
                #YOLO 탐지
                yolo_detected_objects = {}
                if yolo_detection_enabled:
                    yolo_roi = undistorted_frame[roi_y_large:roi_y_large + roi_height_large, roi_x_large:roi_x_large + roi_width_large]
                    results = model(yolo_roi)
                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            cls_id = int(box.cls)
                            label = model.names[cls_id]
                            confidence = box.conf.item()
                            if confidence >= confidence_threshold:
                                bbox = box.xyxy[0].tolist()
                                x1, y1, x2, y2 = map(int, bbox)
                                x1 += roi_x_large
                                y1 += roi_y_large
                                x2 += roi_x_large
                                y2 += roi_y_large
                                cv2.rectangle(undistorted_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                
                                # Draw the label and confidence
                                text = f'{label} {confidence:.2f}'
                                (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                                cv2.rectangle(undistorted_frame, (x1, y1 - text_height - baseline), (x1 + text_width, y1), (0, 255, 0), -1)
                                cv2.putText(undistorted_frame, text, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                                
                                trash_detected = True
                                
                                if trash_detected:
                                    cv2.putText(undistorted_frame, 'TRASH DETECTED', (10, undistorted_frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                                
                                obj_id = f'{label}_{len(yolo_detected_objects)}'
                                yolo_detected_objects[obj_id] = (x1, y1, x2 - x1, y2 - y1)
                                
                                if label in ['side_cup', 'side_star']:
                                    center_x = (x1 + x2) / 2
                                    center_y = (y1 + y2) / 2
                                    similar_object_id = find_similar_object(center_x, center_y, label, object_coords, 50)
                                    if similar_object_id:
                                        object_id = similar_object_id
                                    else:
                                        object_id = f'{label}_{len(object_coords)}'
                                        object_coords[object_id] = []
                                        print(f"새 객체 발견: {object_id}")

                                    # 좌표 추가, 20개까지만 저장
                                    if len(object_coords[object_id]) < 20:
                                        object_coords[object_id].append((center_x, center_y))
                                        print(f"{object_id}의 좌표 추가됨: ({center_x}, {center_y})")
                                        print(f"{object_id}의 좌표 개수: {len(object_coords[object_id])}")

                 #모든 객체가 20개의 좌표를 수집하면 제어 모드로 전환
                if all(len(coords) >= 20 for coords in object_coords.values()):
                    control_mode = True
                    print("제어 모드로 전환")
            
            
            else:
            # 객체 좌표 계산 후 집는 동작
                print(f"object_coords:",object_coords)
                avg_coords = {}
                for object_id, coords in object_coords.items():
                    avg_x = sum([coord[0] for coord in coords]) / 20
                    avg_y = sum([coord[1] for coord in coords]) / 20
                    avg_coords[object_id] = (avg_x, avg_y)
                    print(f"avg_coords:",avg_coords)

                # 카메라 화면의 중심 좌표
                center_x_cam, center_y_cam = 320, 240

                # 객체들을 중심으로부터의 거리 순으로 정렬 (가장 먼 객체부터)
                sorted_objects = sorted(avg_coords.items(), key=lambda item: np.sqrt((item[1][0] - center_x_cam) ** 2 + (item[1][1] - center_x_cam) ** 2), reverse=True)
                print(f"sorted_objects:",sorted_objects) 
                for rank, (object_id, (avg_x, avg_y)) in enumerate(sorted_objects,1):
                    label = '_'.join(object_id.split('_')[:-1])
                    print(f"label:",label)
                    if label in ['side_cup', 'side_star']:
                        # 중심으로부터의 거리 계산
                        distance = np.sqrt((avg_x - center_x_cam) ** 2 + (avg_y - center_y_cam) ** 2)
                        print(f"{object_id}의 중심으로부터 거리 계산됨: {distance:.2f} 픽셀")

                        # 픽셀 좌표를 MM 좌표로 변환
                        robot_coords_mm_x = (avg_x - robot_origin_x) * pixel_to_mm_ratio * -1
                        robot_coords_mm_y = (avg_y - robot_origin_y) * pixel_to_mm_ratio
                        print(f"{object_id}의 로봇 좌표 계산됨: ({robot_coords_mm_x:.2f} mm, {robot_coords_mm_y:.2f} mm)")

                        # ## 집어버리는 동작
                        print("집는동작")
                         # 박스 위에 순위 표시
                        x1, y1, x2, y2 = yolo_detected_objects[object_id]
                        cv2.putText(undistorted_frame, str(rank), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (125, 125, 125), 2)
                     
                # 나머지 객체 쓸어버리는 동작
                print("쓸어버리는 동작 실시")

    cv2.imshow('frame', undistorted_frame)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
