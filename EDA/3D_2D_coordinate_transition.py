# # YOLOv8 모델 로드import cv2
# import numpy as np
# import time
# import torch
# import cv2.aruco as aruco
# import mediapipe as mp
# from ultralytics import YOLO
# import cv2

# def detect_star_shape(image, canny_thresh1, canny_thresh2):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     edged = cv2.Canny(blurred, canny_thresh1, canny_thresh2)
#     contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     if len(contours) == 1 or len(contours) == 2:
#         contour = contours[0]
#         epsilon = 0.02 * cv2.arcLength(contour, True)
#         approx = cv2.approxPolyDP(contour, epsilon, True)

#         # print(f"approx:", len(approx))
#         if len(approx) >= 10:  # Assuming a star has approximately 10 points
#             return True, approx, edged, len(contours)
#     return False, None, edged, len(contours)

# def nothing(x):
#     pass

# def draw_rounded_rectangle(image, top_left, width, height, radius, color, thickness):

#     """
#     Draw a rectangle with rounded top corners.
#     :param image: The image to draw on.
#     :param top_left: Top left corner of the rectangle (x, y).
#     :param width: Width of the rectangle.
#     :param height: Height of the rectangle.
#     :param radius: Radius of the rounded corners.
#     :param color: Color of the rectangle.
#     :param thickness: Thickness of the rectangle border.
#     """
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

# # YOLOv8 model
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

# while True:
#     ret, frame = cap.read()
    
# model = YOLO('/home/jinjuuk/dev_ws/pt_files/new_custom_m.pt')  # 모델 경로를 설정하세요.


# # 모델을 GPU로 이동
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# if torch.cuda.is_available():
#     model.to(device)

# # GPU 사용 여부 출력    
# print(f'Using device: {device}')

# # 캘리브레이션 데이터를 사용하여 카메라 매트릭스와 왜곡 계수를 설정
# camera_matrix = np.array([[474.51901407, 0, 302.47811758],
#                           [0, 474.18970657, 250.66191453],
#                           [0, 0, 1]])
# dist_coeffs = np.array([[-0.06544764, -0.07654065, -0.00761827, -0.00279316, 0.08062307]])

# # 1 픽셀당 1.55145mm 단위 변환 비율
# pixel_to_mm_ratio = 1.55145

# # 로봇 좌표계의 중앙값 (픽셀 단위)
# robot_origin_x = 295
# robot_origin_y = 184

# # 비디오 캡처 초기화 (카메라 ID는 0으로 설정, 필요 시 변경 가능)
# cap = cv2.VideoCapture(0)

# # 트랙바 콜백 함수 (아무 동작도 하지 않음)
# def nothing(x):
#     pass

# # 트랙바를 위한 윈도우 생성
# cv2.namedWindow('Detection Results')
# cv2.createTrackbar('Confidence', 'Detection Results', 50, 100, nothing)  # 기본값 50, 범위 0-100

# # 객체 좌표를 저장할 딕셔너리
# object_coords = {}

# # 유사한 객체를 인식할 때 사용할 거리 임계값
# similarity_threshold = 50

# # 제어 모드 플래그
# control_mode = False

# def find_similar_object(center_x, center_y, label, object_coords, threshold):
#     for obj_id, coords in object_coords.items():
#         if obj_id.startswith(label) and coords:
#             avg_x = sum([coord[0] for coord in coords]) / len(coords)
#             avg_y = sum([coord[1] for coord in coords]) / len(coords)
#             distance = np.sqrt((avg_x - center_x) ** 2 + (avg_y - center_y) ** 2)
#             if distance < threshold:
#                 return obj_id
#     return None

# while True:
#     # 트랙바에서 현재 Confidence 값 가져오기
#     confidence_threshold = cv2.getTrackbarPos('Confidence', 'Detection Results') / 100.0

#     # 프레임 캡처
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # 왜곡 보정
#     undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs)

#     if not control_mode:
#         # 객체 검출 수행
#         results = model(undistorted_frame, conf=confidence_threshold)

#         # 바운딩 박스 그리기
#         for result in results:
#             boxes = result.boxes
#             for box in boxes:
#                 cls_id = int(box.cls)
#                 label = model.names[cls_id]
#                 confidence = box.conf.item()  # 신뢰도 추출

#                 # 'cup' 및 'star' 객체만 추적
#                 if label in ['cup', 'star'] and confidence >= confidence_threshold:
#                     bbox = box.xyxy[0].tolist()  # 바운딩 박스를 리스트로 변환
#                     x1, y1, x2, y2 = map(int, bbox)
#                     center_x = (x1 + x2) / 2
#                     center_y = (y1 + y2) / 2
#                     cv2.rectangle(undistorted_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

#                     # 텍스트 위치 계산
#                     text = f'{label} {confidence:.2f}'
#                     (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

#                     # 텍스트가 바운딩 박스 내부, 오른쪽 상단에 표시되도록 위치 조정
#                     text_x = x2 - text_width if x2 - text_width > 0 else x1
#                     text_y = y1 - 2 if y1 - 2 > text_height else y1 + text_height + 2

#                     # 텍스트 배경 상자 그리기
#                     cv2.rectangle(undistorted_frame, (text_x, text_y - text_height - baseline), (text_x + text_width, text_y + baseline), (0, 255, 0), cv2.FILLED)

#                     # 텍스트 그리기
#                     cv2.putText(undistorted_frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

#                     # 유사한 객체 찾기
#                     similar_object_id = find_similar_object(center_x, center_y, label, object_coords, similarity_threshold)

#                     if similar_object_id:
#                         object_id = similar_object_id
#                     else:
#                         object_id = f'{label}_{len(object_coords)}'
#                         object_coords[object_id] = []
#                         print(f"새 객체 발견: {object_id}")

#                     # 좌표 추가, 20개까지만 저장
#                     if len(object_coords[object_id]) < 20:
#                         object_coords[object_id].append((center_x, center_y))
#                         print(f"{object_id}의 좌표 추가됨: ({center_x}, {center_y})")
#                         print(f"{object_id}의 좌표 개수: {len(object_coords[object_id])}")

#         # 모든 객체가 20개의 좌표를 수집하면 제어 모드로 전환
#         if all(len(coords) >= 20 for coords in object_coords.values()):
#             control_mode = True
#             print("제어 모드로 전환")

#     else:
#         # 모든 객체의 평균 좌표 계산
#         avg_coords = {}
#         for object_id, coords in object_coords.items():
#             avg_x = sum([coord[0] for coord in coords]) / 20
#             avg_y = sum([coord[1] for coord in coords]) / 20
#             avg_coords[object_id] = (avg_x, avg_y)

#         # 카메라 화면의 중심 좌표
#         center_x_cam, center_y_cam = 320, 240

#         # 객체들을 중심으로부터의 거리 순으로 정렬 (가장 먼 객체부터)
#         sorted_objects = sorted(avg_coords.items(), key=lambda item: np.sqrt((item[1][0] - center_x_cam) ** 2 + (item[1][1] - center_y_cam) ** 2), reverse=True)

#         # 모든 객체 ID와 거리 순서대로 번호 표시
#         for idx, (object_id, (avg_x, avg_y)) in enumerate(sorted_objects, start=1):
#             distance = np.sqrt((avg_x - center_x_cam) ** 2 + (avg_y - center_y_cam) ** 2)
#             display_text = f'{idx}. {object_id} ({distance:.2f} px)'
#             cv2.putText(undistorted_frame, display_text, (10, 30 + 30 * idx), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

#         # 가장 먼 객체부터 순차적으로 제어
#         for object_id, (avg_x, avg_y) in sorted_objects:
#             print(f"{object_id}의 평균 좌표 계산됨: ({avg_x:.2f}, {avg_y:.2f})")

#             # 중심으로부터의 거리 계산
#             distance = np.sqrt((avg_x - center_x_cam) ** 2 + (avg_y - center_y_cam) ** 2)
#             print(f"{object_id}의 중심으로부터 거리 계산됨: {distance:.2f} 픽셀")

#             # 픽셀 좌표를 MM 좌표로 변환
#             robot_coords_mm_x = (avg_x - robot_origin_x) * pixel_to_mm_ratio * -1
#             robot_coords_mm_y = (avg_y - robot_origin_y) * pixel_to_mm_ratio
#             print(f"{object_id}의 로봇 좌표 계산됨: ({robot_coords_mm_x:.2f} mm, {robot_coords_mm_y:.2f} mm)")

#             # 로봇 제어 코드 실행
#             # 여기에 로봇 제어 코드를 추가하세요
#             print(f"로봇 제어 코드 실행: {object_id}")

#         # 모든 객체의 제어가 완료되면 객체 리스트 초기화 및 인식 모드로 전환
#         object_coords.clear()
#         control_mode = False
#         print("모든 객체 제어 완료. 인식 모드로 전환.")

#     # 결과가 포함된 이미지 표시
#     cv2.imshow('Detection Results', undistorted_frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()




import os
import sys
import logging

# YOLO 라이브러리 로깅 설정 (최소화)
logging.getLogger('ultralytics').setLevel(logging.WARNING)

import cv2
import numpy as np
from ultralytics import YOLO
import torch

# YOLOv8 모델 로드
model = YOLO('/home/jinjuuk/dev_ws/pt_files/new_custom_m.pt')  # 모델 경로를 설정하세요.

# 모델을 GPU로 이동
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    model.to(device)

# GPU 사용 여부 출력
print(f'Using device: {device}')

# 캘리브레이션 데이터를 사용하여 카메라 매트릭스와 왜곡 계수를 설정
camera_matrix = np.array([[474.51901407, 0, 302.47811758],
                          [0, 474.18970657, 250.66191453],
                          [0, 0, 1]])
dist_coeffs = np.array([[-0.06544764, -0.07654065, -0.00761827, -0.00279316, 0.08062307]])

# 1 픽셀당 1.55145mm 단위 변환 비율
pixel_to_mm_ratio = 1.55145

# 로봇 좌표계의 중앙값 (픽셀 단위)
robot_origin_x = 295
robot_origin_y = 184

# 비디오 캡처 초기화 (카메라 ID는 0으로 설정, 필요 시 변경 가능)
cap = cv2.VideoCapture(0)

# 트랙바 콜백 함수 (아무 동작도 하지 않음)
def nothing(x):
    pass

# 트랙바를 위한 윈도우 생성
cv2.namedWindow('Detection Results')
cv2.createTrackbar('Confidence', 'Detection Results', 50, 100, nothing)  # 기본값 50, 범위 0-100

# 객체별 중심 좌표를 저장할 딕셔너리 초기화
center_points_dict = {}

# 로봇 제어 함수 (여기서는 단순히 좌표를 출력하는 함수로 대체)
def send_to_robot(x, y):
    print(f"Sending to robot: x={x:.2f} mm, y={y:.2f} mm")

# 함수 추가: 바운딩 박스를 기반으로 객체의 고유 ID 생성
def generate_object_id(label, x1, y1, x2, y2):
    return f"{label}_{x1}_{y1}_{x2}_{y2}"

# 표준 출력을 비활성화하는 클래스
class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

def process_detections(results, confidence_threshold, undistorted_frame):
    for result in results:
        for box in result.boxes:
            label = model.names[int(box.cls)]
            confidence = box.conf.item()
            if confidence < confidence_threshold:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            object_id = generate_object_id(label, x1, y1, x2, y2)

            if label not in center_points_dict:
                center_points_dict[label] = {}
            if object_id not in center_points_dict[label]:
                center_points_dict[label][object_id] = []
            center_points_dict[label][object_id].append((center_x, center_y))

            cv2.rectangle(undistorted_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f'{label} {confidence:.2f}'
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            text_x = x2 - text_width if x2 - text_width > 0 else x1
            text_y = y1 - 2 if y1 - 2 > text_height else y1 + text_height + 2
            cv2.rectangle(undistorted_frame, (text_x, text_y - text_height - baseline), (text_x + text_width, text_y + baseline), (0, 255, 0), cv2.FILLED)
            cv2.putText(undistorted_frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

def process_center_points():
    for label in list(center_points_dict.keys()):
        for object_id, points in list(center_points_dict[label].items()):
            if len(points) < 20:
                continue

            avg_center_x = np.mean([pt[0] for pt in points])
            avg_center_y = np.mean([pt[1] for pt in points])
            robot_coords_mm_x = (avg_center_x - robot_origin_x) * pixel_to_mm_ratio * -1
            robot_coords_mm_y = (avg_center_y - robot_origin_y) * pixel_to_mm_ratio

            print(f"Class: {label}, ID: {object_id}, Average Center point in camera coordinates: ({avg_center_x:.2f}, {avg_center_y:.2f}), Average Center point in robot coordinates: ({robot_coords_mm_x:.2f} mm, {robot_coords_mm_y:.2f} mm)")
            # send_to_robot(robot_coords_mm_x, robot_coords_mm_y)
            center_points_dict[label][object_id] = []

while True:
    confidence_threshold = cv2.getTrackbarPos('Confidence', 'Detection Results') / 100.0
    ret, frame = cap.read()
    if not ret:
        break
    undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs)

    with SuppressOutput():
        results = model(undistorted_frame, conf=confidence_threshold)

    process_detections(results, confidence_threshold, undistorted_frame)
    process_center_points()

    cv2.imshow('Detection Results', undistorted_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 모델 종료
del model
cap.release()
cv2.destroyAllWindows()



import os
import sys
import logging

# YOLO 라이브러리 로깅 설정 (최소화)
logging.getLogger('ultralytics').setLevel(logging.WARNING)

