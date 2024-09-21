import cv2
import numpy as np
from ultralytics import YOLO
import torch

# YOLOv8 모델 로드
model = YOLO('/home/jinjuuk/dev_ws/pt_files/new_custom_m_freeze10.pt')  # 모델 경로를 설정하세요.

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

# 객체 중심 좌표를 저장할 리스트
center_points = []

# 로봇 제어 함수 (여기서는 단순히 좌표를 출력하는 함수로 대체)
def send_to_robot(x, y):
    print(f"Sending to robot: x={x:.2f} mm, y={y:.2f} mm")

while True:
    # 트랙바에서 현재 Confidence 값 가져오기
    confidence_threshold = cv2.getTrackbarPos('Confidence', 'Detection Results') / 100.0

    # 프레임 캡처
    ret, frame = cap.read()
    if not ret:
        break

    # 왜곡 보정
    undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs)

    # 객체 검출 수행
    results = model(undistorted_frame, conf=confidence_threshold)

    # 바운딩 박스 그리기
    for result in results:
        boxes = result.boxes

        for box in boxes:
            cls_id = int(box.cls)
            label = model.names[cls_id]
            confidence = box.conf.item()  # 신뢰도 추출

            # 신뢰도가 트랙바에서 설정한 값보다 높은 객체만 표시
            if confidence >= confidence_threshold:
                bbox = box.xyxy[0].tolist()  # 바운딩 박스를 리스트로 변환
                x1, y1, x2, y2 = map(int, bbox)
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                # 중심 좌표 저장
                center_points.append((center_x, center_y, label, confidence))

                # 20개의 좌표를 얻으면 평균을 계산
                if len(center_points) == 20:
                    # 중심 좌표와 거리 계산
                    distances = []
                    for (cx, cy, label, conf) in center_points:
                        distance = np.sqrt((cx - robot_origin_x)**2 + (cy - robot_origin_y)**2)
                        distances.append((distance, cx, cy, label, conf))

                    # 거리 순으로 정렬
                    distances.sort(reverse=True, key=lambda x: x[0])

                    for (distance, avg_center_x, avg_center_y, label, confidence) in distances:
                        # 평균 중심 좌표를 로봇 좌표로 변환
                        robot_coords_mm_x = (avg_center_x - robot_origin_x) * pixel_to_mm_ratio * -1
                        robot_coords_mm_y = (avg_center_y - robot_origin_y) * pixel_to_mm_ratio

                        # 필요한 정보 출력
                        print(f"Class: {label}, Confidence: {confidence:.2f}, Distance: {distance:.2f} pixels, "
                              f"Average Center point in camera coordinates: ({avg_center_x:.2f}, {avg_center_y:.2f}), "
                              f"Average Center point in robot coordinates: ({robot_coords_mm_x:.2f} mm, {robot_coords_mm_y:.2f} mm)")

                        # 로봇 제어 함수 호출
                        send_to_robot(robot_coords_mm_x, robot_coords_mm_y)

                    # 리스트 초기화
                    center_points = []

                # 바운딩 박스 및 텍스트 그리기
                cv2.rectangle(undistorted_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 텍스트 위치 계산
                text = f'{label} {confidence:.2f}'
                (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

                # 텍스트가 바운딩 박스 내부, 오른쪽 상단에 표시되도록 위치 조정
                text_x = x2 - text_width if x2 - text_width > 0 else x1
                text_y = y1 - 2 if y1 - 2 > text_height else y1 + text_height + 2

                # 텍스트 배경 상자 그리기
                cv2.rectangle(undistorted_frame, (text_x, text_y - text_height - baseline), (text_x + text_width, text_y + baseline), (0, 255, 0), cv2.FILLED)

                # 텍스트 그리기
                cv2.putText(undistorted_frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # 결과가 포함된 이미지 표시
    cv2.imshow('Detection Results', undistorted_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 모델 종료
del model

cap.release()
cv2.destroyAllWindows()
