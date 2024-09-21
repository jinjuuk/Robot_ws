# import cv2
# import numpy as np
# import torch
# from ultralytics import YOLO

# # 세그멘테이션 모델 로드
# model = YOLO('/home/jinjuuk/dev_ws/pt_files/newjeans.pt')  # 세그멘테이션 모델 경로를 설정하세요.

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

# # 비디오 캡처 초기화 (카메라 ID는 0으로 설정, 필요 시 변경 가능)
# cap = cv2.VideoCapture(0)

# # 트랙바 콜백 함수 (아무 동작도 하지 않음)
# def nothing(x):
#     pass

# # 트랙바를 위한 윈도우 생성
# cv2.namedWindow('Segmentation Results')
# cv2.createTrackbar('Confidence', 'Segmentation Results', 50, 100, nothing)  # 기본값 50, 범위 0-100

# def calculate_angle(pt1, pt2):
#     angle = np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0]) * 180.0 / np.pi
#     return angle

# while True:
#     # 트랙바에서 현재 Confidence 값 가져오기
#     confidence_threshold = cv2.getTrackbarPos('Confidence', 'Segmentation Results') / 100.0

#     # 프레임 캡처
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # 왜곡 보정
#     undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs)

#     # 객체 검출 수행
#     results = model(undistorted_frame, conf=confidence_threshold)
    
#     for result in results:
#         masks = result.masks  # 세그멘테이션 마스크 가져오기
#         for mask in masks.data.cpu().numpy():  # CPU 메모리로 복사한 후 numpy 배열로 변환
#             mask = mask.astype(np.uint8)  # numpy 배열을 uint8로 변환
#             contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
#             # 디버깅: 세그멘테이션 마스크 표시
#             debug_mask = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)
#             cv2.imshow('Debug Mask', debug_mask)

#             if contours:
#                 cnt = max(contours, key=cv2.contourArea)
#                 rect = cv2.minAreaRect(cnt)
#                 box = cv2.boxPoints(rect)
#                 box = np.int0(box)
#                 cv2.drawContours(undistorted_frame, [box], 0, (0, 255, 0), 2)

#                 # 중심점 계산
#                 M = cv2.moments(cnt)
#                 if M["m00"] != 0:
#                     center_x = int(M["m10"] / M["m00"])
#                     center_y = int(M["m01"] / M["m00"])
#                 else:
#                     center_x, center_y = 0, 0

#                 center_point = (center_x, center_y)
#                 cv2.circle(undistorted_frame, center_point, 5, (255, 0, 0), -1)

#                 # 밑면의 중심점 찾기
#                 bottom_points = sorted(box, key=lambda pt: pt[1], reverse=True)[:2]
#                 bottom_center_x = int((bottom_points[0][0] + bottom_points[1][0]) / 2)
#                 bottom_center_y = int((bottom_points[0][1] + bottom_points[1][1]) / 2)
#                 bottom_center_point = (bottom_center_x, bottom_center_y)

#                 cv2.circle(undistorted_frame, bottom_center_point, 5, (0, 0, 255), -1)
#                 cv2.line(undistorted_frame, center_point, bottom_center_point, (0, 255, 255), 2)

#                 # 기울기 계산
#                 angle = calculate_angle(center_point, bottom_center_point)
#                 cv2.putText(undistorted_frame, f"Angle: {angle:.2f}", (center_x, center_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
#                 print(f"Detected angle: {angle:.2f}")

#                 # 세그멘테이션 마스크를 컬러로 변환
#                 colored_mask = np.zeros_like(undistorted_frame)
#                 colored_mask[mask == 1] = [0, 255, 0]  # Green color for the mask

#                 # 원본 이미지에 세그멘테이션 마스크 오버레이
#                 overlay = cv2.addWeighted(undistorted_frame, 0.7, colored_mask, 0.3, 0)

#                 # 결과가 포함된 이미지 표시
#                 cv2.imshow('Segmentation Results', overlay)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


# import cv2
# import numpy as np
# import torch
# from ultralytics import YOLO

# # 세그멘테이션 모델 로드
# model = YOLO('/home/jinjuuk/dev_ws/pt_files/newjeans.pt')  # 세그멘테이션 모델 경로를 설정하세요.

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

# # 비디오 캡처 초기화 (카메라 ID는 0으로 설정, 필요 시 변경 가능)
# cap = cv2.VideoCapture(0)

# # 트랙바 콜백 함수 (아무 동작도 하지 않음)
# def nothing(x):
#     pass

# # 트랙바를 위한 윈도우 생성
# cv2.namedWindow('Segmentation Results')
# cv2.createTrackbar('Confidence', 'Segmentation Results', 50, 100, nothing)  # 기본값 50, 범위 0-100

# def calculate_angle(pt1, pt2):
#     angle = np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0]) * 180.0 / np.pi
#     return angle

# # 기울기 값을 저장할 리스트 초기화
# angle_list = []

# while True:
#     # 트랙바에서 현재 Confidence 값 가져오기
#     confidence_threshold = cv2.getTrackbarPos('Confidence', 'Segmentation Results') / 100.0

#     # 프레임 캡처
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # 왜곡 보정
#     undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs)

#     # 객체 검출 수행
#     results = model(undistorted_frame, conf=confidence_threshold)

#     for result in results:
#         if result.masks is None:
#             print("No masks detected.")
#             continue

#         masks = result.masks  # 세그멘테이션 마스크 가져오기
#         for mask in masks.data.cpu().numpy():  # CPU 메모리로 복사한 후 numpy 배열로 변환
#             mask = mask.astype(np.uint8)  # numpy 배열을 uint8로 변환
#             contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#             # 디버깅: 세그멘테이션 마스크 표시
#             debug_mask = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)
#             cv2.imshow('Debug Mask', debug_mask)

#             if contours:
#                 cnt = max(contours, key=cv2.contourArea)
#                 rect = cv2.minAreaRect(cnt)
#                 box = cv2.boxPoints(rect)
#                 box = np.int0(box)
#                 cv2.drawContours(undistorted_frame, [box], 0, (0, 255, 0), 2)

#                 # 중심점 계산
#                 M = cv2.moments(cnt)
#                 if M["m00"] != 0:
#                     center_x = int(M["m10"] / M["m00"])
#                     center_y = int(M["m01"] / M["m00"])
#                 else:
#                     center_x, center_y = 0, 0

#                 center_point = (center_x, center_y)
#                 cv2.circle(undistorted_frame, center_point, 5, (255, 0, 0), -1)

#                 # 밑면의 중심점 찾기
#                 bottom_points = sorted(box, key=lambda pt: pt[1], reverse=True)[:2]
#                 bottom_center_x = int((bottom_points[0][0] + bottom_points[1][0]) / 2)
#                 bottom_center_y = int((bottom_points[0][1] + bottom_points[1][1]) / 2)
#                 bottom_center_point = (bottom_center_x, bottom_center_y)

#                 cv2.circle(undistorted_frame, bottom_center_point, 5, (0, 0, 255), -1)
#                 cv2.line(undistorted_frame, center_point, bottom_center_point, (0, 255, 255), 2)

#                 # 기울기 계산
#                 angle = calculate_angle(center_point, bottom_center_point)
#                 angle_list.append(angle)

#                 # angle_list의 길이가 20이 되면 평균 계산
#                 if len(angle_list) == 20:
                   
#                     avg_angle = np.mean(angle_list)
#                     print(f"Average angle: {avg_angle:.2f}")
#                     # angle_list 초기화
#                     angle_list = []
#                     print("angle_list reset")
                
#                 # #개별 각도를 프레임에 출력
#                 # cv2.putText(undistorted_frame, f"Angle: {angle:.2f}", (center_x, center_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
#                 # print(f"Detected angle: {angle:.2f}")


#                 # 세그멘테이션 마스크를 컬러로 변환
#                 colored_mask = np.zeros_like(undistorted_frame)
#                 colored_mask[mask == 1] = [0, 255, 0]  # Green color for the mask

#                 # 원본 이미지에 세그멘테이션 마스크 오버레이
#                 overlay = cv2.addWeighted(undistorted_frame, 0.7, colored_mask, 0.3, 0)

#                 # 결과가 포함된 이미지 표시
#                 cv2.imshow('Segmentation Results', overlay)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()




import cv2
import numpy as np
import torch
from ultralytics import YOLO

# 세그멘테이션 모델 로드
model = YOLO('/home/jinjuuk/dev_ws/pt_files/newjeans.pt')  # 세그멘테이션 모델 경로를 설정하세요.

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

# 비디오 캡처 초기화 (카메라 ID는 0으로 설정, 필요 시 변경 가능)
cap = cv2.VideoCapture(0)

# 트랙바 콜백 함수 (아무 동작도 하지 않음)
def nothing(x):
    pass

# 트랙바를 위한 윈도우 생성
cv2.namedWindow('Segmentation Results')
cv2.createTrackbar('Confidence', 'Segmentation Results', 50, 100, nothing)  # 기본값 50, 범위 0-100

def calculate_angle(pt1, pt2):
    angle = np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0]) * 180.0 / np.pi
    return angle

# 기울기 값을 저장할 리스트 초기화
angle_list = []
avg_angle = None  # 평균 각도를 저장할 변수 초기화

while True:
    # 트랙바에서 현재 Confidence 값 가져오기
    confidence_threshold = cv2.getTrackbarPos('Confidence', 'Segmentation Results') / 100.0

    # 프레임 캡처
    ret, frame = cap.read()
    if not ret:
        break

    # 왜곡 보정
    undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs)

    # 객체 검출 수행
    results = model(undistorted_frame, conf=confidence_threshold)

    for result in results:
        if result.masks is None:
            print("No masks detected.")
            continue

        masks = result.masks  # 세그멘테이션 마스크 가져오기
        for mask in masks.data.cpu().numpy():  # CPU 메모리로 복사한 후 numpy 배열로 변환
            mask = mask.astype(np.uint8)  # numpy 배열을 uint8로 변환
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 디버깅: 세그멘테이션 마스크 표시
            debug_mask = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)
            cv2.imshow('Debug Mask', debug_mask)

            if contours:
                cnt = max(contours, key=cv2.contourArea)
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(undistorted_frame, [box], 0, (0, 255, 0), 2)

                # 중심점 계산
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    center_x = int(M["m10"] / M["m00"])
                    center_y = int(M["m01"] / M["m00"])
                else:
                    center_x, center_y = 0, 0

                center_point = (center_x, center_y)
                cv2.circle(undistorted_frame, center_point, 5, (255, 0, 0), -1)

                # 밑면의 중심점 찾기
                bottom_points = sorted(box, key=lambda pt: pt[1], reverse=True)[:2]
                bottom_center_x = int((bottom_points[0][0] + bottom_points[1][0]) / 2)
                bottom_center_y = int((bottom_points[0][1] + bottom_points[1][1]) / 2)
                bottom_center_point = (bottom_center_x, bottom_center_y)

                cv2.circle(undistorted_frame, bottom_center_point, 5, (0, 0, 255), -1)
                cv2.line(undistorted_frame, center_point, bottom_center_point, (0, 255, 255), 2)

                # 기울기 계산
                angle = calculate_angle(center_point, bottom_center_point)
                angle_list.append(angle)

                # angle_list의 길이가 20이 되면 평균 계산
                if len(angle_list) == 20:
                    avg_angle = np.mean(angle_list)
                    print(f"Average angle: {avg_angle:.2f}")
                    # angle_list 초기화
                    angle_list = []
                    print("angle_list reset")

                cv2.putText(undistorted_frame, f"Angle: {angle:.2f}", (center_x, center_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                print(f"Detected angle: {angle:.2f}")

    # 평균 각도가 있으면 디스플레이에 표시
    if avg_angle is not None:
        cv2.putText(undistorted_frame, f"Avg Angle: {avg_angle:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 234, 255), 2)

    # 세그멘테이션 마스크를 컬러로 변환
    colored_mask = np.zeros_like(undistorted_frame)
    colored_mask[mask == 1] = [0, 255, 0]  # Green color for the mask

    # 원본 이미지에 세그멘테이션 마스크 오버레이
    overlay = cv2.addWeighted(undistorted_frame, 0.7, colored_mask, 0.3, 0)

    # 결과가 포함된 이미지 표시
    cv2.imshow('Segmentation Results', overlay)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
