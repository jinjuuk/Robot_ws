# import os
# import sys
# import cv2
# import numpy as np
# import cv2.aruco as aruco

# # 캘리브레이션 데이터를 사용하여 카메라 매트릭스와 왜곡 계수를 설정
# camera_matrix = np.array([[468.79867234,   0,         317.32738292],
#                         [  0,         471.55156912, 222.53655653],
#                         [  0,           0,           1        ]])
# dist_coeffs = np.array([[ 0.01251737, -0.11174468,  0.00256191, -0.00344855,  0.09965856]])

# # ArUco 마커 딕셔너리 선택
# aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

# # 로봇 좌표계의 중앙값 (픽셀 단위)
# robot_origin_x = 316 # 카메라에서 보이는 로봇의 중심 픽셀 좌표의 x값
# robot_origin_y = 442 # 카메라에서 보이는 로봇의 중심 픽셀 좌표의 y값

# real_robot_base_x = 0 # 로봇의 실제 중심 좌표의 x값
# real_robot_base_y = 0 # 로봇의 실제 중심 좌표의 y값

# cam_arcomarker_x = [-120.8, 442, 251, 169] # 카메라에서 보이는 아르코마커의 픽셀 x좌표
# cam_arcomarker_y = [-358.84, 150, 138, 145] # 카메라에서 보이는 아르코마커의 픽셀 y좌표

# real_arcomaker_x = [-134.9, -214.7, 119.3, 260.4] #실제 로봇이 위치해야할 로봇좌표의 x값
# real_arcomaker_y = [-367.8, -337.7, -357.2, -344.4] #실제 로봇이 위치해야할 로봇좌표의 y값

# # 변환 비율을 한 번만 계산합니다.
# # 로봇의 중심으로부터 카메라에서 보이는 아르코마커의 좌표 차이 (픽셀 단위)
# delta_x_cam = cam_arcomarker_x - robot_origin_x
# delta_y_cam = cam_arcomarker_y - robot_origin_y

# # 로봇의 실제 좌표계에서 아르코마커의 좌표 차이 (실제 거리 단위)
# delta_x_real = real_arcomaker_x - real_robot_base_x
# delta_y_real = real_arcomaker_y - real_robot_base_y

# # 픽셀 단위와 실제 거리 단위 간의 스케일 계산
# scale_x = delta_x_real / delta_x_cam
# scale_y = delta_y_real / delta_y_cam



# # 비디오 캡처 초기화 (카메라 ID는 0으로 설정, 필요 시 변경 가능)
# cap = cv2.VideoCapture(0)

# # 카메라 밝기 설정 (0 ~ 1 사이의 값으로 설정)
# cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)  # 밝기를 중간 정도로 설정

# # 비디오 프레임 크기 가져오기
# ret, frame = cap.read()
# h, w = frame.shape[:2]

# # 새로운 카메라 매트릭스를 계산하고 ROI를 가져옵니다
# new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))

# while True:
#     ret, frame = cap.read()

#     if not ret:
#         print("비디오 프레임을 가져올 수 없습니다.")
#         break

#     # 왜곡 보정된 프레임 생성
#     undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)

#     # 그레이스케일로 변환
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # ArUco 마커 감지
#     corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict)

#     # 감지된 마커를 프레임에 그리기
#     if ids is not None:
#         cv2.aruco.drawDetectedMarkers(undistorted_frame, corners, ids)

#         # ID가 3번과 4번인 마커의 위치 출력
#         for i, id in enumerate(ids):
#             if id[0] == 3 or id[0] == 4:
#                 # 마커의 중심 좌표 계산
#                 c = corners[i][0]
#                 center_x = int((c[0][0] + c[1][0] + c[2][0] + c[3][0]) / 4)
#                 center_y = int((c[0][1] + c[1][1] + c[2][1] + c[3][1]) / 4)
#                 print(f"ArUco Marker ID: {id[0]}, Position: ({center_x}, {center_y})")
#                 print()
#                 # 변환 비율을 사용하여 로봇 좌표로 변환
#                 # 카메라에서 보이는 아르코마커의 픽셀 좌표를 실제 로봇 좌표로 변환
#                 real_x = real_robot_base_x + scale_x * (center_x - robot_origin_x)
#                 real_y = real_robot_base_y + scale_y * (center_y - robot_origin_y)
#                 # 1차 설정
#                 #(-120.8, -358.84)
#                 #(-134.9, -367.8)
                
#                 # 2차 설정
#                 #(442, 150)
#                 #(-214.7, -337.7)

#                 # 3차 설정
#                 #(251, 138)
#                 #(119.3, -357.2)
                
#                 # 4차 설정
#                 #(169, 145)
#                 #(260.4, -344.4)
#                 print(f"ArUco Marker ID: {id[0]}, Transformed Robot Coor: ({real_x}, {real_y})")
#                 print()
#     # 프레임을 화면에 표시
#     cv2.imshow('ArUco Marker Detection', undistorted_frame)

#     # 'q' 키를 누르면 종료
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # 사용이 끝난 후 비디오 캡처 객체와 모든 창 닫기
# cap.release()
# cv2.destroyAllWindows()







import cv2
import numpy as np
import math

# 카메라 매트릭스 설정
camera_matrix = np.array([[468.0575217805572, 0, 323.884225502888],
                          [0, 469.436228483968, 219.7904487508031],
                          [0, 0, 1]])

# 왜곡 계수 (없음)
dist_coeffs = np.array([])  # Distortion model is 'none', so no distortion coefficients

# 회전 행렬 생성 (x축 기준, theta = 145도)
theta_deg = 145
theta_rad = math.radians(theta_deg)
cos_theta = math.cos(theta_rad)
sin_theta = math.sin(theta_rad)

rotation_matrix_x = np.array([
    [1, 0, 0],
    [0, cos_theta, -sin_theta],
    [0, sin_theta, cos_theta]
])

# ArUco 마커 설정 (6x6 크기, 250개의 마커를 포함하는 사전 사용)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()

# 웹캠 캡처 시작
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("카메라에서 프레임을 가져올 수 없습니다.")
        break

    # 그레이스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ArUco 마커 검출
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        # 검출된 마커의 경계선을 그립니다.
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    
        # 각 마커에 대해 포즈 추정
        for corner, marker_id in zip(corners, ids):
            # 마커의 포즈 추정
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corner, 0.05, camera_matrix, dist_coeffs)

            # 마커의 좌표 추출 (카메라 좌표계 기준)
            marker_pos = tvecs[0][0]

            # 회전 행렬 적용하여 좌표 변환
            rotated_marker_pos = rotation_matrix_x.dot(marker_pos)

            # x 값의 부호 반전
            rotated_marker_pos[0] = -rotated_marker_pos[0]

            # 좌표 출력
            print(f"Marker ID: {marker_id[0]}")
            print("Original Marker Position (Camera Coord):", marker_pos)
            print("Rotated Marker Position:", rotated_marker_pos)

            # 마커의 중심 좌표 계산
            center_x = int(np.mean(corner[0][:, 0]))
            center_y = int(np.mean(corner[0][:, 1]))

            # 변환된 좌표를 프레임에 표시
            text = f"ID: {marker_id[0]} ({rotated_marker_pos[0]:.2f}, {rotated_marker_pos[1]:.2f}, {rotated_marker_pos[2]:.2f})"
            cv2.putText(frame, text, (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 결과 이미지 보여주기
    cv2.imshow('ArUco Marker Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 웹캠 릴리즈 및 창 닫기
cap.release()
cv2.destroyAllWindows()


