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

# # 1. 로봇 좌표계에서의 카메라 위치
# camera_position_robot = np.array([0, 9.3, 80])

# # 2. 카메라가 바라보는 방향 각도 (59도 or 122도)
# theta_deg = -59
# theta_rad = np.deg2rad(theta_deg)  # 라디안으로 변환



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

#                 # 3. 아르코마커의 카메라 좌표계에서의 위치 (2D 좌표 -> 3D로 확장)
#                 marker_position_cam = np.array([center_x, center_y, 0])

#                 # 4. 카메라의 y축 회전 변환 행렬 정의
#                 R_y = np.array([
#                     [np.cos(theta_rad), 0, np.sin(theta_rad)],
#                     [0, 1, 0],
#                     [-np.sin(theta_rad), 0, np.cos(theta_rad)]
#                 ])


#                 # 5. 카메라 좌표계에서 로봇 좌표계로의 변환 수행
#                 marker_position_robot = R_y.dot(marker_position_cam) + camera_position_robot
#                 print(f"ArUco Marker ID: {id[0]}, Transformed Robot Coor: ({marker_position_robot})")
#                 print()



#         # center_x = 442
#         # center_y = 150
              
#         # print(f" Position: ({center_x}, {center_y})")
#         # print()

#         # # 3. 아르코마커의 카메라 좌표계에서의 위치 (2D 좌표 -> 3D로 확장)
#         # marker_position_cam = np.array([center_x, center_y, 0])

#         # # 4. 카메라의 y축 회전 변환 행렬 정의
#         # R_y = np.array([
#         #     [np.cos(theta_rad), 0, np.sin(theta_rad)],
#         #     [0, 1, 0],
#         #     [-np.sin(theta_rad), 0, np.cos(theta_rad)]
#         # ])


#         # # 5. 카메라 좌표계에서 로봇 좌표계로의 변환 수행
#         # marker_position_robot = R_y.dot(marker_position_cam) + camera_position_robot
#         # print(f" Transformed Robot Coor: ({marker_position_robot})")
#         # print()


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
import cv2.aruco as aruco

# 캘리브레이션 데이터를 사용하여 카메라 매트릭스와 왜곡 계수를 설정
camera_matrix = np.array([[459.77670243, 0, 325.24447432], 
                          [0, 463.47026232, 227.25521262], 
                          [0, 0, 1]])
dist_coeffs = np.array([0.01718349, -0.26100891, 0.00317879, -0.00069208, 0.26564309])

# ArUco 마커 딕셔너리 선택
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

# 로봇 좌표계의 중앙값 (픽셀 단위)
robot_origin_x = 316  # 카메라에서 보이는 로봇의 중심 픽셀 좌표의 x값
robot_origin_y = 442  # 카메라에서 보이는 로봇의 중심 픽셀 좌표의 y값

# 실제 로봇의 중심 좌표 (로봇 좌표계)
real_robot_base_x = 0  # 실제 중심 좌표의 x값
real_robot_base_y = 0  # 실제 중심 좌표의 y값

# 카메라에서 보이는 아르코마커의 픽셀 좌표 (예제 값)
cam_arcomarker_x = 442  # 픽셀 x좌표
cam_arcomarker_y = 150 # 픽셀 y좌표

# 실제 로봇 좌표계에서 아르코마커의 목표 좌표 (예제 값)
real_arcomaker_x = -214.1  # 로봇 좌표의 x값 
real_arcomaker_y = -337.7  # 로봇 좌표의 y값

# 변환 비율을 계산
delta_x_cam = cam_arcomarker_x - robot_origin_x
delta_y_cam = cam_arcomarker_y - robot_origin_y

delta_x_real = real_arcomaker_x - real_robot_base_x
delta_y_real = real_arcomaker_y - real_robot_base_y

# 픽셀 단위와 실제 거리 단위 간의 스케일 계산
scale_x = delta_x_real / delta_x_cam
scale_y = delta_y_real / delta_y_cam

# 카메라의 로봇 좌표계에서의 위치와 방향
camera_position_robot = np.array([0, 930, 800]) #cm
theta_deg = -59  # -y 방향으로 회전한 각도
theta_rad = np.deg2rad(theta_deg)

# y축 회전 행렬 정의
R_y = np.array([
    [np.cos(theta_rad), 0, np.sin(theta_rad)],
    [0, 1, 0],
    [-np.sin(theta_rad), 0, np.cos(theta_rad)]
])

# 비디오 캡처 초기화 (카메라 ID는 0으로 설정, 필요 시 변경 가능)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("비디오 프레임을 가져올 수 없습니다.")
        break

    # 왜곡 보정된 프레임 생성
    undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None)

    # 그레이스케일로 변환
    gray = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2GRAY)

    # ArUco 마커 감지
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict)

    if ids is not None:
        # Pose estimation
        marker_length = 0.045  # 4.5 cm
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)
        
        for i, id in enumerate(ids):
            if id[0] == 3 or id[0] == 4:
                # Draw the marker axis for visualization
                cv2.drawFrameAxes(undistorted_frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.1)

                # Extract the translation vector (tvec)
                tvec = tvecs[i][0]
                print(f"ArUco Marker ID: {id[0]}, Position in camera coordinates: {tvec}")

                # 카메라 좌표계에서 로봇 좌표계로 변환
                # 카메라 좌표를 로봇 좌표로 변환하는 과정 추가
                # 카메라 좌표에서 로봇 좌표로 변환하기 위해 픽셀 스케일 적용
                real_x = (tvec[0] - robot_origin_x) * scale_x + real_robot_base_x
                real_y = (tvec[1] - robot_origin_y) * scale_y + real_robot_base_y

                # 변환된 로봇 좌표
                marker_position_robot = R_y.dot(np.array([real_x, real_y, 0])) + camera_position_robot
                print(f"ArUco Marker ID: {id[0]}, Position in robot coordinates: {marker_position_robot[:2]}")

    # 프레임을 화면에 표시
    cv2.imshow('ArUco Marker Detection', undistorted_frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 사용이 끝난 후 비디오 캡처 객체와 모든 창 닫기
cap.release()
cv2.destroyAllWindows()
