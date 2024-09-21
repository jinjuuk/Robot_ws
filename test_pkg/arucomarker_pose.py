import cv2
import cv2.aruco as aruco
import numpy as np
import time

# 마커를 생성하는 데 사용된 딕셔너리를 로드합니다.
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

# 기본값을 사용하여 탐지기 매개변수를 초기화합니다.
parameters = aruco.DetectorParameters()

# # 카메라 행렬과 왜곡 계수 설정 (예제 값)
# # 실제 카메라 매개변수를 사용해야 합니다.
# camera_matrix = np.array([[1000, 0, 320],
#                           [0, 1000, 240],
#                           [0, 0, 1]], dtype=float)
# dist_coeffs = np.zeros((4, 1))  # 왜곡 계수를 알 수 없을 때 0으로 설정

camera_matrix = np.array([[468.79867234, 0, 317.32738292],
                          [0, 471.55156912, 222.53655653],
                          [0, 0, 1]])
dist_coeffs = np.array([[0.01251737, -0.11174468, 0.00256191, -0.00344855, 0.09965856]])

# 각 마커의 마지막 감지 시간을 저장할 딕셔너리 초기화
last_seen = {}

# 웹캠을 통해 비디오 캡처를 시작합니다.
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ROI 영역에서 마커 감지
    corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

    # 현재 시간 기록
    current_time = time.time()

    # 감지된 마커를 ROI 이미지에 그립니다.
    if ids is not None and len(ids) > 0:
        aruco.drawDetectedMarkers(frame, corners, ids)
        print(f"Detected ArUco markers: {ids.flatten()}")

        # 마커 위치와 회전 벡터 계산
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_coeffs)

        # 각 마커의 위치 및 회전 정보를 출력합니다.
        for i, id in enumerate(ids.flatten()):
            rvec = rvecs[i][0]
            tvec = tvecs[i][0]
            print(f"Marker ID {id}: Rotation Vector (rvec) = {rvec}, Translation Vector (tvec) = {tvec}")

            # 마커의 축을 그립니다.
            # aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)


            # 감지된 모든 마커의 마지막 감지 시간을 업데이트합니다.
            last_seen[id] = current_time

        # 모든 마커에 대해 마지막 감지 시간 확인
        for id, last_time in list(last_seen.items()):
            if current_time - last_time > 5:
                print(f"Action executed for marker ID {id} (not moved for 5 seconds)")
                del last_seen[id]
                break
    else:
        print("No markers detected")

    # 감지된 마커가 있는 이미지를 표시합니다.
    cv2.imshow('frame', frame)

    # 'q' 키를 누르면 루프를 종료합니다.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 캡처 객체와 모든 창을 해제 및 닫습니다.
cap.release()
cv2.destroyAllWindows()