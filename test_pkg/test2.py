import cv2
import numpy as np
import cv2.aruco as aruco

# 캘리브레이션 데이터를 사용하여 카메라 매트릭스와 왜곡 계수 설정
camera_matrix = np.array([[468.79867234, 0, 317.32738292],
                          [0, 471.55156912, 222.53655653],
                          [0, 0, 1]])
dist_coeffs = np.array([0.01251737, -0.11174468, 0.00256191, -0.00344855, 0.09965856])

# ArUco 마커 딕셔너리 정의
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters_create()

# 웹캠 비디오 스트림 열기
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 그레이스케일 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 마커 탐지
    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        # 포즈 추정
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_coeffs)

        for rvec, tvec in zip(rvecs, tvecs):
            # 축 그리기
            cv2.aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)

    # 마커 테두리 그리기
    cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    # 화면에 결과 보여주기
    cv2.imshow('Frame', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()

