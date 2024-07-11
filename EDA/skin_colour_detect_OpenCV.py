import cv2
import numpy as np

def nothing(x):
    pass

# 카메라 캡처 초기화
cap = cv2.VideoCapture(0)

# 윈도우 생성
cv2.namedWindow('Skin Detection')

# 트랙바 생성
cv2.createTrackbar('H Lower', 'Skin Detection', 0, 179, nothing)
cv2.createTrackbar('H Upper', 'Skin Detection', 50, 179, nothing)
cv2.createTrackbar('S Lower', 'Skin Detection', 40, 255, nothing)
cv2.createTrackbar('S Upper', 'Skin Detection', 255, 255, nothing)
cv2.createTrackbar('V Lower', 'Skin Detection', 60, 255, nothing)
cv2.createTrackbar('V Upper', 'Skin Detection', 255, 255, nothing)

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break

    # BGR에서 HSV로 변환
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 트랙바 위치 읽기
    h_lower = cv2.getTrackbarPos('H Lower', 'Skin Detection')
    h_upper = cv2.getTrackbarPos('H Upper', 'Skin Detection')
    s_lower = cv2.getTrackbarPos('S Lower', 'Skin Detection')
    s_upper = cv2.getTrackbarPos('S Upper', 'Skin Detection')
    v_lower = cv2.getTrackbarPos('V Lower', 'Skin Detection')
    v_upper = cv2.getTrackbarPos('V Upper', 'Skin Detection')

    # 피부색 범위 설정
    lower_skin = np.array([h_lower, s_lower, v_lower], dtype=np.uint8)
    upper_skin = np.array([h_upper, s_upper, v_upper], dtype=np.uint8)

    # 피부색 범위 내의 픽셀들로 마스크 생성
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # 원본 이미지와 마스크를 AND 연산
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # 결과 화면 표시
    cv2.imshow('Original', frame)
    cv2.imshow('Skin Mask', mask)
    cv2.imshow('Skin Detection', result)

    # 'q' 키를 누르면 루프 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 캡처 해제 및 창 닫기
cap.release()
cv2.destroyAllWindows()
