import cv2
import numpy as np
import time
import torch
import cv2.aruco as aruco
import mediapipe as mp
from ultralytics import YOLO


#별 모양을 검출하는 함수(아이스크림 통을 짜면 별모양으로 나오는 통을 인식)
def detect_star_shape(image, canny_thresh1, canny_thresh2): 

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # 회색조 변환, 입력된 컬러 이미지를 회색조 이미지로 변환한다.(엣지 및 컨투어 검출을 더 쉽게 하기 위함)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0) # 가우시안 블러를 사용하여 이미지를 부드럽게 만든다(노이즈를 줄이고 엣지 검출을 더 정확하게 하기 위함이다)
    edged = cv2.Canny(blurred, canny_thresh1, canny_thresh2) # #canny edge detector algorithm을 사용하여 이미지의 엣지를 검출 (canny_thresh1, canny_thresh2는 엣지를 검출에 사용되는 임계값)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 검출된 엣지를 기반으로 컨투어(외곽선)를 검출합니다. cv2.RETR_EXTERNAL은 외부 컨투어만을 검출하고, cv2.CHAIN_APPROX_SIMPLE은 컨투어 포인트를 간소화하여 저장합니다.

    # 검출된 컨투어의 수가 1개 또는 2개인 경우에만 다음 단계를 진행합니다. 이는 별 모양을 검출하기 위한 예비 조건입니다.
    if len(contours) == 1 or len(contours) == 2: # 컨투어 수 검사

        contour = contours[0] #첫번째 컨투어를 선택
        epsilon = 0.02 * cv2.arcLength(contour, True) # 컨투어의 둘레 길이에 기반하여 근사화 정밀도를 설정합니다. epsilon은 근사화 정확도를 제어합니다.
        approx = cv2.approxPolyDP(contour, epsilon, True) # 선택된 컨투어를 다각형으로 근사화합니다. True는 컨투어가 닫혀있음을 나타냅니다.

        # 별 모양 판단
        if len(approx) >= 10:  # 근사화된 다각형의 점(꼭지점) 개수가 10개 이상인 경우에만 별 모양으로 판단합니다. 별 모양은 일반적으로 많은 꼭지점을 가지므로, 이를 기준으로 합니다.
            return True, approx, edged, len(contours) # 별 모양이 검출되면, True와 근사화된 다각형(컨투어), 엣지 이미지, 컨투어 수를 반환합니다

    return False, None, edged, len(contours) # 별 모양이 아니면, False와 None, 엣지 이미지, 컨투어 수를 반환합니다.


# 트랙바 콜백 함수
# 이 함수는 트랙바의 콜백 함수로, 아무 작업도 수행하지 않습니다. 트랙바를 만들 때 콜백 함수가 필요하지만, 여기서는 특정 동작을 수행하지 않기 때문에 빈 함수로 정의됩니다.
def nothing(x):
    
    pass


# 카메라 초기화
cap = cv2.VideoCapture(0) # VideoCapture 객체를 생성하여 카메라 스트림을 캡처합니다. 여기서 1은 두 번째 카메라를 사용함을 의미합니다(기본 카메라는 0)



# YOLOv8 model
model = YOLO('/home/jinjuuk/dev_ws/pt_files/new_custom_m_freeze8.pt') # YOLOv8 객체 검출 모델을 로드합니다

device = 'cuda' if torch.cuda.is_available() else 'cpu' # 사용 가능한 디바이스를 설정합니다. 만약 GPU(CUDA)가 사용 가능하면 'cuda', 그렇지 않으면 'cpu'를 사용합니다.

# GPU가 사용 가능한 경우 모델을 GPU로 이동시킵니다.
if torch.cuda.is_available(): 
    model.to(device) 

print(f'Using device: {device}') # 사용 중인 디바이스를 출력하여 확인합니다.



# MediaPipe 손 검출 초기화
mp_hands = mp.solutions.hands # MediaPipe의 손 검출 모듈을 초기화합니다
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7) # 손 검출기를 설정합니다. 최대 두 개의 손을 검출하며, 최소 검출 신뢰도는 0.7로 설정합니다.



# ArUco 마커 검출 초기화
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250) # ArUco 마커 사전을 설정합니다. 여기서는 6x6 크기의 250개 마커를 사용하는 사전(DICT_6X6_250)을 사용합니다.
parameters = aruco.DetectorParameters() # ArUco 마커 검출 파라미터를 설정합니다.
last_seen = {} # 마지막으로 본 마커의 ID와 시간을 저장하기 위한 딕셔너리를 초기화합니다.



# 트랙바의 임계값, 밝기
cv2.namedWindow('Frame')
cv2.createTrackbar('Threshold', 'Frame', 0, 20000, nothing)
cv2.createTrackbar('Canny Thresh1', 'Frame', 50, 255, nothing)
cv2.createTrackbar('Canny Thresh2', 'Frame', 112, 255, nothing)
cv2.createTrackbar('Brightness', 'Frame', 50, 100, nothing)
cv2.createTrackbar('Confidence', 'Frame', 50, 100, nothing)



# 트랙바의 쓰레기 탐지 
cv2.createTrackbar('Diff_Thresh', 'Frame', 45, 255, nothing)
cv2.createTrackbar('Hue Min', 'Frame', 0, 179, nothing)
cv2.createTrackbar('Hue Max', 'Frame', 179, 179, nothing)
cv2.createTrackbar('Sat Min', 'Frame', 0, 255, nothing)
cv2.createTrackbar('Sat Max', 'Frame', 255, 255, nothing)
cv2.createTrackbar('Val Min', 'Frame', 0, 255, nothing)
cv2.createTrackbar('Val Max', 'Frame', 255, 255, nothing)



# 여러 관심 영역(ROI, Region of Interest)과 배경 제거 객체를 설정하는 부분
# 세 가지 ROI를 정의하는 이유는 각기 다른 크기와 위치에서 특정한 객체나 이벤트를 감지하기 위해서입니다.
# 큰 ROI정의
roi_x_large = 52
roi_y_large = 0
roi_width_large = 500
roi_height_large = 310


# 중간 ROI정의
roi_x_medium = 270
roi_y_medium = 0
roi_width_medium = 270
roi_height_medium = 60


# 작은 ROI정의
roi_x_small = 464
roi_y_small = 118
roi_width_small = 35
roi_height_small = 35


# 배경 제거 객체 초기화
fgbg = cv2.createBackgroundSubtractorMOG2() # MOG2(Mixture of Gaussians 2) 방법을 사용하여 배경 제거 객체를 생성
# 이 객체는 프레임 간의 차이를 계산하여 움직이는 객체를 배경에서 분리하는 데 사용됩니다. 이는 영상에서 움직이는 물체를 검출하는 데 유용



# 초기 배경 이미지 변수
initial_gray = None
post_cleanup_gray = None
detection_enabled = False
cleanup_detection_enabled = False


# 실시간 비디오 스트림을 처리하면서 트랙바를 통해 조절된 다양한 파라미터를 사용하여 이미지 처리를 수행하는 부분
# 비디오 프레임 캡처
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from camera.")
        break

    
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

    
    # 프레임 밝기 조정
    frame = cv2.convertScaleAbs(frame, alpha=1, beta=(brightness - 50) * 2) # cv2.convertScaleAbs를 사용하여 프레임의 밝기를 조정합니다. alpha=1로 설정하여 기본 대비를 유지하고, beta=(brightness - 50) * 2를 통해 트랙바 값에 따라 밝기를 조절

    # 배경 제거 마스크 생성
    fgmask = fgbg.apply(frame) # fgbg.apply(frame)를 사용하여 배경 제거 마스크를 생성합니다. 이 마스크는 현재 프레임과 배경 모델 간의 차이를 계산하여 움직이는 객체를 검출

    # 특정 관심 영역(ROI) 내에서 침입을 감지하는 부분. 이 기능은 배경 제거 마스크를 사용하여 ROI 내에서 움직임이 있는지 확인.
    roi_large = fgmask[roi_y_large:roi_y_large + roi_height_large, roi_x_large:roi_x_large + roi_width_large] # 큰 ROI 내의 배경 제거 마스크 추출, fgmask는 이전에 생성된 배경 제거 마스크, fgmask에서 정의된 ROI 영역을 추출하여 roi_large 변수에 저장합니다. 이는 큰 ROI 내의 배경 제거 마스크를 의미

    # 침입 감지
    intrusion_detected = np.sum(roi_large) > threshold 

    

    # MediaPipe를 사용하여 손을 검출하고, 특정 영역 내에서 손이 검출되었을 때 침입 경고를 표시하는 부분
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # 프레임을 RGB로 변환

    result = hands.process(frame_rgb) # MediaPipe를 사용하여 손 검출

    if result.multi_hand_landmarks: # 손 랜드마크가 검출된 경우 처리
        for hand_landmarks in result.multi_hand_landmarks: # result.multi_hand_landmarks가 None이 아닌 경우, 하나 이상의 손 랜드마크가 검출되었음을 의미합니다.
            for landmark in hand_landmarks.landmark: # 각 손에 대해 반복문을 돌면서 손 랜드마크를 처리합니다.
                
                # 손 랜드마크는 손의 각 관절 위치를 나타내며, 각 랜드마크의 x, y 좌표를 이미지 크기에 맞게 변환합니다.
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])

                # 변환된 x, y 좌표가 큰 ROI 내에 있고, intrusion_detected가 True인 경우, 침입 경고 메시지를 프레임에 표시
                if roi_x_large < x < roi_x_large + roi_width_large and roi_y_large < y < roi_y_large + roi_height_large and intrusion_detected:
                    cv2.putText(frame, "Warning: Intrusion detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    break

            # mp.solutions.drawing_utils.draw_landmarks 함수를 사용하여 프레임에 손 랜드마크와 연결선을 그린다
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    

    # ArUco 마커를 검출하고, 특정 시간이 지난 후 마커에 대해 액션을 실행하는 기능을 수행
    roi_medium = frame[roi_y_medium:roi_y_medium + roi_height_medium, roi_x_medium:roi_x_medium + roi_width_medium] # 중간 ROI 설정
    corners, ids, rejectedImgPoints = aruco.detectMarkers(roi_medium, aruco_dict, parameters=parameters) # ArUco 마커 검출
    current_time = time.time() # 현재 시간 기록

    # 마커가 검출된 경우 처리
    if ids is not None and len(ids) > 0: # ids가 None이 아니고 길이가 0보다 크면 마커가 검출
        aruco.drawDetectedMarkers(roi_medium, corners, ids) # aruco.drawDetectedMarkers 함수를 사용하여 검출된 마커를 ROI에 그림

        for id in ids.flatten(): # 검출된 각 마커의 ID를 반복하며 last_seen 딕셔너리에 현재 시간을 기록
            last_seen[id] = current_time

        # 현재 시간과 마지막으로 검출된 시간의 차이가 3초 이상인 경우 Action executed for marker ID {id}"라는 메시지 출력
        for id, last_time in list(last_seen.items()): 
            if current_time - last_time > 3:
                cv2.putText(frame, f"Action executed for marker ID {id}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                del last_seen[id]
                break

    

    # 특정 작은 관심 영역(ROI) 내에서 별 모양을 검출하고, 그 결과를 프레임에 표시하는 기능을 수행
    small_roi = frame[roi_y_small:roi_y_small + roi_height_small, roi_x_small:roi_x_small + roi_width_small] # 작은 ROI 설정
    star_detected, star_contour, edged, contour_count = detect_star_shape(small_roi, canny_thresh1, canny_thresh2) # 별 모양 검출

    # 별 모양이 검출된 경우 처리
    if star_detected:
        star_contour += [roi_x_small, roi_y_small] # 검출된 컨투어의 좌표를 원래 프레임의 좌표로 변환
        cv2.drawContours(frame, [star_contour], -1, (0, 255, 0), 3) # 프레임에 검출된 별 모양의 컨투어를 그린다.
        cv2.putText(frame, "Star detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2) # "Star detected"라는 텍스트를 프레임의 (10, 30) 위치에 표시
    
    # 별 모양이 검출되지 않은 경우
    else:
        cv2.putText(frame, "Star not detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2) # Star not detected"라는 텍스트를 프레임의 (10, 30) 위치에 표시

    

    # Trash detection
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.GaussianBlur(frame_gray, (21, 21), 0)
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    

    # 초기값 설정 또는 해제
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        initial_gray = frame_gray
        detection_enabled = True
        print("Initial background set, detection enabled.")

    

    # 검출 활성화된 경우에만 실행
    if detection_enabled and initial_gray is not None and not cleanup_detection_enabled:

        # 초기 이미지와 현재 이미지의 차이 계산
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
        # 탐지된 객체를 저장할 리스트 초기화
        detected_objects = []

        
        # 윤곽선을 순회하며 조건 검사
        for cnt in contours:
            if cv2.contourArea(cnt) > 80:  # 최소 면적 기준, 윤곽선의 면적이 80보다 큰 경우에만 처리, 노이즈나 작은 객체를 무시하기 위한 최소 면적 기준
                x, y, w, h = cv2.boundingRect(cnt) # 윤곽선을 포함하는 최소 크기의 직사각형

                # 탐지된 객체가 지정된 탐지 영역 내에 있는지 확인
                if (roi_x_large <= x <= roi_x_large + roi_width_large and
                    roi_y_large <= y <= roi_y_large + roi_height_large):

                    detected_objects.append((x, y, w, h)) # 탐지된 객체의 좌표와 크기를 리스트에 추가
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, 'Trash Detected', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    

    # YOLOv8 객체 검출을 위한 큰 ROI 설정
    yolo_roi = frame[roi_y_large:roi_y_large + roi_height_large, roi_x_large:roi_x_large + roi_width_large] # 전체 프레임에서 큰 ROI를 설정
    results = model(yolo_roi) # YOLOv8 모델을 사용하여 객체 검출, 

    for result in results: # results 변수에는 검출된 여러 객체의 정보가 포함
        boxes = result.boxes # 각 result에 대해 객체의 boxes를 가져옴

        for box in boxes: # boxes에는 검출된 각 객체의 경계 상자 정보가 포함
            cls_id = int(box.cls) # 객체의 클래스 ID를 정수로 변환
            label = model.names[cls_id] # 클래스 ID를 사용하여 객체의 라벨을 가져온다
            confidence = box.conf.item() # 객체 검출의 신뢰도를 가져온다

            if confidence >= confidence_threshold: # 신뢰도가 설정된 임계값을 넘는 경우에만 처리를 진행
                bbox = box.xyxy[0].tolist() # 경계 상자의 좌표를 리스트로 변환
                x1, y1, x2, y2 = map(int, bbox)
                x1 += roi_x_large
                y1 += roi_y_large
                x2 += roi_x_large
                y2 += roi_y_large

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # 출된 객체의 경계 상자를 프레임에 그린다
                text = f'{label} {confidence:.2f}' # 객체의 라벨과 신뢰도를 텍스트로 설정
                # 텍스트의 위치를 설정
                text_x = x1 
                text_y = y1 - 10 if y1 - 10 > 10 else y1 + 10

                cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) # 객체 라벨과 신뢰도를 프레임에 텍스트로 표시

                # 객체의 라벨이 'cup' 또는 'star'인 경우
                if label == 'cup' or label == 'star':
                    cv2.putText(frame, 'Trash Detected', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2) # Trash Detected'라는 텍스트를 경고 메시지로 표시

    

    # Draw ROI rectangles
    cv2.rectangle(frame, (roi_x_large, roi_y_large), (roi_x_large + roi_width_large, roi_y_large + roi_height_large), (255, 0, 0), 2)
    cv2.rectangle(frame, (roi_x_medium, roi_y_medium), (roi_x_medium + roi_width_medium, roi_y_medium + roi_height_medium), (0, 255, 0), 2)
    cv2.rectangle(frame, (roi_x_small, roi_y_small), (roi_x_small + roi_width_small, roi_y_small + roi_height_small), (0, 0, 255), 2)
    cv2.putText(frame, f"Contours: {contour_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    

    # Display the edged image in the small ROI
    frame[roi_y_small:roi_y_small + roi_height_small, roi_x_small:roi_x_small + roi_width_small] = cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR)

    

    cv2.imshow('Frame', frame)
    if key == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()