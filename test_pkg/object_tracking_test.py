import cv2
import torch
import numpy as np
from fastreid.config import get_cfg
from fastreid.engine import DefaultPredictor
from fastreid.data.transforms import build_transforms
from PIL import Image
from ultralytics import YOLO

# YOLOv5 Nano 모델 로드
model = YOLO("yolov5n.pt")

# BotSORT 설정 파일 경로
tracker_config = "/home/jinjuuk/.local/lib/python3.10/site-packages/ultralytics/cfg/trackers/botsort.yaml"

# FastReID 설정
cfg = get_cfg()
cfg.merge_from_file("/home/jinjuuk/fast-reid/configs/Market1501/bagtricks_R50.yml")
cfg.MODEL.WEIGHTS = "/home/jinjuuk/dev_ws/pt_files/market_mgn_R50-ibn.pth"
cfg.MODEL.DEVICE = "cuda"  # GPU를 사용할 경우 "cuda"로 변경

predictor = DefaultPredictor(cfg)
transform = build_transforms(cfg, is_train=False)

# 유사도 임계값 초기화
similarity_threshold = 50  # 0에서 100 사이의 값으로 초기화 (나중에 0.5로 매핑됨)

# 사람들의 특징 벡터를 저장할 딕셔너리 초기화
known_persons = {}

def calculate_similarity(features1, features2):
    """유사도 계산 함수."""
    similarity = np.dot(features1, features2.T) / (np.linalg.norm(features1) * np.linalg.norm(features2))
    return similarity.item()  # numpy.ndarray를 float로 변환

def update_threshold(x):
    """트랙바에서 임계값 업데이트."""
    global similarity_threshold
    similarity_threshold = x

# 윈도우 생성 및 트랙바 추가
cv2.namedWindow("Real-time Tracking and Re-ID")
cv2.createTrackbar("Similarity Threshold", "Real-time Tracking and Re-ID", similarity_threshold, 100, update_threshold)

# 웹캡으로부터 비디오 캡처
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO로 객체 감지 및 BotSORT로 추적
    results = model.track(classes=[0],            # Track only the "person" class
                          max_det=5,              # Detect up to 5 objects
                          show=False,             # YOLO 결과를 별도로 보여주지 않음
                          source=frame,           # 현재 프레임을 소스로 사용
                          tracker=tracker_config, # Use BotSORT tracker for Re-ID
                          stream=True             # Stream results in real-time
                          )

    for result in results:
        # 바운딩 박스와 트래킹 ID 추출
        boxes = result.boxes.xyxy.cpu().numpy().astype(int) if result.boxes is not None else []
        track_ids = result.boxes.id.cpu().numpy().astype(int) if result.boxes.id is not None else []

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            person_img = frame[y1:y2, x1:x2]

            # 글로벌 피처 추출 (FastReID)
            pil_img = transform(Image.fromarray(cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB))).unsqueeze(0)
            global_features = predictor(pil_img).cpu().numpy()

            # 유사도 계산 (임계값 적용)
            similar_ids = []
            threshold = similarity_threshold / 100.0  # 트랙바의 값을 0.0-1.0 범위로 변환
            for known_id, known_features in known_persons.items():
                similarity = calculate_similarity(global_features, known_features)
                if similarity > threshold:
                    similar_ids.append(known_id)
                    print(f"ID: {known_id} is similar with similarity: {similarity:.2f}")

            if similar_ids:
                # 유사한 ID가 있다면 가장 높은 유사도를 가진 ID를 사용
                person_id = max(similar_ids, key=lambda id: calculate_similarity(global_features, known_persons[id]))
                color = (0, 255, 0)  # 유사한 기존 ID가 있는 경우 초록색 사용
            else:
                # 새로운 ID 할당
                person_id = max(known_persons.keys(), default=0) + 1
                known_persons[person_id] = global_features
                color = (0, 0, 255)  # 새로운 ID의 경우 빨간색 사용

            # 바운딩 박스와 ID 및 중앙값 출력
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            print(f"ID: {person_id}, Center: ({center_x}, {center_y})")

            # 바운딩 박스와 레이블 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID: {person_id}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(frame, f"Center: ({center_x}, {center_y})", (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # 결과 프레임 출력
    cv2.imshow("Real-time Tracking and Re-ID", frame)

    # 'q' 키를 눌러 종료
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
