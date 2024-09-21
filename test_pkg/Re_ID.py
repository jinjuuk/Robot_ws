import collections
if not hasattr(collections, 'Mapping'):
    import collections.abc
    collections.Mapping = collections.abc.Mapping

import cv2
import torch
import numpy as np
from fastreid.config import get_cfg
from fastreid.engine import DefaultPredictor
from fastreid.data.transforms import build_transforms
from PIL import Image


# YOLO 모델 로드
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# FastReID 설정
cfg = get_cfg()
cfg.merge_from_file("/home/jinjuuk/fast-reid/configs/Market1501/bagtricks_R50.yml")
cfg.MODEL.WEIGHTS = "/home/jinjuuk/dev_ws/pt_files/market_mgn_R50-ibn.pth"
cfg.MODEL.DEVICE = "cpu"

predictor = DefaultPredictor(cfg)
transform = build_transforms(cfg, is_train=False)

# 비디오 캡처 설정
cap = cv2.VideoCapture(0)

# 이전에 감지된 사람들의 특징 벡터를 저장할 딕셔너리
known_persons = {}
person_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO를 사용하여 사람 감지
    results = yolo_model(frame)
    
    # 감지된 사람들에 대해 처리
    for *xyxy, conf, cls in results.xyxy[0]:
        if int(cls) == 0:  # 0은 COCO 데이터셋에서 '사람' 클래스를 나타냅니다
            x1, y1, x2, y2 = map(int, xyxy)
            person_img = frame[y1:y2, x1:x2]
            
            # FastReID를 사용하여 특징 벡터 추출
            img_pil = Image.fromarray(cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB))
            image_tensor = transform(img_pil).unsqueeze(0)
            with torch.no_grad():
                features = predictor(image_tensor).cpu().numpy()

            # 가장 유사한 사람 찾기
            max_similarity = -1
            matched_id = -1
            for id, known_features in known_persons.items():
                similarity = np.dot(features[0], known_features) / (np.linalg.norm(features[0]) * np.linalg.norm(known_features))
                if similarity > max_similarity:
                    max_similarity = similarity
                    matched_id = id

            # 유사도 임계값 (조정 가능)
            threshold = 0.7

            if max_similarity > threshold:
                # 기존 사람으로 인식
                label = f"Person {matched_id}"
            else:
                # 새로운 사람으로 인식
                person_id += 1
                known_persons[person_id] = features[0]
                label = f"Person {person_id}"

            # 바운딩 박스와 레이블 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 결과 프레임 출력
    cv2.imshow('Real-time ReID', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

