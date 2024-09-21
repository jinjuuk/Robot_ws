# import collections
# if not hasattr(collections, 'Mapping'):
#     import collections.abc
#     collections.Mapping = collections.abc.Mapping

# import cv2
# import torch
# import numpy as np
# from fastreid.config import get_cfg
# from fastreid.engine import DefaultPredictor
# from fastreid.data.transforms import build_transforms
# from PIL import Image
# from ultralytics import YOLO

# # YOLOv5 Nano 모델 로드
# model = YOLO("yolov5n.pt")

# # BotSORT 설정 파일 경로
# tracker_config = "/home/jinjuuk/.local/lib/python3.10/site-packages/ultralytics/cfg/trackers/botsort.yaml"

# # FastReID 설정
# cfg = get_cfg()
# cfg.merge_from_file("/home/jinjuuk/fast-reid/configs/Market1501/bagtricks_R50.yml")
# cfg.MODEL.WEIGHTS = "/home/jinjuuk/dev_ws/pt_files/market_mgn_R50-ibn.pth"
# cfg.MODEL.DEVICE = "cpu"  # GPU를 사용할 경우 "cuda"로 변경

# predictor = DefaultPredictor(cfg)
# transform = build_transforms(cfg, is_train=False)

# # 알려진 사람들의 특징 벡터와 ID를 저장할 딕셔너리
# known_persons = {}

# # 유사도 임계값 설정
# SIMILARITY_THRESHOLD = 0.7

# def calculate_histogram(image):
#     """히스토그램을 계산하여 색상 정보를 추출합니다."""
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
#     cv2.normalize(hist, hist)
#     return hist.flatten()

# def calculate_similarity(feature1, feature2, hist1, hist2):
#     """FastReID 특징 벡터와 히스토그램을 결합한 유사도를 계산합니다."""
#     # FastReID 특징 벡터 유사도 계산 (코사인 유사도)
#     cosine_similarity = np.dot(feature1, feature2.T) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))
    
#     # 히스토그램 유사도 계산 (상관관계 방법)
#     hist_similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
#     # 두 유사도를 결합하여 최종 유사도 계산
#     combined_similarity = 0.5 * cosine_similarity + 0.5 * hist_similarity
#     return combined_similarity

# # 웹캡으로부터 비디오 캡처
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # YOLO로 객체 감지 및 BotSORT로 추적
#     results = model.track(classes=[0],            # Track only the "person" class
#                           max_det=5,              # Detect up to 5 objects
#                           show=False,             # YOLO 결과를 별도로 보여주지 않음
#                           source=frame,           # 현재 프레임을 소스로 사용
#                           tracker=tracker_config, # Use BotSORT tracker for Re-ID
#                           stream=True             # Stream results in real-time
#                           )
    
#     for result in results:
#         # 바운딩 박스와 트래킹 ID 추출
#         boxes = result.boxes.xyxy.cpu().numpy().astype(int) if result.boxes is not None else []
#         track_ids = result.boxes.id.cpu().numpy().astype(int) if result.boxes.id is not None else []

#         for box, track_id in zip(boxes, track_ids):
#             x1, y1, x2, y2 = box
#             person_img = frame[y1:y2, x1:x2]

#             # 1. FastReID를 사용하여 특징 벡터 추출
#             pil_img = Image.fromarray(cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB))
#             input_img = transform(pil_img).unsqueeze(0)
#             features = predictor(input_img).cpu().numpy()

#             # 2. 옷 색상 히스토그램 계산
#             hist = calculate_histogram(person_img)

#             # 3. 가장 유사한 사람 찾기
#             max_similarity = -1
#             max_similarity_id = -1

#             for known_id, (known_features, known_hist) in known_persons.items():
#                 similarity = calculate_similarity(features, known_features, hist, known_hist)
#                 if similarity > max_similarity:
#                     max_similarity = similarity
#                     max_similarity_id = known_id

#             # 4. 유사도 임계값 적용 및 ID 부여
#             if max_similarity > SIMILARITY_THRESHOLD:
#                 person_id = max_similarity_id
#             else:
#                 person_id = int(track_id)
#                 known_persons[person_id] = (features, hist)

#             # 5. 바운딩 박스와 레이블 그리기
#             color = (0, 255, 0)  # 초록색
#             cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#             label = f"ID: {person_id}"
#             cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

#     # 결과 프레임 출력
#     cv2.imshow("Real-time Tracking and Re-ID", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()





# import collections
# if not hasattr(collections, 'Mapping'):
#     import collections.abc
#     collections.Mapping = collections.abc.Mapping

# import cv2
# import torch
# import numpy as np
# from fastreid.config import get_cfg
# from fastreid.engine import DefaultPredictor
# from fastreid.data.transforms import build_transforms
# from PIL import Image
# from ultralytics import YOLO

# # YOLOv5 Nano 모델 로드
# model = YOLO("yolov5n.pt")

# # BotSORT 설정 파일 경로
# tracker_config = "/home/jinjuuk/.local/lib/python3.10/site-packages/ultralytics/cfg/trackers/botsort.yaml"

# # FastReID 설정
# cfg = get_cfg()
# cfg.merge_from_file("/home/jinjuuk/fast-reid/configs/Market1501/bagtricks_R50.yml")
# cfg.MODEL.WEIGHTS = "/home/jinjuuk/dev_ws/pt_files/market_mgn_R50-ibn.pth"
# cfg.MODEL.DEVICE = "cpu"  # GPU를 사용할 경우 "cuda"로 변경

# predictor = DefaultPredictor(cfg)
# transform = build_transforms(cfg, is_train=False)

# # 알려진 사람들의 특징 벡터와 ID를 저장할 딕셔너리
# known_persons = {}

# # 유사도 임계값 설정
# SIMILARITY_THRESHOLD = 0.7

# def extract_global_features(image):
#     """글로벌 피처를 추출합니다."""
#     pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     input_img = transform(pil_img).unsqueeze(0)
#     features = predictor(input_img).cpu().numpy()
#     return features

# def extract_local_features(image):
#     """지역 피처를 추출합니다. 이미지의 상하, 좌우로 나누어 피처를 추출합니다."""
#     height, width, _ = image.shape
#     segments = [
#         image[:height//2, :],  # 상체
#         image[height//2:, :],  # 하체
#         image[:, :width//2],   # 좌측
#         image[:, width//2:]    # 우측
#     ]
    
#     local_features = []
#     for segment in segments:
#         pil_img = Image.fromarray(cv2.cvtColor(segment, cv2.COLOR_BGR2RGB))
#         input_img = transform(pil_img).unsqueeze(0)
#         features = predictor(input_img).cpu().numpy()
#         local_features.append(features)
    
#     return np.concatenate(local_features, axis=1)  # 각 지역 피처를 결합합니다.

# def calculate_combined_similarity(global_features1, global_features2, local_features1, local_features2):
#     """글로벌 피처와 지역 피처를 결합하여 유사도를 계산합니다."""
#     global_similarity = np.dot(global_features1, global_features2.T) / (np.linalg.norm(global_features1) * np.linalg.norm(global_features2))
#     local_similarity = np.dot(local_features1, local_features2.T) / (np.linalg.norm(local_features1) * np.linalg.norm(local_features2))
    
#     combined_similarity = 0.5 * global_similarity + 0.5 * local_similarity  # 글로벌과 지역 피처를 50:50 비율로 결합
#     return combined_similarity

# # 웹캡으로부터 비디오 캡처
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # YOLO로 객체 감지 및 BotSORT로 추적
#     results = model.track(classes=[0],            # Track only the "person" class
#                           max_det=5,              # Detect up to 5 objects
#                           show=False,             # YOLO 결과를 별도로 보여주지 않음
#                           source=frame,           # 현재 프레임을 소스로 사용
#                           tracker=tracker_config, # Use BotSORT tracker for Re-ID
#                           stream=True             # Stream results in real-time
#                           )
    
#     for result in results:
#         # 바운딩 박스와 트래킹 ID 추출
#         boxes = result.boxes.xyxy.cpu().numpy().astype(int) if result.boxes is not None else []
#         track_ids = result.boxes.id.cpu().numpy().astype(int) if result.boxes.id is not None else []

#         for box, track_id in zip(boxes, track_ids):
#             x1, y1, x2, y2 = box
#             person_img = frame[y1:y2, x1:x2]

#             # 1. 글로벌 피처 추출
#             global_features = extract_global_features(person_img)

#             # 2. 지역 피처 추출
#             local_features = extract_local_features(person_img)

#             # 3. 가장 유사한 사람 찾기
#             max_similarity = -1
#             max_similarity_id = -1

#             for known_id, (known_global_features, known_local_features) in known_persons.items():
#                 similarity = calculate_combined_similarity(global_features, known_global_features, local_features, known_local_features)
#                 print(f"Known_Id: {known_id},  Calculated similarity:  {similarity}")  # 유사도 값 디버깅 출력
#                 if similarity > max_similarity:
#                     max_similarity = similarity
#                     max_similarity_id = known_id

#             # 4. 유사도 임계값 적용 및 ID 부여
#             if max_similarity > SIMILARITY_THRESHOLD:
#                 person_id = max_similarity_id
#             else:
#                 person_id = int(track_id)
#                 known_persons[person_id] = (global_features, local_features)

#             # 5. 바운딩 박스와 레이블 그리기
#             color = (0, 255, 0)  # 초록색
#             cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#             label = f"ID: {person_id}"
#             cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

#     # 결과 프레임 출력
#     cv2.imshow("Real-time Tracking and Re-ID", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


  


# import collections
# if not hasattr(collections, 'Mapping'):
#     import collections.abc
#     collections.Mapping = collections.abc.Mapping

# import cv2
# import torch
# import numpy as np
# from fastreid.config import get_cfg
# from fastreid.engine import DefaultPredictor
# from fastreid.data.transforms import build_transforms
# from PIL import Image
# from ultralytics import YOLO

# # YOLOv5 Nano 모델 로드
# model = YOLO("yolov5n.pt")

# # BotSORT 설정 파일 경로
# tracker_config = "/home/jinjuuk/.local/lib/python3.10/site-packages/ultralytics/cfg/trackers/botsort.yaml"

# # FastReID 설정
# cfg = get_cfg()
# cfg.merge_from_file("/home/jinjuuk/fast-reid/configs/Market1501/bagtricks_R50.yml")
# cfg.MODEL.WEIGHTS = "/home/jinjuuk/dev_ws/pt_files/market_mgn_R50-ibn.pth"
# cfg.MODEL.DEVICE = "cpu"  # GPU를 사용할 경우 "cuda"로 변경

# predictor = DefaultPredictor(cfg)
# transform = build_transforms(cfg, is_train=False)

# # 알려진 사람들의 특징 벡터와 ID를 저장할 딕셔너리
# known_persons = {}

# # 유사도 임계값 설정
# SIMILARITY_THRESHOLD = 0.95

# def extract_global_features(image):
#     """글로벌 피처를 추출합니다."""
#     pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     input_img = transform(pil_img).unsqueeze(0)
#     features = predictor(input_img).cpu().numpy()
#     return features

# def calculate_histogram(image):
#     """히스토그램을 계산하여 색상 정보를 추출합니다."""
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
#     cv2.normalize(hist, hist)
#     return hist.flatten()

# def extract_local_features(image):
#     """지역 피처를 추출합니다. 이미지의 상하, 좌우로 나누어 피처와 히스토그램을 추출합니다."""
#     height, width, _ = image.shape
#     segments = [
#         image[:height//2, :],  # 상체
#         image[height//2:, :],  # 하체
#         image[:, :width//2],   # 좌측
#         image[:, width//2:]    # 우측
#     ]
    
#     local_features = []
#     local_histograms = []
#     for segment in segments:
#         pil_img = Image.fromarray(cv2.cvtColor(segment, cv2.COLOR_BGR2RGB))
#         input_img = transform(pil_img).unsqueeze(0)
#         features = predictor(input_img).cpu().numpy()
#         hist = calculate_histogram(segment)
#         local_features.append(features)
#         local_histograms.append(hist)
    
#     combined_local_features = np.concatenate(local_features, axis=1)  # 각 지역 피처를 결합합니다.
#     return combined_local_features, local_histograms

# def calculate_combined_similarity(global_features1, global_features2, local_features1, local_features2, histograms1, histograms2):
#     """글로벌 피처, 지역 피처, 로컬 히스토그램을 결합하여 유사도를 계산합니다."""
#     # 글로벌 피처 유사도 계산
#     global_similarity = np.dot(global_features1, global_features2.T) / (np.linalg.norm(global_features1) * np.linalg.norm(global_features2))
    
#     # 로컬 피처 유사도 계산
#     local_similarity = np.dot(local_features1, local_features2.T) / (np.linalg.norm(local_features1) * np.linalg.norm(local_features2))
    
#     # 로컬 히스토그램 유사도 계산 (상관관계 방식)
#     hist_similarity = 0
#     for hist1, hist2 in zip(histograms1, histograms2):
#         hist_similarity += cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
#     hist_similarity /= len(histograms1)  # 평균 히스토그램 유사도
    
#     # 최종 유사도 계산
#     combined_similarity = 0.4 * global_similarity + 0.4 * local_similarity + 0.2 * hist_similarity
#     return combined_similarity

# # 웹캡으로부터 비디오 캡처
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # YOLO로 객체 감지 및 BotSORT로 추적
#     results = model.track(classes=[0],            # Track only the "person" class
#                           max_det=5,              # Detect up to 5 objects
#                           show=False,             # YOLO 결과를 별도로 보여주지 않음
#                           source=frame,           # 현재 프레임을 소스로 사용
#                           tracker=tracker_config, # Use BotSORT tracker for Re-ID
#                           stream=True             # Stream results in real-time
#                           )
    
#     for result in results:
#         # 바운딩 박스와 트래킹 ID 추출
#         boxes = result.boxes.xyxy.cpu().numpy().astype(int) if result.boxes is not None else []
#         track_ids = result.boxes.id.cpu().numpy().astype(int) if result.boxes.id is not None else []

#         for box, track_id in zip(boxes, track_ids):
#             x1, y1, x2, y2 = box
#             person_img = frame[y1:y2, x1:x2]

#             # 1. 글로벌 피처 추출
#             global_features = extract_global_features(person_img)

#             # 2. 지역 피처 및 히스토그램 추출
#             local_features, local_histograms = extract_local_features(person_img)

#             # 3. 가장 유사한 사람 찾기
#             max_similarity = -1
#             max_similarity_id = -1

#             for known_id, (known_global_features, known_local_features, known_histograms) in known_persons.items():
#                 similarity = calculate_combined_similarity(global_features, known_global_features, local_features, known_local_features, local_histograms, known_histograms)
#                 print(f"Known_Id: {known_id},  Calculated similarity:  {similarity}")  # 유사도 값 디버깅 출력
                
#                 if similarity > max_similarity:
#                     max_similarity = similarity
#                     max_similarity_id = known_id

#             # 4. 유사도 임계값 적용 및 ID 부여
#             if max_similarity > SIMILARITY_THRESHOLD:
#                 person_id = max_similarity_id
#             else:
#                 person_id = int(track_id)
#                 known_persons[person_id] = (global_features, local_features, local_histograms)

#             # 5. 바운딩 박스와 레이블 그리기
#             color = (0, 255, 0)  # 초록색
#             cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#             label = f"ID: {person_id}"
#             cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

#     # 결과 프레임 출력
#     cv2.imshow("Real-time Tracking and Re-ID", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()






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
from ultralytics import YOLO

# YOLOv5 Nano 모델 로드
model = YOLO("yolov5n.pt")

# BotSORT 설정 파일 경로
tracker_config = "/home/jinjuuk/.local/lib/python3.10/site-packages/ultralytics/cfg/trackers/botsort.yaml"

# FastReID 설정
cfg = get_cfg()
cfg.merge_from_file("/home/jinjuuk/fast-reid/configs/Market1501/bagtricks_R50.yml")
cfg.MODEL.WEIGHTS = "/home/jinjuuk/dev_ws/pt_files/market_mgn_R50-ibn.pth"
cfg.MODEL.DEVICE = "cpu"  # GPU를 사용할 경우 "cuda"로 변경

predictor = DefaultPredictor(cfg)
transform = build_transforms(cfg, is_train=False)

# 알려진 사람들의 특징 벡터와 ID를 저장할 딕셔너리
known_persons = {}

# 유사도 임계값 설정
SIMILARITY_THRESHOLD = 0.95

def extract_global_features(image):
    """글로벌 피처를 추출합니다."""
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    input_img = transform(pil_img).unsqueeze(0)
    features = predictor(input_img).cpu().numpy()
    return features

def calculate_histogram(image):
    """히스토그램을 계산하여 색상 정보를 추출합니다."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def extract_local_features(image):
    """지역 피처를 추출합니다. 이미지의 상하, 좌우로 나누어 피처와 히스토그램을 추출합니다."""
    height, width, _ = image.shape
    segments = [
        image[:height//2, :],  # 상체
        image[height//2:, :],  # 하체
        image[:, :width//2],   # 좌측
        image[:, width//2:]    # 우측
    ]
    
    local_features = []
    local_histograms = []
    for segment in segments:
        pil_img = Image.fromarray(cv2.cvtColor(segment, cv2.COLOR_BGR2RGB))
        input_img = transform(pil_img).unsqueeze(0)
        features = predictor(input_img).cpu().numpy()
        hist = calculate_histogram(segment)
        local_features.append(features)
        local_histograms.append(hist)
    
    combined_local_features = np.concatenate(local_features, axis=1)  # 각 지역 피처를 결합합니다.
    return combined_local_features, local_histograms

def calculate_combined_similarity(global_features1, global_features2, local_features1, local_features2, histograms1, histograms2):
    """글로벌 피처, 지역 피처, 로컬 히스토그램을 결합하여 유사도를 계산합니다."""
    # 글로벌 피처 유사도 계산
    global_similarity = np.dot(global_features1, global_features2.T) / (np.linalg.norm(global_features1) * np.linalg.norm(global_features2))
    
    # 로컬 피처 유사도 계산
    local_similarity = np.dot(local_features1, local_features2.T) / (np.linalg.norm(local_features1) * np.linalg.norm(local_features2))
    
    # 로컬 히스토그램 유사도 계산 (상관관계 방식)
    hist_similarity = 0
    for hist1, hist2 in zip(histograms1, histograms2):
        hist_similarity += cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    hist_similarity /= len(histograms1)  # 평균 히스토그램 유사도
    
    # 최종 유사도 계산
    combined_similarity = 0.4 * global_similarity + 0.4 * local_similarity + 0.2 * hist_similarity
    return combined_similarity

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

            # 1. 글로벌 피처 추출
            global_features = extract_global_features(person_img)

            # 2. 지역 피처 및 히스토그램 추출
            local_features, local_histograms = extract_local_features(person_img)

            # 3. 가장 유사한 사람 찾기
            max_similarity = -1
            max_similarity_id = -1

            for known_id, (known_global_features, known_local_features, known_histograms) in known_persons.items():
                similarity = calculate_combined_similarity(global_features, known_global_features, local_features, known_local_features, local_histograms, known_histograms)
                print(f"Known_Id: {known_id},  Calculated similarity:  {similarity}")  # 유사도 값 디버깅 출력
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    max_similarity_id = known_id

            # 4. 유사도 임계값 적용 및 ID 부여
            if max_similarity > SIMILARITY_THRESHOLD:
                person_id = max_similarity_id
                color = (0, 255, 0)  # 기존 ID의 경우 초록색 사용
            else:
                person_id = int(track_id)
                known_persons[person_id] = (global_features, local_features, local_histograms)
                color = (0, 0, 255)  # 새로운 ID의 경우 빨간색 사용

            # 5. 바운딩 박스와 레이블 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID: {person_id}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # 결과 프레임 출력
    cv2.imshow("Real-time Tracking and Re-ID", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
