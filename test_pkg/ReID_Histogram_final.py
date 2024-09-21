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
# SIMILARITY_THRESHOLD = 0.99

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
#                 color = (0, 255, 0)  # 기존 ID의 경우 초록색 사용
#             else:
#                 person_id = int(track_id)
#                 known_persons[person_id] = (global_features, local_features, local_histograms)
#                 color = (0, 0, 255)  # 새로운 ID의 경우 빨간색 사용

#             # 바운딩 박스의 중앙값 계산 및 출력
#             center_x = (x1 + x2) // 2
#             center_y = (y1 + y2) // 2
#             print(f"ID: {person_id}, Center: ({center_x}, {center_y})")
#             centp = f"Center: ({center_x}, {center_y})"
#             cv2.putText(frame, centp, (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

#             # 5. 바운딩 박스와 레이블 그리기
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
# SIMILARITY_THRESHOLD = 0.99

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
#     combined_similarity = 0.3 * global_similarity + 0.4 * local_similarity + 0.3 * hist_similarity
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
#                 color = (0, 255, 0)  # 기존 ID의 경우 초록색 사용
#             else:
#                 person_id = int(track_id)
#                 known_persons[person_id] = (global_features, local_features, local_histograms)
#                 color = (0, 0, 255)  # 새로운 ID의 경우 빨간색 사용

#             # 바운딩 박스의 중앙값 계산 및 출력
#             center_x = (x1 + x2) // 2
#             center_y = (y1 + y2) // 2
#             print(f"ID: {person_id}, Center: ({center_x}, {center_y})")
#             centp = f"Center: ({center_x}, {center_y})"
#             text_x = x2 + 10  # 바운딩 박스 오른쪽에 출력
#             text_y = y2  # 바운딩 박스 하단에 맞춰 출력
#             cv2.putText(frame, centp, (text_x, text_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

#             # 5. 바운딩 박스와 레이블 그리기
#             cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#             label = f"ID: {person_id}"
#             cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

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
#     for i, segment in enumerate(segments):
#         pil_img = Image.fromarray(cv2.cvtColor(segment, cv2.COLOR_BGR2RGB))
#         input_img = transform(pil_img).unsqueeze(0)
#         features = predictor(input_img).cpu().numpy()
#         hist = calculate_histogram(segment)
#         local_features.append(features)
#         local_histograms.append(hist)
    
#     # 상반신 피처와 하반신 피처를 분리하여 리턴
#     upper_body_features = local_features[0]  # 상체 피처
#     lower_body_features = local_features[1]  # 하체 피처
    
#     upper_body_histogram = local_histograms[0]  # 상체 히스토그램
#     lower_body_histogram = local_histograms[1]  # 하체 히스토그램

#     return upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram

# def calculate_combined_similarity(global_features1, global_features2, upper_body_features1, upper_body_features2, lower_body_features1, lower_body_features2, upper_body_histogram1, upper_body_histogram2, lower_body_histogram1, lower_body_histogram2):
#     """글로벌 피처, 지역 피처, 로컬 히스토그램을 결합하여 유사도를 계산합니다."""
#     # 글로벌 피처 유사도 계산
#     global_similarity = np.dot(global_features1, global_features2.T) / (np.linalg.norm(global_features1) * np.linalg.norm(global_features2))
    
#     # 상반신 로컬 피처 유사도 계산
#     upper_body_similarity = np.dot(upper_body_features1, upper_body_features2.T) / (np.linalg.norm(upper_body_features1) * np.linalg.norm(upper_body_features2))
    
#     # 하반신 로컬 피처 유사도 계산
#     lower_body_similarity = np.dot(lower_body_features1, lower_body_features2.T) / (np.linalg.norm(lower_body_features1) * np.linalg.norm(lower_body_features2))
    
#     # 상반신 히스토그램 유사도 계산
#     upper_body_hist_similarity = cv2.compareHist(upper_body_histogram1, upper_body_histogram2, cv2.HISTCMP_CORREL)
    
#     # 하반신 히스토그램 유사도 계산
#     lower_body_hist_similarity = cv2.compareHist(lower_body_histogram1, lower_body_histogram2, cv2.HISTCMP_CORREL)
    
#     # 최종 유사도 계산 0.2, 0.1, 0.4, 0.3
#     combined_similarity = (0.3 * global_similarity +  
#                            0.1 * upper_body_similarity +
#                            0.3 * lower_body_similarity +
#                            0.3 * lower_body_hist_similarity)
    
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
#             upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram = extract_local_features(person_img)

#             # 3. 가장 유사한 사람 찾기
#             max_similarity = -1
#             max_similarity_id = -1

#             for known_id, (known_global_features, known_upper_body_features, known_lower_body_features, known_upper_body_histogram, known_lower_body_histogram) in known_persons.items():
#                 similarity = calculate_combined_similarity(global_features, known_global_features, upper_body_features, known_upper_body_features, lower_body_features, known_lower_body_features, upper_body_histogram, known_upper_body_histogram, lower_body_histogram, known_lower_body_histogram)
#                 print(f"Known_Id: {known_id},  Calculated similarity:  {similarity}")  # 유사도 값 디버깅 출력
                
#                 if similarity > max_similarity:
#                     max_similarity = similarity
#                     max_similarity_id = known_id

#             # 4. 유사도 임계값 적용 및 ID 부여
#             if max_similarity > SIMILARITY_THRESHOLD:
#                 person_id = max_similarity_id
#                 color = (0, 255, 0)  # 기존 ID의 경우 초록색 사용
#             else:
#                 person_id = int(track_id)
#                 known_persons[person_id] = (global_features, upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram)
#                 color = (0, 0, 255)  # 새로운 ID의 경우 빨간색 사용

#             # 바운딩 박스의 중앙값 계산 및 출력
#             center_x = (x1 + x2) // 2
#             center_y = (y1 + y2) // 2
#             print(f"ID: {person_id}, Center: ({center_x}, {center_y})")
#             centp = f"Center: ({center_x}, {center_y})"
#             text_x = x2 + 10  # 바운딩 박스 오른쪽에 출력
#             text_y = y2  # 바운딩 박스 하단에 맞춰 출력
#             cv2.putText(frame, centp, (text_x, text_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

#             # 5. 바운딩 박스와 레이블 그리기
#             cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#             label = f"ID: {person_id}"
#             cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

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
# import time

# # YOLOv5 Nano 모델 로드
# model = YOLO("yolov5n.pt")

# # BotSORT 설정 파일 경로
# tracker_config = "/home/jinjuuk/.local/lib/python3.10/site-packages/ultralytics/cfg/trackers/botsort.yaml"

# # FastReID 설정
# cfg = get_cfg()
# cfg.merge_from_file("/home/jinjuuk/fast-reid/configs/Market1501/bagtricks_R50.yml")
# cfg.MODEL.WEIGHTS = "/home/jinjuuk/dev_ws/pt_files/market_mgn_R50-ibn.pth"
# cfg.MODEL.DEVICE = "cuda"  # GPU를 사용할 경우 "cuda"로 변경

# predictor = DefaultPredictor(cfg)
# transform = build_transforms(cfg, is_train=False)

# # 알려진 사람들의 특징 벡터와 ID를 저장할 딕셔너리
# known_persons = {}
# # 기존의 딕셔너리 대신, 사람의 마지막 인식 시간을 저장하는 딕셔너리 추가
# id_last_seen = {}  # {person_id: last_seen_timestamp}
# ID_HOLD_TIME = 10  # ID 유지 시간 (초)

# # 유사도 임계값 설정
# SIMILARITY_THRESHOLD = 0.97 #0.95

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
#     for i, segment in enumerate(segments):
#         pil_img = Image.fromarray(cv2.cvtColor(segment, cv2.COLOR_BGR2RGB))
#         input_img = transform(pil_img).unsqueeze(0)
#         features = predictor(input_img).cpu().numpy()
#         hist = calculate_histogram(segment)
#         local_features.append(features)
#         local_histograms.append(hist)
    
#     # 상반신 피처와 하반신 피처를 분리하여 리턴
#     upper_body_features = local_features[0]  # 상체 피처
#     lower_body_features = local_features[1]  # 하체 피처
    
#     upper_body_histogram = local_histograms[0]  # 상체 히스토그램
#     lower_body_histogram = local_histograms[1]  # 하체 히스토그램

#     return upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram

# def calculate_combined_similarity(global_features1, global_features2, upper_body_features1, upper_body_features2, lower_body_features1, lower_body_features2, upper_body_histogram1, upper_body_histogram2, lower_body_histogram1, lower_body_histogram2):
#     """글로벌 피처, 지역 피처, 로컬 히스토그램을 결합하여 유사도를 계산합니다."""
#     # 글로벌 피처 유사도 계산
#     global_similarity = np.dot(global_features1, global_features2.T) / (np.linalg.norm(global_features1) * np.linalg.norm(global_features2))
    
#     # 상반신 로컬 피처 유사도 계산
#     upper_body_similarity = np.dot(upper_body_features1, upper_body_features2.T) / (np.linalg.norm(upper_body_features1) * np.linalg.norm(upper_body_features2))
    
#     # 하반신 로컬 피처 유사도 계산
#     lower_body_similarity = np.dot(lower_body_features1, lower_body_features2.T) / (np.linalg.norm(lower_body_features1) * np.linalg.norm(lower_body_features2))
    
#     # 상반신 히스토그램 유사도 계산
#     upper_body_hist_similarity = cv2.compareHist(upper_body_histogram1, upper_body_histogram2, cv2.HISTCMP_CORREL)
    
#     # 하반신 히스토그램 유사도 계산
#     lower_body_hist_similarity = cv2.compareHist(lower_body_histogram1, lower_body_histogram2, cv2.HISTCMP_CORREL)
    
#     # 최종 유사도 계산 0.2, 0.1, 0.4, 0.3
#     combined_similarity = (0.3 * global_similarity +  
#                            0.1 * upper_body_similarity +
#                            0.3 * lower_body_similarity +
#                            0.3 * lower_body_hist_similarity)
    
#     return combined_similarity


# # 웹캡으로부터 비디오 캡처
# cap = cv2.VideoCapture(0)


# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # 현재 시간 기록
#     current_time = time.time()

#     # YOLO로 객체 감지 및 BotSORT로 추적
#     results = model.track(classes=[0],            # Track only the "person" class
#                           max_det=5,              # Detect up to 5 objects
#                           show=False,             # YOLO 결과를 별도로 보여주지 않음
#                           source=frame,           # 현재 프레임을 소스로 사용
#                           tracker=tracker_config, # Use BotSORT tracker for Re-ID
#                           stream=True             # Stream results in real-time
#                           )

#     # 추적된 사람들의 ID가 중복되지 않도록 보장하기 위한 딕셔너리
#     active_ids = set()

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
#             upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram = extract_local_features(person_img)

#             # 3. 가장 유사한 사람 찾기
#             max_similarity = -1
#             max_similarity_id = -1

#             for known_id, (known_global_features, known_upper_body_features, known_lower_body_features, known_upper_body_histogram, known_lower_body_histogram) in known_persons.items():
#                 similarity = calculate_combined_similarity(global_features, known_global_features, upper_body_features, known_upper_body_features, lower_body_features, known_lower_body_features, upper_body_histogram, known_upper_body_histogram, lower_body_histogram, known_lower_body_histogram)
#                 print(f"Known_Id: {known_id},  Calculated similarity:  {similarity}")  # 유사도 값 디버깅 출력

#                 if similarity > max_similarity:
#                     max_similarity = similarity
#                     max_similarity_id = known_id

#             # 4. 유사도 임계값 적용 및 ID 부여
#             if max_similarity > SIMILARITY_THRESHOLD:
#                 if max_similarity_id not in active_ids:
#                     person_id = max_similarity_id
#                     color = (0, 255, 0)  # 기존 ID의 경우 초록색 사용
#             else:
#                 # 완전 새로운 객체인 경우
#                 person_id = int(track_id)
#                 if person_id in known_persons:
#                     # 만약 새롭게 할당된 ID가 이미 존재하는 경우, 기존 ID 유지
#                     pass
#                 else:
#                     # 새로운 ID를 부여
#                     new_id = max(known_persons.keys(), default=0) + 1
#                     person_id = new_id
#                 known_persons[person_id] = (global_features, upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram)
#                 color = (0, 0, 255)  # 새로운 ID의 경우 빨간색 사용

#             # 마지막으로 본 시간 업데이트
#             id_last_seen[person_id] = current_time
#             active_ids.add(person_id)

#             # 바운딩 박스의 중앙값 계산 및 출력
#             center_x = (x1 + x2) // 2
#             center_y = (y1 + y2) // 2
#             print(f"ID: {person_id}, Center: ({center_x}, {center_y})")
#             centp = f"Center: ({center_x}, {center_y})"
#             text_x = x2 + 10  # 바운딩 박스 오른쪽에 출력
#             text_y = y2  # 바운딩 박스 하단에 맞춰 출력
#             cv2.putText(frame, centp, (text_x, text_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

#             # 5. 바운딩 박스와 레이블 그리기
#             cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#             label = f"ID: {person_id}"
#             cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

#     # 기존 ID들 중 오래된 ID 제거 (5초 이상 프레임에 보이지 않으면 제거)
#     for person_id in list(id_last_seen.keys()):
#         if current_time - id_last_seen[person_id] > ID_HOLD_TIME:
#             del known_persons[person_id]
#             del id_last_seen[person_id]

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
# import time

# # YOLOv5 Nano 모델 로드
# model = YOLO("yolov5n.pt")

# # BotSORT 설정 파일 경로
# tracker_config = "/home/jinjuuk/.local/lib/python3.10/site-packages/ultralytics/cfg/trackers/botsort.yaml"

# # FastReID 설정
# cfg = get_cfg()
# cfg.merge_from_file("/home/jinjuuk/fast-reid/configs/Market1501/bagtricks_R50.yml")
# cfg.MODEL.WEIGHTS = "/home/jinjuuk/dev_ws/pt_files/market_mgn_R50-ibn.pth"
# cfg.MODEL.DEVICE = "cuda"  # GPU를 사용할 경우 "cuda"로 변경

# predictor = DefaultPredictor(cfg)
# transform = build_transforms(cfg, is_train=False)

# # 알려진 사람들의 특징 벡터와 ID를 저장할 딕셔너리
# known_persons = {}
# # 기존의 딕셔너리 대신, 사람의 마지막 인식 시간을 저장하는 딕셔너리 추가
# id_last_seen = {}

# # 초기 설정값
# SIMILARITY_THRESHOLD = 0.97  # 초기 유사도 임계값
# ID_HOLD_TIME = 10  # 초기 ID 유지 시간 (초)

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
#     for i, segment in enumerate(segments):
#         pil_img = Image.fromarray(cv2.cvtColor(segment, cv2.COLOR_BGR2RGB))
#         input_img = transform(pil_img).unsqueeze(0)
#         features = predictor(input_img).cpu().numpy()
#         hist = calculate_histogram(segment)
#         local_features.append(features)
#         local_histograms.append(hist)
    
#     # 상반신 피처와 하반신 피처를 분리하여 리턴
#     upper_body_features = local_features[0]  # 상체 피처
#     lower_body_features = local_features[1]  # 하체 피처
    
#     upper_body_histogram = local_histograms[0]  # 상체 히스토그램
#     lower_body_histogram = local_histograms[1]  # 하체 히스토그램

#     return upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram

# def calculate_combined_similarity(global_features1, global_features2, upper_body_features1, upper_body_features2, lower_body_features1, lower_body_features2, upper_body_histogram1, upper_body_histogram2, lower_body_histogram1, lower_body_histogram2):
#     """글로벌 피처, 지역 피처, 로컬 히스토그램을 결합하여 유사도를 계산합니다."""
#     # 글로벌 피처 유사도 계산
#     global_similarity = np.dot(global_features1, global_features2.T) / (np.linalg.norm(global_features1) * np.linalg.norm(global_features2))
    
#     # 상반신 로컬 피처 유사도 계산
#     upper_body_similarity = np.dot(upper_body_features1, upper_body_features2.T) / (np.linalg.norm(upper_body_features1) * np.linalg.norm(upper_body_features2))
    
#     # 하반신 로컬 피처 유사도 계산
#     lower_body_similarity = np.dot(lower_body_features1, lower_body_features2.T) / (np.linalg.norm(lower_body_features1) * np.linalg.norm(lower_body_features2))
    
#     # 상반신 히스토그램 유사도 계산
#     upper_body_hist_similarity = cv2.compareHist(upper_body_histogram1, upper_body_histogram2, cv2.HISTCMP_CORREL)
    
#     # 하반신 히스토그램 유사도 계산
#     lower_body_hist_similarity = cv2.compareHist(lower_body_histogram1, lower_body_histogram2, cv2.HISTCMP_CORREL)
    
#     # 최종 유사도 계산
#     combined_similarity = (0.3 * global_similarity +  
#                            0.1 * upper_body_similarity +
#                            0.3 * lower_body_similarity +
#                            0.3 * lower_body_hist_similarity)
    
#     return combined_similarity


# # 슬라이더 콜백 함수
# def update_threshold(val):
#     global SIMILARITY_THRESHOLD
#     SIMILARITY_THRESHOLD = val / 100.0  # 슬라이더 값이 0~100 범위이므로 100으로 나눔

# def update_hold_time(val):
#     global ID_HOLD_TIME
#     ID_HOLD_TIME = val  # 슬라이더 값이 곧 ID_HOLD_TIME

# # 웹캡으로부터 비디오 캡처
# cap = cv2.VideoCapture(0)

# # 창과 슬라이더 생성
# cv2.namedWindow("Real-time Tracking and Re-ID")
# cv2.createTrackbar("Similarity Threshold", "Real-time Tracking and Re-ID", int(SIMILARITY_THRESHOLD * 100), 100, update_threshold)
# cv2.createTrackbar("ID Hold Time", "Real-time Tracking and Re-ID", ID_HOLD_TIME, 30, update_hold_time)  # ID_HOLD_TIME의 범위는 0~30초

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # 현재 시간 기록
#     current_time = time.time()

#     # YOLO로 객체 감지 및 BotSORT로 추적
#     results = model.track(classes=[0],            # Track only the "person" class
#                           max_det=3,              # Detect up to 5 objects
#                           show=False,             # YOLO 결과를 별도로 보여주지 않음
#                           source=frame,           # 현재 프레임을 소스로 사용
#                           tracker=tracker_config, # Use BotSORT tracker for Re-ID
#                           stream=True             # Stream results in real-time
#                           )

#     # 추적된 사람들의 ID가 중복되지 않도록 보장하기 위한 딕셔너리
#     active_ids = set()

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
#             upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram = extract_local_features(person_img)

#             # 3. 가장 유사한 사람 찾기
#             max_similarity = -1
#             max_similarity_id = -1

#             for known_id, (known_global_features, known_upper_body_features, known_lower_body_features, known_upper_body_histogram, known_lower_body_histogram) in known_persons.items():
#                 similarity = calculate_combined_similarity(global_features, known_global_features, upper_body_features, known_upper_body_features, lower_body_features, known_lower_body_features, upper_body_histogram, known_upper_body_histogram, lower_body_histogram, known_lower_body_histogram)
#                 print(f"Known_Id: {known_id},  Calculated similarity:  {similarity}")  # 유사도 값 디버깅 출력

#                 if similarity > max_similarity:
#                     max_similarity = similarity
#                     max_similarity_id = known_id

#             # 4. 유사도 임계값 적용 및 ID 부여
#             if max_similarity > SIMILARITY_THRESHOLD:
#                 if max_similarity_id not in active_ids:
#                     person_id = max_similarity_id
#                     color = (0, 255, 0)  # 기존 ID의 경우 초록색 사용
#             else:
#                 # 완전 새로운 객체인 경우
#                 person_id = int(track_id)
#                 if person_id in known_persons:
#                     # 만약 새롭게 할당된 ID가 이미 존재하는 경우, 기존 ID 유지
#                     pass
#                 else:
#                     # 새로운 ID를 부여
#                     new_id = max(known_persons.keys(), default=0) + 1
#                     person_id = new_id
#                 known_persons[person_id] = (global_features, upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram)
#                 color = (0, 0, 255)  # 새로운 ID의 경우 빨간색 사용

#             # 마지막으로 본 시간 업데이트
#             id_last_seen[person_id] = current_time
#             active_ids.add(person_id)

#             # 바운딩 박스의 중앙값 계산 및 출력
#             center_x = (x1 + x2) // 2
#             center_y = (y1 + y2) // 2
#             print(f"ID: {person_id}, Center: ({center_x}, {center_y})")
#             centp = f"Center: ({center_x}, {center_y})"
#             text_x = x2 + 10  # 바운딩 박스 오른쪽에 출력
#             text_y = y2  # 바운딩 박스 하단에 맞춰 출력
#             cv2.putText(frame, centp, (text_x, text_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

#             # 5. 바운딩 박스와 레이블 그리기
#             cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#             label = f"ID: {person_id}"
#             cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

#     # 기존 ID들 중 오래된 ID 제거 (ID_HOLD_TIME 초 이상 프레임에 보이지 않으면 제거)
#     for person_id in list(id_last_seen.keys()):
#         if current_time - id_last_seen[person_id] > ID_HOLD_TIME:
#             del known_persons[person_id]
#             del id_last_seen[person_id]

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
# import time

# # YOLOv5 Nano 모델 로드
# model = YOLO("yolov5su.pt")

# # BotSORT 설정 파일 경로
# tracker_config = "/home/jinjuuk/.local/lib/python3.10/site-packages/ultralytics/cfg/trackers/botsort.yaml"

# # FastReID 설정
# cfg = get_cfg()
# cfg.merge_from_file("/home/jinjuuk/fast-reid/configs/Market1501/bagtricks_R50.yml") # bagtricks_R50.yml, mgn_R50-ibn.yml, AGW_R50.yml, sbs_R50.yml
# cfg.MODEL.WEIGHTS = "/home/jinjuuk/dev_ws/pt_files/market_mgn_R50-ibn.pth" # market_bot_R50.pth, market_mgn_R50-ibn.pth, market_agw_R50.pth, market_sbs_R50.pth
# cfg.MODEL.DEVICE = "cuda"  # GPU를 사용할 경우 "cuda"로 변경

# predictor = DefaultPredictor(cfg)
# transform = build_transforms(cfg, is_train=False)

# # 알려진 사람들의 특징 벡터와 ID를 저장할 딕셔너리
# known_persons = {}
# id_last_seen = {}
# assigned_ids = set()  # A set to keep track of currently assigned IDs

# # 초기 설정값
# SIMILARITY_THRESHOLD = 0.97  # 초기 유사도 임계값 0.97
# ID_HOLD_TIME = 10  # 초기 ID 유지 시간 (초)
# YOLO_CONF_THRESHOLD = 0.80  # YOLO confidence threshold
# YOLO_IOU_THRESHOLD = 0.80  # YOLO IoU threshold


# def extract_global_features(image):
#     pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     input_img = transform(pil_img).unsqueeze(0)
#     features = predictor(input_img).cpu().numpy()
#     return features

# def calculate_histogram(image):
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
#     cv2.normalize(hist, hist)
#     return hist.flatten()

# def extract_local_features(image):
#     height, width, _ = image.shape
#     segments = [
#         image[:height//2, :],  
#         image[height//2:, :],  
#         image[:, :width//2],   
#         image[:, width//2:]    
#     ]
    
#     local_features = []
#     local_histograms = []
#     for i, segment in enumerate(segments):
#         pil_img = Image.fromarray(cv2.cvtColor(segment, cv2.COLOR_BGR2RGB))
#         input_img = transform(pil_img).unsqueeze(0)
#         features = predictor(input_img).cpu().numpy()
#         hist = calculate_histogram(segment)
#         local_features.append(features)
#         local_histograms.append(hist)
    
#     upper_body_features = local_features[0]  
#     lower_body_features = local_features[1]  
    
#     upper_body_histogram = local_histograms[0]  
#     lower_body_histogram = local_histograms[1]  

#     return upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram

# def calculate_combined_similarity(global_features1, global_features2, upper_body_features1, upper_body_features2, lower_body_features1, lower_body_features2, upper_body_histogram1, upper_body_histogram2, lower_body_histogram1, lower_body_histogram2):
#     global_similarity = np.dot(global_features1, global_features2.T) / (np.linalg.norm(global_features1) * np.linalg.norm(global_features2))
#     upper_body_similarity = np.dot(upper_body_features1, upper_body_features2.T) / (np.linalg.norm(upper_body_features1) * np.linalg.norm(upper_body_features2))
#     lower_body_similarity = np.dot(lower_body_features1, lower_body_features2.T) / (np.linalg.norm(lower_body_features1) * np.linalg.norm(lower_body_features2))
#     upper_body_hist_similarity = cv2.compareHist(upper_body_histogram1, upper_body_histogram2, cv2.HISTCMP_CORREL)
#     lower_body_hist_similarity = cv2.compareHist(lower_body_histogram1, lower_body_histogram2, cv2.HISTCMP_CORREL)
    
#     # Updated combined similarity calculation to include upper_body_hist_similarity with the new weight
#     combined_similarity = (0.3 * global_similarity +  
#                            0.25 * upper_body_similarity +
#                            0.1 * lower_body_similarity +
#                            0.1 * lower_body_hist_similarity +
#                            0.25 * upper_body_hist_similarity)
    
#     return combined_similarity

# # 슬라이더 콜백 함수
# def update_similarity_threshold(val):
#     global SIMILARITY_THRESHOLD
#     SIMILARITY_THRESHOLD = val / 100.0

# def update_hold_time(val):
#     global ID_HOLD_TIME
#     ID_HOLD_TIME = val

# def update_yolo_conf_threshold(val):
#     global YOLO_CONF_THRESHOLD
#     YOLO_CONF_THRESHOLD = val / 100.0

# def update_yolo_iou_threshold(val):
#     global YOLO_IOU_THRESHOLD
#     YOLO_IOU_THRESHOLD = val / 100.0


# # 웹캡으로부터 비디오 캡처
# cap = cv2.VideoCapture(0)

# # 창과 슬라이더 생성
# cv2.namedWindow("Real-time Tracking and Re-ID")
# cv2.createTrackbar("Similarity Threshold", "Real-time Tracking and Re-ID", int(SIMILARITY_THRESHOLD * 100), 100, update_similarity_threshold)
# cv2.createTrackbar("ID Hold Time", "Real-time Tracking and Re-ID", ID_HOLD_TIME, 30, update_hold_time)
# cv2.createTrackbar("YOLO Conf Threshold", "Real-time Tracking and Re-ID", int(YOLO_CONF_THRESHOLD * 100), 100, update_yolo_conf_threshold)
# cv2.createTrackbar("YOLO IoU Threshold", "Real-time Tracking and Re-ID", int(YOLO_IOU_THRESHOLD * 100), 100, update_yolo_iou_threshold)


# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     current_time = time.time()

#     results = model.track(classes=[0],
#                            max_det=10, 
#                            show=False, 
#                            source=frame, 
#                            tracker=tracker_config, 
#                            stream=True,
#                            conf = YOLO_CONF_THRESHOLD,
#                            iou = YOLO_IOU_THRESHOLD,)

#     active_ids = set()

#     for result in results:
#         boxes = result.boxes.xyxy.cpu().numpy().astype(int) if result.boxes is not None else []
#         track_ids = result.boxes.id.cpu().numpy().astype(int) if result.boxes.id is not None else []

#         for box, track_id in zip(boxes, track_ids):
#             x1, y1, x2, y2 = box
#             person_img = frame[y1:y2, x1:x2]

#             # Extract global and local features along with histograms
#             global_features = extract_global_features(person_img)
#             upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram = extract_local_features(person_img)

#             max_similarity = -1
#             max_similarity_id = -1

#             for known_id, (known_global_features, known_upper_body_features, known_lower_body_features, known_upper_body_histogram, known_lower_body_histogram) in known_persons.items():
#                 # Calculate combined similarity including the upper_body_hist_similarity
#                 similarity = calculate_combined_similarity(
#                     global_features,
#                     known_global_features,
#                     upper_body_features,
#                     known_upper_body_features,
#                     lower_body_features,
#                     known_lower_body_features,
#                     upper_body_histogram,
#                     known_upper_body_histogram,
#                     lower_body_histogram,
#                     known_lower_body_histogram
#                 )
#                 print(f"Known_Id: {known_id}, {max_similarity_id},  Calculated similarity:  {similarity}, {max_similarity}")

#                 if similarity > max_similarity:
#                     max_similarity = similarity
#                     max_similarity_id = known_id

#             if max_similarity > SIMILARITY_THRESHOLD:
#                 if max_similarity_id not in active_ids:
#                     person_id = max_similarity_id
#                     color = (0, 255, 0)
#             else:
#                 # Assign a new ID that hasn't been used
#                 if int(track_id) in assigned_ids:
#                     person_id = int(track_id)
#                 else:
#                     new_id = max(assigned_ids) + 1 if assigned_ids else 1
#                     person_id = new_id
#                     assigned_ids.add(person_id)
                
#                 if person_id not in known_persons:
#                     known_persons[person_id] = (global_features, upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram)
#                 color = (0, 0, 255)

#             id_last_seen[person_id] = current_time
#             active_ids.add(person_id)

#             center_x = (x1 + x2) // 2
#             center_y = (y1 + y2) // 2
#             print(f"ID: {person_id}, Center: ({center_x}, {center_y})")
#             centp = f"Center: ({center_x}, {center_y})"
#             text_x = x2 + 10
#             text_y = y2

#             cv2.putText(frame, centp, (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
#             #cv2.putText(frame, centp, (text_x, text_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

#             cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#             label = f"ID: {person_id}"
#             cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
#             #cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

#     # Clean up IDs that haven't been seen recently
#     for person_id in list(id_last_seen.keys()):
#         if current_time - id_last_seen[person_id] > ID_HOLD_TIME:
#             del known_persons[person_id]
#             del id_last_seen[person_id]
#             assigned_ids.remove(person_id)  # Free up the ID for future use

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
import time

# YOLOv5 Nano 모델 로드
model = YOLO("yolov8m.pt")

# BotSORT 설정 파일 경로
tracker_config = "/home/jinjuuk/.local/lib/python3.10/site-packages/ultralytics/cfg/trackers/botsort.yaml"

# FastReID 설정
cfg = get_cfg()
cfg.merge_from_file("/home/jinjuuk/fast-reid/configs/Market1501/bagtricks_R50.yml") # bagtricks_R50.yml, mgn_R50-ibn.yml, AGW_R50.yml, sbs_R50.yml
cfg.MODEL.WEIGHTS = "/home/jinjuuk/dev_ws/pt_files/market_mgn_R50-ibn.pth" # market_bot_R50.pth, market_mgn_R50-ibn.pth, market_agw_R50.pth, market_sbs_R50.pth
cfg.MODEL.DEVICE = "cuda"  # GPU를 사용할 경우 "cuda"로 변경

predictor = DefaultPredictor(cfg)
transform = build_transforms(cfg, is_train=False)

# 알려진 사람들의 특징 벡터와 ID를 저장할 딕셔너리
known_persons = {}
id_last_seen = {}
assigned_ids = set()  # A set to keep track of currently assigned IDs

# 초기 설정값
SIMILARITY_THRESHOLD = 0.85  # Lowered initial similarity threshold
ID_HOLD_TIME = 10  # 초기 ID 유지 시간 (초)
YOLO_CONF_THRESHOLD = 0.80  # YOLO confidence threshold
YOLO_IOU_THRESHOLD = 0.80  # YOLO IoU threshold


def extract_global_features(image):
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    input_img = transform(pil_img).unsqueeze(0)
    features = predictor(input_img).cpu().numpy()
    return features

def calculate_histogram(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def extract_local_features(image):
    height, width, _ = image.shape
    segments = [
        image[:height//2, :],  
        image[height//2:, :],  
        image[:, :width//2],   
        image[:, width//2:]    
    ]
    
    local_features = []
    local_histograms = []
    for i, segment in enumerate(segments):
        pil_img = Image.fromarray(cv2.cvtColor(segment, cv2.COLOR_BGR2RGB))
        input_img = transform(pil_img).unsqueeze(0)
        features = predictor(input_img).cpu().numpy()
        hist = calculate_histogram(segment)
        local_features.append(features)
        local_histograms.append(hist)
    
    upper_body_features = local_features[0]  
    lower_body_features = local_features[1]  
    
    upper_body_histogram = local_histograms[0]  
    lower_body_histogram = local_histograms[1]  

    return upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram

def normalize_features(features):
    norm = np.linalg.norm(features)
    if norm == 0:
        return features
    return features / norm

def calculate_combined_similarity(global_features1, global_features2, upper_body_features1, upper_body_features2, lower_body_features1, lower_body_features2, upper_body_histogram1, upper_body_histogram2, lower_body_histogram1, lower_body_histogram2):
    # Normalize features to enhance comparison
    global_features1 = normalize_features(global_features1)
    global_features2 = normalize_features(global_features2)
    upper_body_features1 = normalize_features(upper_body_features1)
    upper_body_features2 = normalize_features(upper_body_features2)
    lower_body_features1 = normalize_features(lower_body_features1)
    lower_body_features2 = normalize_features(lower_body_features2)

    global_similarity = np.dot(global_features1, global_features2.T)
    upper_body_similarity = np.dot(upper_body_features1, upper_body_features2.T)
    lower_body_similarity = np.dot(lower_body_features1, lower_body_features2.T)
    upper_body_hist_similarity = cv2.compareHist(upper_body_histogram1, upper_body_histogram2, cv2.HISTCMP_CORREL)
    lower_body_hist_similarity = cv2.compareHist(lower_body_histogram1, lower_body_histogram2, cv2.HISTCMP_CORREL)
    
    # Updated combined similarity calculation to include upper_body_hist_similarity with the new weight
    combined_similarity = (0.4 * global_similarity +  
                           0.1 * upper_body_similarity +
                           0.2 * lower_body_similarity +
                           0.2 * lower_body_hist_similarity +
                           0.1 * upper_body_hist_similarity)
    
    return combined_similarity

# 슬라이더 콜백 함수
def update_similarity_threshold(val):
    global SIMILARITY_THRESHOLD
    SIMILARITY_THRESHOLD = val / 100.0

def update_hold_time(val):
    global ID_HOLD_TIME
    ID_HOLD_TIME = val

def update_yolo_conf_threshold(val):
    global YOLO_CONF_THRESHOLD
    YOLO_CONF_THRESHOLD = val / 100.0

def update_yolo_iou_threshold(val):
    global YOLO_IOU_THRESHOLD
    YOLO_IOU_THRESHOLD = val / 100.0


# 웹캡으로부터 비디오 캡처
cap = cv2.VideoCapture(0)

# 창과 슬라이더 생성
cv2.namedWindow("Real-time Tracking and Re-ID")
cv2.createTrackbar("Similarity Threshold", "Real-time Tracking and Re-ID", int(SIMILARITY_THRESHOLD * 100), 100, update_similarity_threshold)
cv2.createTrackbar("ID Hold Time", "Real-time Tracking and Re-ID", ID_HOLD_TIME, 30, update_hold_time)
cv2.createTrackbar("YOLO Conf Threshold", "Real-time Tracking and Re-ID", int(YOLO_CONF_THRESHOLD * 100), 100, update_yolo_conf_threshold)
cv2.createTrackbar("YOLO IoU Threshold", "Real-time Tracking and Re-ID", int(YOLO_IOU_THRESHOLD * 100), 100, update_yolo_iou_threshold)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()

    results = model.track(classes=[0],
                           max_det=10, 
                           show=False, 
                           source=frame, 
                           tracker=tracker_config, 
                           stream=True,
                           conf = YOLO_CONF_THRESHOLD,
                           iou = YOLO_IOU_THRESHOLD,)

    active_ids = set()

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy().astype(int) if result.boxes is not None else []
        track_ids = result.boxes.id.cpu().numpy().astype(int) if result.boxes.id is not None else []

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            person_img = frame[y1:y2, x1:x2]

            # Extract global and local features along with histograms
            global_features = extract_global_features(person_img)
            upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram = extract_local_features(person_img)

            max_similarity = -1
            max_similarity_id = -1

            for known_id, (known_global_features, known_upper_body_features, known_lower_body_features, known_upper_body_histogram, known_lower_body_histogram) in known_persons.items():
                # Calculate combined similarity including the upper_body_hist_similarity
                similarity = calculate_combined_similarity(
                    global_features,
                    known_global_features,
                    upper_body_features,
                    known_upper_body_features,
                    lower_body_features,
                    known_lower_body_features,
                    upper_body_histogram,
                    known_upper_body_histogram,
                    lower_body_histogram,
                    known_lower_body_histogram
                )
                print(f"Known_Id: {known_id}, {max_similarity_id},  Calculated similarity:  {similarity}, {max_similarity}")

                if similarity > max_similarity:
                    max_similarity = similarity
                    max_similarity_id = known_id

            # Adjust the threshold for better distinction based on similarity scores
            if max_similarity > SIMILARITY_THRESHOLD and (max_similarity - SIMILARITY_THRESHOLD) > 0.05:
                if max_similarity_id not in active_ids:
                    person_id = max_similarity_id
                    color = (0, 255, 0)
            else:
                # Assign a new ID that hasn't been used
                if int(track_id) in assigned_ids:
                    person_id = int(track_id)
                else:
                    new_id = max(assigned_ids) + 1 if assigned_ids else 1
                    person_id = new_id
                    assigned_ids.add(person_id)
                
                if person_id not in known_persons:
                    known_persons[person_id] = (global_features, upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram)
                color = (0, 0, 255)

            id_last_seen[person_id] = current_time
            active_ids.add(person_id)

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            print(f"ID: {person_id}, Center: ({center_x}, {center_y})")
            centp = f"Center: ({center_x}, {center_y})"
            text_x = x2 + 10
            text_y = y2

            cv2.putText(frame, centp, (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            #cv2.putText(frame, centp, (text_x, text_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID: {person_id}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            #cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Clean up IDs that haven't been seen recently
    for person_id in list(id_last_seen.keys()):
        if current_time - id_last_seen[person_id] > ID_HOLD_TIME:
            del known_persons[person_id]
            del id_last_seen[person_id]
            assigned_ids.remove(person_id)  # Free up the ID for future use

    cv2.imshow("Real-time Tracking and Re-ID", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
