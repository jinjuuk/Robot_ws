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
# model = YOLO("yolov5n.pt")  # YOLOv5 Nano 모델 로드

# # BotSORT 설정 파일 경로
# tracker_config = "/home/jinjuuk/.local/lib/python3.10/site-packages/ultralytics/cfg/trackers/botsort.yaml"

# # FastReID 설정
# cfg = get_cfg()
# cfg.merge_from_file("/home/jinjuuk/fast-reid/configs/Market1501/bagtricks_R50.yml")
# cfg.MODEL.WEIGHTS = "/home/jinjuuk/dev_ws/pt_files/market_mgn_R50-ibn.pth"
# cfg.MODEL.DEVICE = "cpu"  # GPU를 사용할 경우 "cuda"로 변경

# predictor = DefaultPredictor(cfg)
# transform = build_transforms(cfg, is_train=False)

# # 이전에 감지된 사람들의 특징 벡터를 저장할 딕셔너리
# known_persons = {}
# person_id = 0

# try:
#     # 실시간 트래킹 시작
#     results = model.track(classes=[0],            # Track only the "person" class
#                           max_det=5,              # Detect up to 5 objects
#                           show=False,             # YOLO 결과를 별도로 보여주지 않음
#                           source=0,               # Use webcam as source
#                           tracker=tracker_config, # Use BotSORT tracker for Re-ID
#                           stream=True             # Stream results in real-time
#                           )

#     for result in results:
#         frame = result.orig_img  # 현재 프레임
#         boxes = result.boxes
#         for box in boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             person_img = frame[y1:y2, x1:x2]
            
#             # FastReID를 사용하여 특징 벡터 추출
#             img_pil = Image.fromarray(cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB))
#             image_tensor = transform(img_pil).unsqueeze(0)
#             with torch.no_grad():
#                 features = predictor(image_tensor).cpu().numpy()

#             # 가장 유사한 사람 찾기
#             max_similarity = -1
#             matched_id = -1

#             for id, known_features in known_persons.items():
#                 similarity = np.dot(features[0], known_features) / (np.linalg.norm(features[0]) * np.linalg.norm(known_features))
#                 if similarity > max_similarity:
#                     max_similarity = similarity
#                     matched_id = id

#             # 유사도 임계값 (조정 가능)
#             threshold = 0.7

#             if max_similarity > threshold:
#                 # 기존 사람으로 인식
#                 label = f"Person {matched_id}"
#             else:
#                 # 새로운 사람으로 인식
#                 person_id += 1
#                 known_persons[person_id] = features[0]
#                 label = f"Person {person_id}"


#             # 바운딩 박스와 레이블 그리기
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#         # 결과 프레임 출력
#         cv2.imshow('YOLOv5 + BotSORT + ReID', frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' 키를 눌러서 종료
#             break

# except Exception as e:
#     print(f"Error occurred: {e}")

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
# model = YOLO("yolov5n.pt")  # YOLOv5 Nano 모델 로드

# # BotSORT 설정 파일 경로
# tracker_config = "/home/jinjuuk/.local/lib/python3.10/site-packages/ultralytics/cfg/trackers/botsort.yaml"

# # FastReID 설정
# cfg = get_cfg()
# cfg.merge_from_file("/home/jinjuuk/fast-reid/configs/Market1501/bagtricks_R50.yml")
# cfg.MODEL.WEIGHTS = "/home/jinjuuk/dev_ws/pt_files/market_mgn_R50-ibn.pth"
# cfg.MODEL.DEVICE = "cpu"  # GPU를 사용할 경우 "cuda"로 변경

# predictor = DefaultPredictor(cfg)
# transform = build_transforms(cfg, is_train=False)


# # 실시간 트래킹 시작
# results = model.track(classes=[0],            # Track only the "person" class
#                       max_det=5,              # Detect up to 5 objects
#                       show=False,             # YOLO 결과를 별도로 보여주지 않음
#                       source=0,               # Use webcam as source
#                       tracker=tracker_config, # Use BotSORT tracker for Re-ID
#                       stream=True             # Stream results in real-time
#                       )



# # 유사도 임계값 설정 (조정 가능)
# SIMILARITY_THRESHOLD = 0.25

# # 알려진 사람들의 특징 벡터와 ID를 저장할 딕셔너리
# known_persons = {}


# # 실시간 트래킹 시작
# for result in results:
#     # YOLO 결과에서 바운딩 박스 추출
#     boxes = result.boxes.xyxy.cpu().numpy().astype(int)
    
#     # 트래킹 ID 추출 (없을 경우 대비)
#     if result.boxes.id is not None:
#         track_ids = result.boxes.id.cpu().numpy().astype(int)
#     else:
#         track_ids = np.arange(len(boxes))  # 임시 ID 생성
    
#     # 원본 이미지 가져오기
#     img = result.orig_img
    
#     for box, track_id in zip(boxes, track_ids):
#         x1, y1, x2, y2 = box
        
#         # 1. FastReID를 사용하여 특징 벡터 추출
#         person_img = img[y1:y2, x1:x2]
#         pil_img = Image.fromarray(cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB))
#         input_img = transform(pil_img).unsqueeze(0)
#         features = predictor(input_img).cpu().numpy()
        
#         # 2. 가장 유사한 사람 찾기
#         max_similarity = -1
#         max_similarity_id = -1
        
#         for known_id, known_features in known_persons.items():
#             similarity = np.dot(features, known_features.T) / (np.linalg.norm(features) * np.linalg.norm(known_features))
#             if similarity > max_similarity:
#                 max_similarity = similarity
#                 max_similarity_id = known_id
        
#         # 유사도 임계값 적용
#         if max_similarity > SIMILARITY_THRESHOLD:
#             # 기존 사람으로 인식 (기존 ID 유지)
#             person_id = max_similarity_id
#         else:
#             # 새로운 사람으로 인식 (새 ID 부여 또는 BotSORT ID 사용)
#             person_id = int(track_id)
#             known_persons[person_id] = features
        
#         # 3. 바운딩 박스와 레이블 그리기
#         color = (0, 255, 0)  # 초록색
#         cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
#         label = f"ID: {person_id}"
#         cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
#     # 결과 이미지 표시
#     cv2.imshow("Real-time Tracking and Re-ID", img)
    
#     # 'q' 키를 누르면 종료
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

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
# model = YOLO("yolov5n.pt")  # YOLOv5 Nano 모델 로드

# # BotSORT 설정 파일 경로
# tracker_config = "/home/jinjuuk/.local/lib/python3.10/site-packages/ultralytics/cfg/trackers/botsort.yaml"

# # FastReID 설정
# cfg = get_cfg()
# cfg.merge_from_file("/home/jinjuuk/fast-reid/configs/Market1501/bagtricks_R50.yml")
# cfg.MODEL.WEIGHTS = "/home/jinjuuk/dev_ws/pt_files/market_mgn_R50-ibn.pth"
# cfg.MODEL.DEVICE = "cpu"  # GPU를 사용할 경우 "cuda"로 변경

# predictor = DefaultPredictor(cfg)
# transform = build_transforms(cfg, is_train=False)


# # 실시간 트래킹 시작
# results = model.track(classes=[0],            # Track only the "person" class
#                       max_det=5,              # Detect up to 5 objects
#                       show=False,             # YOLO 결과를 별도로 보여주지 않음
#                       source=0,               # Use webcam as source
#                       tracker=tracker_config, # Use BotSORT tracker for Re-ID
#                       stream=True             # Stream results in real-time
#                       )



# # 유사도 임계값 설정 (조정 가능)
# SIMILARITY_THRESHOLD = 0.25

# # 알려진 사람들의 특징 벡터와 ID를 저장할 딕셔너리
# known_persons = {}

# # 실시간 트래킹 시작
# for result in results:
#     # YOLO 결과에서 바운딩 박스 추출
#     boxes = result.boxes.xyxy.cpu().numpy().astype(int)
    
#     # 트래킹 ID 추출 (없을 경우 대비)
#     if result.boxes.id is not None:
#         track_ids = result.boxes.id.cpu().numpy().astype(int)
#     else:
#         track_ids = np.arange(len(boxes))  # 임시 ID 생성
    
#     # 원본 이미지 가져오기
#     img = result.orig_img
    
#     for box, track_id in zip(boxes, track_ids):
#         x1, y1, x2, y2 = box
        
#         # 1. FastReID를 사용하여 특징 벡터 추출
#         person_img = img[y1:y2, x1:x2]
#         pil_img = Image.fromarray(cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB))
#         input_img = transform(pil_img).unsqueeze(0)
#         features = predictor(input_img).cpu().numpy()
        
#         # 2. 가장 유사한 사람 찾기
#         max_similarity = -1
#         max_similarity_id = -1
        
#         for known_id, known_features in known_persons.items():
#             similarity = np.dot(features, known_features.T) / (np.linalg.norm(features) * np.linalg.norm(known_features))
#             if similarity > max_similarity:
#                 max_similarity = similarity
#                 max_similarity_id = known_id
        
#         # 유사도 임계값 적용
#         if max_similarity > SIMILARITY_THRESHOLD:
#             # 기존 사람으로 인식 (기존 ID 유지)
#             person_id = max_similarity_id
#         else:
#             # 새로운 사람으로 인식 (새 ID 부여 또는 BotSORT ID 사용)
#             person_id = int(track_id)
#             known_persons[person_id] = features
        
#         # 3. 바운딩 박스와 레이블 그리기
#         color = (0, 255, 0)  # 초록색
#         cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
#         label = f"ID: {person_id}"
#         cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
#     # 결과 이미지 표시
#     cv2.imshow("Real-time Tracking and Re-ID", img)
    
#     # 'q' 키를 누르면 종료
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

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
# model = YOLO("yolov5n.pt")  # YOLOv5 Nano 모델 로드

# # BotSORT 설정 파일 경로
# tracker_config = "/home/jinjuuk/.local/lib/python3.10/site-packages/ultralytics/cfg/trackers/botsort.yaml"

# # FastReID 설정
# cfg = get_cfg()
# cfg.merge_from_file("/home/jinjuuk/fast-reid/configs/Market1501/bagtricks_R50.yml")
# cfg.MODEL.WEIGHTS = "/home/jinjuuk/dev_ws/pt_files/market_mgn_R50-ibn.pth"
# cfg.MODEL.DEVICE = "cpu"  # GPU를 사용할 경우 "cuda"로 변경

# predictor = DefaultPredictor(cfg)
# transform = build_transforms(cfg, is_train=False)

# # 유사도 임계값 설정 (조정 가능)
# SIMILARITY_THRESHOLD = 0.65

# # 알려진 사람들의 특징 벡터와 ID를 저장할 딕셔너리
# known_persons = {}

# # 실시간 트래킹 시작
# results = model.track(classes=[0],            # Track only the "person" class
#                       max_det=5,              # Detect up to 5 objects
#                       show=False,             # YOLO 결과를 별도로 보여주지 않음
#                       source=0,               # Use webcam as source
#                       tracker=tracker_config, # Use BotSORT tracker for Re-ID
#                       stream=True             # Stream results in real-time
#                       )

# for result in results:
#     # YOLO 결과에서 바운딩 박스 추출
#     boxes = result.boxes.xyxy.cpu().numpy().astype(int)
    
#     # 트래킹 ID 추출 (없을 경우 객체 무시)
#     if result.boxes.id is None:
#         continue  # ID가 없는 객체는 무시
#     track_ids = result.boxes.id.cpu().numpy().astype(int)
    
#     # 원본 이미지 가져오기
#     img = result.orig_img
    
#     for box, track_id in zip(boxes, track_ids):
#         x1, y1, x2, y2 = box
        
#         # 1. FastReID를 사용하여 특징 벡터 추출
#         person_img = img[y1:y2, x1:x2]
#         pil_img = Image.fromarray(cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB))
#         input_img = transform(pil_img).unsqueeze(0)
#         features = predictor(input_img).cpu().numpy()
        
#         # 2. 가장 유사한 사람 찾기
#         max_similarity = -1
#         max_similarity_id = -1
        
#         for known_id, known_features in known_persons.items():
#             similarity = np.dot(features, known_features.T) / (np.linalg.norm(features) * np.linalg.norm(known_features))
#             if similarity > max_similarity:
#                 max_similarity = similarity
#                 max_similarity_id = known_id
        
#         # 3. 유사도 임계값 적용
#         if max_similarity > SIMILARITY_THRESHOLD:
#             # 기존 사람으로 인식 (기존 ID 유지)
#             person_id = max_similarity_id
#         else:
#             # 새로운 사람으로 인식 (BotSORT ID 사용)
#             person_id = int(track_id)
#             known_persons[person_id] = features
        
#         # 4. 바운딩 박스와 레이블 그리기
#         color = (0, 255, 0)  # 초록색
#         cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
#         label = f"ID: {person_id}"
#         cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
#     # 결과 이미지 표시
#     cv2.imshow("Real-time Tracking and Re-ID", img)
    
#     # 'q' 키를 누르면 종료
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

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
# model = YOLO("yolov5n.pt")  # YOLOv5 Nano 모델 로드

# # BotSORT 설정 파일 경로
# tracker_config = "/home/jinjuuk/.local/lib/python3.10/site-packages/ultralytics/cfg/trackers/botsort.yaml"

# # FastReID 설정
# cfg = get_cfg()
# cfg.merge_from_file("/home/jinjuuk/fast-reid/configs/Market1501/bagtricks_R50.yml")
# cfg.MODEL.WEIGHTS = "/home/jinjuuk/dev_ws/pt_files/market_mgn_R50-ibn.pth"
# cfg.MODEL.DEVICE = "cpu"  # GPU를 사용할 경우 "cuda"로 변경

# predictor = DefaultPredictor(cfg)
# transform = build_transforms(cfg, is_train=False)

# # 유사도 임계값 설정 (조정 가능)
# SIMILARITY_THRESHOLD = 0.99992


# # 알려진 사람들의 특징 벡터와 ID를 저장할 딕셔너리
# known_persons = {}

# # 실시간 트래킹 시작
# results = model.track(classes=[0],            # Track only the "person" class
#                       max_det=5,              # Detect up to 5 objects
#                       show=False,             # YOLO 결과를 별도로 보여주지 않음
#                       source=0,               # Use webcam as source
#                       tracker=tracker_config, # Use BotSORT tracker for Re-ID
#                       stream=True             # Stream results in real-time
#                       )

# for result in results:
#     # 원본 이미지 가져오기
#     img = result.orig_img

#     # YOLO 결과에서 바운딩 박스 추출
#     boxes = result.boxes.xyxy.cpu().numpy().astype(int) if result.boxes is not None else []

#     # 트래킹 ID 추출 (없을 경우 객체 무시)
#     track_ids = result.boxes.id.cpu().numpy().astype(int) if result.boxes is not None and result.boxes.id is not None else []

#     if len(boxes) > 0:  # 사람이 감지된 경우에만 처리
#         for box, track_id in zip(boxes, track_ids):
#             x1, y1, x2, y2 = box

#             # 1. FastReID를 사용하여 특징 벡터 추출
#             person_img = img[y1:y2, x1:x2]
#             pil_img = Image.fromarray(cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB))
#             input_img = transform(pil_img).unsqueeze(0)
#             features = predictor(input_img).cpu().numpy()

#             # 2. 가장 유사한 사람 찾기
#             max_similarity = -1
#             max_similarity_id = -1

#             for known_id, known_features in known_persons.items():
#                 similarity = np.dot(features, known_features.T) / (np.linalg.norm(features) * np.linalg.norm(known_features))
#                 print(f"Calculated similarity and Known_Id: {known_id},    {similarity}")  # 유사도 값 디버깅 출력
#                 if similarity > max_similarity:
#                     max_similarity = similarity
#                     max_similarity_id = known_id


#             # 3. 유사도 임계값 적용
#             if max_similarity > SIMILARITY_THRESHOLD:
#                 # 기존 사람으로 인식 (기존 ID 유지)
#                 person_id = max_similarity_id
#             else:
#                 # 새로운 사람으로 인식 (BotSORT ID 사용)
#                 person_id = int(track_id)
#                 known_persons[person_id] = features

#             # 4. 바운딩 박스와 레이블 그리기
#             color = (0, 255, 0)  # 초록색
#             cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
#             label = f"ID: {person_id}"
#             cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
#     # 사람이 감지되지 않아도 프레임을 계속 출력
#     cv2.imshow("Real-time Tracking and Re-ID", img)
    
#     # 'q' 키를 누르면 종료
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

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
# model = YOLO("yolov5n.pt")  # YOLOv5 Nano 모델 로드

# # BotSORT 설정 파일 경로
# tracker_config = "/home/jinjuuk/.local/lib/python3.10/site-packages/ultralytics/cfg/trackers/botsort.yaml"

# # FastReID 설정
# cfg = get_cfg()
# cfg.merge_from_file("/home/jinjuuk/fast-reid/configs/Market1501/bagtricks_R50.yml")
# cfg.MODEL.WEIGHTS = "/home/jinjuuk/dev_ws/pt_files/market_mgn_R50-ibn.pth"
# cfg.MODEL.DEVICE = "cpu"  # GPU를 사용할 경우 "cuda"로 변경

# predictor = DefaultPredictor(cfg)
# transform = build_transforms(cfg, is_train=False)

# # 유사도 임계값 설정 (조정 가능)
# SIMILARITY_THRESHOLD = 0.99982


# # 알려진 사람들의 특징 벡터와 ID를 저장할 딕셔너리
# known_persons = {}

# # 실시간 트래킹 시작
# results = model.track(classes=[0],            # Track only the "person" class
#                       max_det=5,              # Detect up to 5 objects
#                       show=False,             # YOLO 결과를 별도로 보여주지 않음
#                       source=0,               # Use webcam as source
#                       tracker=tracker_config, # Use BotSORT tracker for Re-ID
#                       stream=True             # Stream results in real-time
#                       )

# for result in results:
#     # 원본 이미지 가져오기
#     img = result.orig_img

#     # YOLO 결과에서 바운딩 박스 추출
#     boxes = result.boxes.xyxy.cpu().numpy().astype(int) if result.boxes is not None else []

#     # 트래킹 ID 추출 (없을 경우 객체 무시)
#     track_ids = result.boxes.id.cpu().numpy().astype(int) if result.boxes is not None and result.boxes.id is not None else []

#     if len(boxes) > 0:  # 사람이 감지된 경우에만 처리
#         for box, track_id in zip(boxes, track_ids):
#             x1, y1, x2, y2 = box

#             # 1. FastReID를 사용하여 특징 벡터 추출
#             person_img = img[y1:y2, x1:x2]
#             pil_img = Image.fromarray(cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB))
#             input_img = transform(pil_img).unsqueeze(0)
#             features = predictor(input_img).cpu().numpy()

#             # 2. 가장 유사한 사람 찾기
#             max_similarity = -1
#             max_similarity_id = -1

#             for known_id, known_features in known_persons.items():
#                 similarity = np.dot(features, known_features.T) / (np.linalg.norm(features) * np.linalg.norm(known_features))
#                 print(f"Known_Id: {known_id},  Calculated similarity:  {similarity}")  # 유사도 값 디버깅 출력
#                 if similarity > max_similarity:
#                     max_similarity = similarity
#                     max_similarity_id = known_id


#             # 3. 유사도 임계값 적용
#             if max_similarity > SIMILARITY_THRESHOLD:
#                 # 기존 사람으로 인식 (기존 ID 유지)
#                 person_id = max_similarity_id
#             else:
#                 # 새로운 사람으로 인식 (BotSORT ID 사용)
#                 person_id = int(track_id)
#                 known_persons[person_id] = features
  

#             # 4. 바운딩 박스와 레이블 그리기
#             color = (0, 255, 0)  # 초록색
#             cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
#             label = f"ID: {person_id}"
#             cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
#     # 사람이 감지되지 않아도 프레임을 계속 출력
#     cv2.imshow("Real-time Tracking and Re-ID", img)
    
#     # 'q' 키를 누르면 종료
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

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
model = YOLO("yolov5n.pt")  # YOLOv5 Nano 모델 로드

# BotSORT 설정 파일 경로
tracker_config = "/home/jinjuuk/.local/lib/python3.10/site-packages/ultralytics/cfg/trackers/botsort.yaml"

# FastReID 설정
cfg = get_cfg()
cfg.merge_from_file("/home/jinjuuk/fast-reid/configs/Market1501/bagtricks_R50.yml")
cfg.MODEL.WEIGHTS = "/home/jinjuuk/dev_ws/pt_files/market_mgn_R50-ibn.pth"
cfg.MODEL.DEVICE = "cpu"  # GPU를 사용할 경우 "cuda"로 변경

predictor = DefaultPredictor(cfg)
transform = build_transforms(cfg, is_train=False)

# 유사도 임계값 설정
SIMILARITY_THRESHOLD = 0.99982
TIME_THRESHOLD = 3  # 3초 동안 유지된 유사도를 기준으로 ID 부여
SIMILARITY_RANGE = 0.01  # 유사도 범위값 설정 (0.001은 ±0.0005의 범위)

# 알려진 사람들의 특징 벡터와 ID를 저장할 딕셔너리
known_persons = {}
time_tracking = {}  # 각 ID에 대한 마지막 인식 시간을 기록하는 딕셔너리

# 실시간 트래킹 시작
results = model.track(classes=[0],            # Track only the "person" class
                      max_det=5,              # Detect up to 5 objects
                      show=False,             # YOLO 결과를 별도로 보여주지 않음
                      source=0,               # Use webcam as source
                      tracker=tracker_config, # Use BotSORT tracker for Re-ID
                      stream=True             # Stream results in real-time
                      )

for result in results:
    # 원본 이미지 가져오기
    img = result.orig_img

    # YOLO 결과에서 바운딩 박스 추출
    boxes = result.boxes.xyxy.cpu().numpy().astype(int) if result.boxes is not None else []

    # 트래킹 ID 추출 (없을 경우 객체 무시)
    track_ids = result.boxes.id.cpu().numpy().astype(int) if result.boxes is not None and result.boxes.id is not None else []

    current_time = time.time()

    if len(boxes) > 0:  # 사람이 감지된 경우에만 처리
        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box

            # 1. FastReID를 사용하여 특징 벡터 추출
            person_img = img[y1:y2, x1:x2]
            pil_img = Image.fromarray(cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB))
            input_img = transform(pil_img).unsqueeze(0)
            features = predictor(input_img).cpu().numpy()

            # 2. 가장 유사한 사람 찾기
            max_similarity = -1
            max_similarity_id = -1

            for known_id, known_features in known_persons.items():
                similarity = np.dot(features, known_features.T) / (np.linalg.norm(features) * np.linalg.norm(known_features))
                print(f"Known_Id: {known_id},  Calculated similarity:  {similarity}")  # 유사도 값 디버깅 출력
                if similarity > max_similarity:
                    max_similarity = similarity
                    max_similarity_id = known_id

            # 3. 유사도와 시간 임계값 적용
            if max_similarity > SIMILARITY_THRESHOLD:
                # 유사도가 범위 내에서 일정 시간 동안 유지된 경우 동일한 ID 부여
                if max_similarity_id in time_tracking:
                    if current_time - time_tracking[max_similarity_id] <= TIME_THRESHOLD:
                        # 3초 이내 동일 유사도 범위 유지 시 동일인으로 간주
                        if abs(max_similarity - SIMILARITY_THRESHOLD) <= SIMILARITY_RANGE:
                            person_id = max_similarity_id
                        else:
                            person_id = int(track_id)
                            known_persons[person_id] = features
                            time_tracking[person_id] = current_time
                    else:
                        person_id = int(track_id)
                        known_persons[person_id] = features
                        time_tracking[person_id] = current_time
                else:
                    person_id = max_similarity_id
                    time_tracking[max_similarity_id] = current_time
            else:
                # 새로운 사람으로 인식 (BotSORT ID 사용)
                person_id = int(track_id)
                known_persons[person_id] = features
                time_tracking[person_id] = current_time

            # 4. 바운딩 박스와 레이블 그리기
            color = (0, 255, 0)  # 초록색
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = "Person"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # 사람이 감지되지 않아도 프레임을 계속 출력
    cv2.imshow("Real-time Tracking and Re-ID", img)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
