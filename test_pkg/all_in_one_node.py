# import collections
# if not hasattr(collections, 'Mapping'):
#     import collections.abc
#     collections.Mapping = collections.abc.Mapping
# import cv2
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# from geometry_msgs.msg import Twist, Point
# from cv_bridge import CvBridge
# import numpy as np
# import torch
# from fastreid.config import get_cfg
# from fastreid.engine import DefaultPredictor
# from fastreid.data.transforms import build_transforms
# from PIL import Image as PILImage
# from ultralytics import YOLO
# from message.msg import IdDistanceInfo


# # CUDA 디바이스 설정
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # YOLOv5 모델 로드
# model = YOLO("yolov5su.pt").to(device)

# # BotSORT 설정 파일 경로
# tracker_config = "/home/jinjuuk/.local/lib/python3.10/site-packages/ultralytics/cfg/trackers/botsort.yaml"

# # FastReID 설정
# cfg = get_cfg()
# cfg.merge_from_file("/home/jinjuuk/fast-reid/configs/Market1501/bagtricks_R50.yml")
# cfg.MODEL.WEIGHTS = "/home/jinjuuk/dev_ws/pt_files/market_mgn_R50-ibn.pth"
# cfg.MODEL.DEVICE = "cuda"  # GPU 사용 설정

# predictor = DefaultPredictor(cfg)
# transform = build_transforms(cfg, is_train=False)

# # 알려진 사람들의 특징 벡터와 ID를 저장할 딕셔너리
# known_persons = {}

# # 초기 설정값
# SIMILARITY_THRESHOLD = 0.97

# class AllInOneNode(Node):
#     def __init__(self):
#         super().__init__('all_in_one_node')
        
#         # 로그 출력: CUDA 사용 여부 확인
#         if torch.cuda.is_available():
#             cuda_device_name = torch.cuda.get_device_name(0)
#             self.get_logger().info(f"CUDA is available. Using GPU: {cuda_device_name}")
#         else:
#             self.get_logger().info("CUDA is not available. Using CPU")

#         # 구독자 설정
#         self.subscription_color = self.create_subscription(
#             Image,
#             '/camera/color/image_raw',
#             self.color_image_callback,
#             10
#         )
#         self.subscription_depth = self.create_subscription(
#             Image,
#             '/camera/depth/image_raw',
#             self.depth_image_callback,
#             10
#         )
        
#         # 발행자 설정
#         self.publisher_cmd_vel = self.create_publisher(
#             Twist,
#             'Twist/cmd_vel',
#             10
#         )
#         self.publisher_id_distance = self.create_publisher(
#             IdDistanceInfo,  
#             'id_distance_info',
#             10
#         )
        
#         # OpenCV를 위한 CVBridge 초기화
#         self.bridge = CvBridge()

#         # 트래킹 정보 저장
#         self.tracked_objects = {}  # id -> (bbox_center, bbox, depth)
#         self.id_last_seen = {}  # id -> last seen time

#         self.get_logger().info("AllInOneNode initialized")


#     def color_image_callback(self, msg):
#         self.get_logger().info("Received color image")
        
#         # 이미지 변환
#         frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
#         self.current_color_image = frame.copy()
#         self.get_logger().info("Color image converted to OpenCV format")
        
#         results = model.track(classes=[0], max_det=5, show=False, source=frame, stream=True, tracker=tracker_config)
#         results = list(results)  # generator를 list로 변환
#         self.get_logger().info(f"YOLO tracking complete. Detected {len(results)} results")

#         active_ids = set()
#         current_time = self.get_clock().now()

#         for result in results:
#             boxes = result.boxes.xyxy.cpu().numpy().astype(int) if result.boxes is not None else []
#             track_ids = result.boxes.id.cpu().numpy().astype(int) if result.boxes.id is not None else []

#             for box, track_id in zip(boxes, track_ids):
#                 x1, y1, x2, y2 = box
#                 person_img = frame[y1:y2, x1:x2]

#                 global_features = self.extract_global_features(person_img)
#                 upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram = self.extract_local_features(person_img)

#                 max_similarity = -1
#                 max_similarity_id = -1

#                 for known_id, (known_global_features, known_upper_body_features, known_lower_body_features, known_upper_body_histogram, known_lower_body_histogram) in known_persons.items():
#                     similarity = self.calculate_combined_similarity(global_features, known_global_features, upper_body_features, known_upper_body_features, lower_body_features, known_lower_body_features, upper_body_histogram, known_upper_body_histogram, lower_body_histogram, known_lower_body_histogram)
#                     self.get_logger().info(f"Known_Id: {known_id}, {max_similarity_id},  Calculated similarity:  {similarity}, {max_similarity}")

#                     if similarity > max_similarity:
#                         max_similarity = similarity
#                         max_similarity_id = known_id

#                 if max_similarity > SIMILARITY_THRESHOLD:
#                     if max_similarity_id not in active_ids:
#                         person_id = max_similarity_id
#                         color = (0, 255, 0)
#                 else:
#                     person_id = int(track_id)
#                     if person_id in known_persons:
#                         pass
#                     else:
#                         new_id = max(known_persons.keys(), default=0) + 1
#                         person_id = new_id
#                     known_persons[person_id] = (global_features, upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram)
#                     color = (0, 0, 255)

#                 self.id_last_seen[person_id] = current_time
#                 active_ids.add(person_id)

#                 # 바운딩 박스의 중심 좌표 계산
#                 bbox_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
#                 self.tracked_objects[person_id] = (bbox_center, (x1, y1, x2, y2), None)  # depth_value는 아직 없음

#                 # 실시간으로 바운딩 박스 표시
#                 cv2.rectangle(self.current_color_image, (x1, y1), (x2, y2), color, 2)
#                 label = f"ID: {person_id}"
#                 cv2.putText(self.current_color_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

#                 # 추가적으로 센터 포인트 및 그 정보 표시
#                 center_x = (x1 + x2) // 2
#                 center_y = (y1 + y2) // 2
#                 centp = f"Center: ({center_x}, {center_y})"
#                 cv2.putText(self.current_color_image, centp, (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)



#     def depth_image_callback(self, msg):
#         self.get_logger().info("Received depth image")
        
#         if not self.tracked_objects:
#             self.get_logger().info("No tracked objects available, skipping depth processing")
#             return

#         # 깊이 이미지를 OpenCV 형식으로 변환
#         depth_frame = self.bridge.imgmsg_to_cv2(msg, '16UC1')
#         self.get_logger().info("Depth image converted to OpenCV format")
        
#         active_ids = set()
#         current_time = self.get_clock().now()

#         for person_id, (bbox_center, bbox, _) in self.tracked_objects.items():
#             if bbox is None:
#                 self.get_logger().info(f"No bbox available for Person ID: {person_id}, skipping.")
#                 continue
            
#             x, y = bbox_center
#             x1, y1, x2, y2 = bbox

#             width = x2 - x1
#             height = y2 - y1

#             # 중심 좌표에서 상하좌우 3픽셀씩 확장된 영역 내의 깊이 값을 추출하여 평균 계산
#             depth_values = []
#             for dx in range(-10, 11):  # 중심을 기준으로 -3에서 +3까지의 범위
#                 for dy in range(-10, 11):
#                     nx, ny = x + dx, y + dy
#                     # 이미지 경계를 벗어나는 경우를 처리
#                     if 0 <= nx < depth_frame.shape[1] and 0 <= ny < depth_frame.shape[0]:
#                         depth_values.append(depth_frame[ny, nx])

#             depth_value = round(np.mean(depth_values)/10, 2)
#             self.get_logger().info(f"Person ID: {person_id}, Bbox center: {bbox_center}, Width: {width}, Height: {height}, Depth: {depth_value}cm")

#             # Update tracking information with depth value
#             self.tracked_objects[person_id] = (bbox_center, bbox, depth_value)

#             # Update the last seen time for the ID
#             self.id_last_seen[person_id] = current_time
#             active_ids.add(person_id)

#             # Define color for bounding box and text
#             color = (0, 255, 0)

#             # 확장된 영역을 실시간 이미지에 bbox로 표시
#             cv2.rectangle(self.current_color_image, (x-10, y-10), (x+10, y+10), color, 2)

#             # bbox와 관련된 정보를 실시간 이미지에 표시
#             depth_text = f"Depth: {depth_value}cm"
#             cv2.putText(self.current_color_image, depth_text, (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

#             # 커스텀 메시지 발행
#             self.publish_id_distance_info(person_id, bbox_center, width, height, depth_value)

#         # 실시간으로 처리된 이미지를 표시
#         cv2.imshow("Color Image with Tracking", self.current_color_image)
#         cv2.waitKey(1)



#     def publish_id_distance_info(self, id, bbox_center, width, height, depth_value):
#         self.get_logger().info(f"Preparing to publish IdDistanceInfo: ID={id}, Bbox Center={bbox_center}, Width={width}, Height={height}, Depth={depth_value}")
        
#         msg = IdDistanceInfo()
#         msg.id = id
#         msg.bbox_center = Point(x=float(bbox_center[0]), y=float(bbox_center[1]), z=0.0)
#         msg.width = float(width)
#         msg.height = float(height)
#         msg.distance = float(depth_value)
        
#         # 발행 전 메시지 정보 로그
#         self.get_logger().info(f"IdDistanceInfo message prepared: {msg}")
        
#         self.publisher_id_distance.publish(msg)
#         self.get_logger().info(f"Published IdDistanceInfo: ID={id}, Bbox Center={bbox_center}, Width={width}, Height={height}, Depth={depth_value}")


#     def extract_global_features(self, image):
#         pil_img = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#         input_img = transform(pil_img).unsqueeze(0).to(device)
#         features = predictor(input_img).cpu().numpy()
#         return features

#     def extract_local_features(self, image):
#         height, width, _ = image.shape
#         segments = [
#             image[:height//2, :],  
#             image[height//2:, :],  
#             image[:, :width//2],   
#             image[:, width//2:]    
#         ]
        
#         local_features = []
#         local_histograms = []
#         for segment in segments:
#             pil_img = PILImage.fromarray(cv2.cvtColor(segment, cv2.COLOR_BGR2RGB))
#             input_img = transform(pil_img).unsqueeze(0).to(device)
#             features = predictor(input_img).cpu().numpy()
#             hist = self.calculate_histogram(segment)
#             local_features.append(features)
#             local_histograms.append(hist)
        
#         upper_body_features = local_features[0]  
#         lower_body_features = local_features[1]  
        
#         upper_body_histogram = local_histograms[0]  
#         lower_body_histogram = local_histograms[1]  

#         return upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram

#     def calculate_histogram(self, image):
#         hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#         hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
#         cv2.normalize(hist, hist)
#         return hist.flatten()

#     def calculate_combined_similarity(self, global_features1, global_features2, upper_body_features1, upper_body_features2, lower_body_features1, lower_body_features2, upper_body_histogram1, upper_body_histogram2, lower_body_histogram1, lower_body_histogram2):
#         global_similarity = np.dot(global_features1, global_features2.T) / (np.linalg.norm(global_features1) * np.linalg.norm(global_features2))
#         upper_body_similarity = np.dot(upper_body_features1, upper_body_features2.T) / (np.linalg.norm(upper_body_features1) * np.linalg.norm(upper_body_features2))
#         lower_body_similarity = np.dot(lower_body_features1, lower_body_features2.T) / (np.linalg.norm(lower_body_features1) * np.linalg.norm(lower_body_features2))
#         upper_body_hist_similarity = cv2.compareHist(upper_body_histogram1, upper_body_histogram2, cv2.HISTCMP_CORREL)
#         lower_body_hist_similarity = cv2.compareHist(lower_body_histogram1, lower_body_histogram2, cv2.HISTCMP_CORREL)
        
#         combined_similarity = (0.3 * global_similarity +  
#                                 0.1 * upper_body_similarity +
#                                 0.3 * lower_body_similarity +
#                                 0.3 * upper_body_hist_similarity)
#         return combined_similarity

# def main(args=None):
#     rclpy.init(args=args)
#     node = AllInOneNode()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()





# import collections
# if not hasattr(collections, 'Mapping'):
#     import collections.abc
#     collections.Mapping = collections.abc.Mapping
# import cv2
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# from geometry_msgs.msg import Twist, Point
# from cv_bridge import CvBridge
# import numpy as np
# import torch
# from fastreid.config import get_cfg
# from fastreid.engine import DefaultPredictor
# from fastreid.data.transforms import build_transforms
# from PIL import Image as PILImage
# from ultralytics import YOLO
# from message.msg import IdDistanceInfo
# import time


# # CUDA 디바이스 설정
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # YOLOv5 모델 로드
# model = YOLO("yolov5su.pt").to(device)

# # BotSORT 설정 파일 경로
# tracker_config = "/home/jinjuuk/.local/lib/python3.10/site-packages/ultralytics/cfg/trackers/botsort.yaml"

# # FastReID 설정
# cfg = get_cfg()
# cfg.merge_from_file("/home/jinjuuk/fast-reid/configs/Market1501/bagtricks_R50.yml")
# cfg.MODEL.WEIGHTS = "/home/jinjuuk/dev_ws/pt_files/market_mgn_R50-ibn.pth"
# cfg.MODEL.DEVICE = "cuda"  # GPU 사용 설정

# predictor = DefaultPredictor(cfg)
# transform = build_transforms(cfg, is_train=False)

# # 알려진 사람들의 특징 벡터와 ID를 저장할 딕셔너리
# known_persons = {}

# # 초기 설정값
# SIMILARITY_THRESHOLD = 0.97

# class AllInOneNode(Node):
#     def __init__(self):
#         super().__init__('all_in_one_node')
        
#         # 로그 출력: CUDA 사용 여부 확인
#         if torch.cuda.is_available():
#             cuda_device_name = torch.cuda.get_device_name(0)
#             self.get_logger().info(f"CUDA is available. Using GPU: {cuda_device_name}")
#         else:
#             self.get_logger().info("CUDA is not available. Using CPU")

#         # 구독자 설정
#         self.subscription_color = self.create_subscription(
#             Image,
#             '/camera/color/image_raw',
#             self.color_image_callback,
#             10
#         )
#         self.subscription_depth = self.create_subscription(
#             Image,
#             '/camera/depth/image_raw',
#             self.depth_image_callback,
#             10
#         )
        
#         # 발행자 설정
#         self.publisher_cmd_vel = self.create_publisher(
#             Twist,
#             'Twist/cmd_vel',
#             10
#         )
#         self.publisher_id_distance = self.create_publisher(
#             IdDistanceInfo,  
#             'id_distance_info',
#             10
#         )
        
#         # OpenCV를 위한 CVBridge 초기화
#         self.bridge = CvBridge()

#         # 트래킹 정보 저장
#         self.tracked_objects = {}  # id -> (bbox_center, bbox, depth)
#         self.id_last_seen = {}  # id -> last seen time
#          # 각 person_id에 대한 마지막 유효한 깊이 값을 저장하는 딕셔너리 초기화
#         self.last_valid_depth = {}

#         self.get_logger().info("AllInOneNode initialized")


#     def color_image_callback(self, msg):
#         self.get_logger().info("Received color image")
        
#         # 이미지 변환
#         frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
#         self.current_color_image = frame.copy()
#         self.get_logger().info("Color image converted to OpenCV format")
        
#         results = model.track(classes=[0], max_det=10, show=False, source=frame, stream=True, tracker=tracker_config)#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#         results = list(results)  # generator를 list로 변환
#         self.get_logger().info(f"YOLO tracking complete. Detected {len(results)} results")

#         active_ids = set()
#         current_time = self.get_clock().now()

#         for result in results:
#             boxes = result.boxes.xyxy.cpu().numpy().astype(int) if result.boxes is not None else []
#             track_ids = result.boxes.id.cpu().numpy().astype(int) if result.boxes.id is not None else []

#             for box, track_id in zip(boxes, track_ids):
#                 x1, y1, x2, y2 = box
#                 person_img = frame[y1:y2, x1:x2]

#                 global_features = self.extract_global_features(person_img)
#                 upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram = self.extract_local_features(person_img)

#                 max_similarity = -1
#                 max_similarity_id = -1

#                 for known_id, (known_global_features, known_upper_body_features, known_lower_body_features, known_upper_body_histogram, known_lower_body_histogram) in known_persons.items():
#                     similarity = self.calculate_combined_similarity(global_features, known_global_features, upper_body_features, known_upper_body_features, lower_body_features, known_lower_body_features, upper_body_histogram, known_upper_body_histogram, lower_body_histogram, known_lower_body_histogram)
#                     self.get_logger().info(f"Known_Id: {known_id}, {max_similarity_id},  Calculated similarity:  {similarity}, {max_similarity}")

#                     if similarity > max_similarity:
#                         max_similarity = similarity
#                         max_similarity_id = known_id

#                 if max_similarity > SIMILARITY_THRESHOLD:
#                     if max_similarity_id not in active_ids:
#                         person_id = max_similarity_id
#                         color = (0, 255, 0)
#                 else:
#                     person_id = int(track_id)
#                     if person_id in known_persons:
#                         pass
#                     else:
#                         new_id = max(known_persons.keys(), default=0) + 1
#                         person_id = new_id
#                     known_persons[person_id] = (global_features, upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram)
#                     color = (0, 0, 255)

#                 self.id_last_seen[person_id] = current_time
#                 active_ids.add(person_id)

#                 # 바운딩 박스의 중심 좌표 계산
#                 bbox_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
#                 self.tracked_objects[person_id] = (bbox_center, (x1, y1, x2, y2), None)  # depth_value는 아직 없음

#                 # 실시간으로 바운딩 박스 표시
#                 cv2.rectangle(self.current_color_image, (x1, y1), (x2, y2), color, 2)
#                 label = f"ID: {person_id}"
#                 cv2.putText(self.current_color_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

#                 # 추가적으로 센터 포인트 및 그 정보 표시
#                 center_x = (x1 + x2) // 2
#                 center_y = (y1 + y2) // 2
#                 centp = f"Center: ({center_x}, {center_y})"
#                 cv2.putText(self.current_color_image, centp, (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


#     def depth_image_callback(self, msg):
#         self.get_logger().info("Received depth image")
        
#         if not self.tracked_objects:
#             self.get_logger().info("No tracked objects available, skipping depth processing")
#             return

#         # 깊이 이미지를 OpenCV 형식으로 변환
#         depth_frame = self.bridge.imgmsg_to_cv2(msg, '16UC1')
#         self.get_logger().info("Depth image converted to OpenCV format")
        
#         active_ids = set()
#         current_time = self.get_clock().now()

#         for person_id, (bbox_center, bbox, _) in self.tracked_objects.items():
#             if bbox is None:
#                 self.get_logger().info(f"No bbox available for Person ID: {person_id}, skipping.")
#                 continue
            
#             x1, y1, x2, y2 = bbox
#             x, y = bbox_center
#             width = x2 - x1
#             height = y2 - y1

#             # 깊이 값이 0인 경우, 이전에 유효했던 깊이 값으로 대체하는 로직
#             if 0 <= x < depth_frame.shape[1] and 0 <= y < depth_frame.shape[0]:
#                 depth_value = round(depth_frame[y, x] / 10, 2)  # mm를 cm로 변환 후 소수점 2자리까지 표시
                
#                 # 깊이 값이 0인 경우
#                 if depth_value == 0:
#                     if person_id in self.last_valid_depth:
#                         depth_value = self.last_valid_depth[person_id]
#                         self.get_logger().info(f"Depth value was 0. Using last valid depth value: {depth_value}cm")
#                     else:
#                         self.get_logger().info(f"No valid depth value available for Person ID: {person_id}. Skipping.")
#                         continue
#                 else:
#                     # 깊이 값이 유효한 경우, 해당 값을 저장
#                     self.last_valid_depth[person_id] = depth_value
#             else:
#                 depth_value = None
#                 self.get_logger().info(f"Bbox center {bbox_center} is out of depth frame bounds.")

#             self.get_logger().info(f"Person ID: {person_id}, Bbox center: {bbox_center}, Width: {width}, Height: {height}, Depth: {depth_value}cm")

#             # Update tracking information with depth value
#             self.tracked_objects[person_id] = (bbox_center, bbox, depth_value)

#             # Update the last seen time for the ID
#             self.id_last_seen[person_id] = current_time
#             active_ids.add(person_id)

#             # Define color for bounding box and text
#             color = (0, 255, 0)

#             depth_text = f"Depth: {depth_value}cm"
#             cv2.putText(self.current_color_image, depth_text, (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

#             # 커스텀 메시지 발행
#             self.publish_id_distance_info(person_id, bbox_center, width, height, depth_value)

#         # 실시간으로 처리된 이미지를 표시
#         cv2.imshow("Color Image with Tracking", self.current_color_image)
#         cv2.waitKey(1)
        



#     def publish_id_distance_info(self, id, bbox_center, width, height, depth_value):
#         self.get_logger().info(f"Preparing to publish IdDistanceInfo: ID={id}, Bbox Center={bbox_center}, Width={width}, Height={height}, Depth={depth_value}")
        
#         msg = IdDistanceInfo()
#         msg.id = id
#         msg.bbox_center = Point(x=float(bbox_center[0]), y=float(bbox_center[1]), z=0.0)
#         msg.width = float(width)
#         msg.height = float(height)
#         msg.distance = float(depth_value)
        
#         # 발행 전 메시지 정보 로그
#         self.get_logger().info(f"IdDistanceInfo message prepared: {msg}")
        
#         self.publisher_id_distance.publish(msg)
#         self.get_logger().info(f"Published IdDistanceInfo: ID={id}, Bbox Center={bbox_center}, bbox_Width={width}, bbox_Height={height}, Depth={depth_value}")


#     def extract_global_features(self, image):
#         pil_img = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#         input_img = transform(pil_img).unsqueeze(0).to(device)
#         features = predictor(input_img).cpu().numpy()
#         return features

#     def extract_local_features(self, image):
#         height, width, _ = image.shape
#         segments = [
#             image[:height//2, :],  
#             image[height//2:, :],  
#             image[:, :width//2],   
#             image[:, width//2:]    
#         ]
        
#         local_features = []
#         local_histograms = []
#         for segment in segments:
#             pil_img = PILImage.fromarray(cv2.cvtColor(segment, cv2.COLOR_BGR2RGB))
#             input_img = transform(pil_img).unsqueeze(0).to(device)
#             features = predictor(input_img).cpu().numpy()
#             hist = self.calculate_histogram(segment)
#             local_features.append(features)
#             local_histograms.append(hist)
        
#         upper_body_features = local_features[0]  
#         lower_body_features = local_features[1]  
        
#         upper_body_histogram = local_histograms[0]  
#         lower_body_histogram = local_histograms[1]  

#         return upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram

#     def calculate_histogram(self, image):
#         hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#         hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
#         cv2.normalize(hist, hist)
#         return hist.flatten()

#     def calculate_combined_similarity(self, global_features1, global_features2, upper_body_features1, upper_body_features2, lower_body_features1, lower_body_features2, upper_body_histogram1, upper_body_histogram2, lower_body_histogram1, lower_body_histogram2):
#         global_similarity = np.dot(global_features1, global_features2.T) / (np.linalg.norm(global_features1) * np.linalg.norm(global_features2))
#         upper_body_similarity = np.dot(upper_body_features1, upper_body_features2.T) / (np.linalg.norm(upper_body_features1) * np.linalg.norm(upper_body_features2))
#         lower_body_similarity = np.dot(lower_body_features1, lower_body_features2.T) / (np.linalg.norm(lower_body_features1) * np.linalg.norm(lower_body_features2))
#         upper_body_hist_similarity = cv2.compareHist(upper_body_histogram1, upper_body_histogram2, cv2.HISTCMP_CORREL)
#         lower_body_hist_similarity = cv2.compareHist(lower_body_histogram1, lower_body_histogram2, cv2.HISTCMP_CORREL)
        
#         combined_similarity = (0.2 * global_similarity +  
#                                 0.1 * upper_body_similarity +
#                                 0.4 * lower_body_similarity +
#                                 0.3 * lower_body_hist_similarity)
#         return combined_similarity


# def main(args=None):
#     rclpy.init(args=args)
#     node = AllInOneNode()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#      main()










# import collections
# if not hasattr(collections, 'Mapping'):
#     import collections.abc
#     collections.Mapping = collections.abc.Mapping
# import cv2
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# from geometry_msgs.msg import Twist, Point
# from cv_bridge import CvBridge
# import numpy as np
# import torch
# from fastreid.config import get_cfg
# from fastreid.engine import DefaultPredictor
# from fastreid.data.transforms import build_transforms
# from PIL import Image as PILImage
# from ultralytics import YOLO
# from message.msg import IdDistanceInfo
# import time


# # CUDA 디바이스 설정
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # YOLOv5 모델 로드
# model = YOLO("yolov5su.pt").to(device)

# # BotSORT 설정 파일 경로
# tracker_config = "/home/jinjuuk/.local/lib/python3.10/site-packages/ultralytics/cfg/trackers/botsort.yaml"

# # FastReID 설정
# cfg = get_cfg()
# cfg.merge_from_file("/home/jinjuuk/fast-reid/configs/Market1501/bagtricks_R50.yml")
# cfg.MODEL.WEIGHTS = "/home/jinjuuk/dev_ws/pt_files/market_mgn_R50-ibn.pth"
# cfg.MODEL.DEVICE = "cuda"  # GPU 사용 설정

# predictor = DefaultPredictor(cfg)
# transform = build_transforms(cfg, is_train=False)

# # 알려진 사람들의 특징 벡터와 ID를 저장할 딕셔너리
# known_persons = {}

# # 초기 설정값
# SIMILARITY_THRESHOLD = 0.97

# class AllInOneNode(Node):
#     def __init__(self):
#         super().__init__('all_in_one_node')
        
#         # 로그 출력: CUDA 사용 여부 확인
#         if torch.cuda.is_available():
#             cuda_device_name = torch.cuda.get_device_name(0)
#             self.get_logger().info(f"CUDA is available. Using GPU: {cuda_device_name}")
#         else:
#             self.get_logger().info("CUDA is not available. Using CPU")

#         # 구독자 설정
#         self.subscription_color = self.create_subscription(
#             Image,
#             '/camera/color/image_raw',
#             self.color_image_callback,
#             10
#         )
#         self.subscription_depth = self.create_subscription(
#             Image,
#             '/camera/depth/image_raw',
#             self.depth_image_callback,
#             10
#         )
        
#         # 발행자 설정
#         self.publisher_cmd_vel = self.create_publisher(
#             Twist,
#             'cmd_vel',
#             10
#         )
#         self.publisher_id_distance = self.create_publisher(
#             IdDistanceInfo,  
#             'id_distance_info',
#             10
#         )
        
#         # OpenCV를 위한 CVBridge 초기화
#         self.bridge = CvBridge()

#         # 트래킹 정보 저장
#         self.tracked_objects = {}  # id -> (bbox_center, bbox, depth)
#         self.id_last_seen = {}  # id -> last seen time
#         # 각 person_id에 대한 마지막 유효한 깊이 값을 저장하는 딕셔너리 초기화
#         self.last_valid_depth = {}
#         # 장애물까지의 가장 가까운 거리를 저장할 변수 초기화
#         self.closest_obstacle_distance = float('inf')

#         self.get_logger().info("AllInOneNode initialized")


#     def color_image_callback(self, msg):
#         self.get_logger().info("Received color image")
        
#         # 이미지 변환
#         frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
#         self.current_color_image = frame.copy()
#         self.get_logger().info("Color image converted to OpenCV format")
        
#         results = model.track(classes=[0], max_det=4, show=False, source=frame, stream=True, tracker=tracker_config)#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#         results = list(results)  # generator를 list로 변환
#         self.get_logger().info(f"YOLO tracking complete. Detected {len(results)} results")

#         active_ids = set()
#         current_time = self.get_clock().now()

#         for result in results:
#             boxes = result.boxes.xyxy.cpu().numpy().astype(int) if result.boxes is not None else []
#             track_ids = result.boxes.id.cpu().numpy().astype(int) if result.boxes.id is not None else []

#             for box, track_id in zip(boxes, track_ids):
#                 x1, y1, x2, y2 = box
#                 person_img = frame[y1:y2, x1:x2]

#                 global_features = self.extract_global_features(person_img)
#                 upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram = self.extract_local_features(person_img)

#                 max_similarity = -1
#                 max_similarity_id = -1

#                 for known_id, (known_global_features, known_upper_body_features, known_lower_body_features, known_upper_body_histogram, known_lower_body_histogram) in known_persons.items():
#                     similarity = self.calculate_combined_similarity(global_features, known_global_features, upper_body_features, known_upper_body_features, lower_body_features, known_lower_body_features, upper_body_histogram, known_upper_body_histogram, lower_body_histogram, known_lower_body_histogram)
#                     self.get_logger().info(f"Known_Id: {known_id}, {max_similarity_id},  Calculated similarity:  {similarity}, {max_similarity}")

#                     if similarity > max_similarity:
#                         max_similarity = similarity
#                         max_similarity_id = known_id

#                 if max_similarity > SIMILARITY_THRESHOLD:
#                     if max_similarity_id not in active_ids:
#                         person_id = max_similarity_id
#                         color = (0, 255, 0)
#                 else:
#                     person_id = int(track_id)
#                     if person_id in known_persons:
#                         pass
#                     else:
#                         new_id = max(known_persons.keys(), default=0) + 1
#                         person_id = new_id
#                     known_persons[person_id] = (global_features, upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram)
#                     color = (0, 0, 255)

#                 self.id_last_seen[person_id] = current_time
#                 active_ids.add(person_id)

#                 # 바운딩 박스의 중심 좌표 계산
#                 bbox_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
#                 self.tracked_objects[person_id] = (bbox_center, (x1, y1, x2, y2), None)  # depth_value는 아직 없음

#                 # 실시간으로 바운딩 박스 표시
#                 cv2.rectangle(self.current_color_image, (x1, y1), (x2, y2), color, 2)
#                 label = f"ID: {person_id}"
#                 cv2.putText(self.current_color_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

#                 # 추가적으로 센터 포인트 및 그 정보 표시
#                 center_x = (x1 + x2) // 2
#                 center_y = (y1 + y2) // 2
#                 centp = f"Center: ({center_x}, {center_y})"
#                 cv2.putText(self.current_color_image, centp, (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


#     def depth_image_callback(self, msg):
#         self.get_logger().info("Received depth image")
        
#         if not self.tracked_objects:
#             self.get_logger().info("No tracked objects available, skipping depth processing")
#             return

#         # 깊이 이미지를 OpenCV 형식으로 변환
#         depth_frame = self.bridge.imgmsg_to_cv2(msg, '16UC1')
#         self.get_logger().info("Depth image converted to OpenCV format")
        

#         # 장애물까지의 거리를 계산
#         self.closest_obstacle_distance = np.min(depth_frame) / 10.0  # 최소 깊이 값을 cm로 변환


#         active_ids = set()
#         current_time = self.get_clock().now()

#         for person_id, (bbox_center, bbox, _) in self.tracked_objects.items():
#             if bbox is None:
#                 self.get_logger().info(f"No bbox available for Person ID: {person_id}, skipping.")
#                 continue
            
#             x1, y1, x2, y2 = bbox
#             x, y = bbox_center
#             width = x2 - x1
#             height = y2 - y1

#             # 깊이 값이 0인 경우, 이전에 유효했던 깊이 값으로 대체하는 로직
#             if 0 <= x < depth_frame.shape[1] and 0 <= y < depth_frame.shape[0]:
#                 depth_value = round(depth_frame[y, x] / 10, 2)  # mm를 cm로 변환 후 소수점 2자리까지 표시
                
#                 # 깊이 값이 0인 경우
#                 if depth_value == 0:
#                     if person_id in self.last_valid_depth:
#                         depth_value = self.last_valid_depth[person_id]
#                         self.get_logger().info(f"Depth value was 0. Using last valid depth value: {depth_value}cm")
#                     else:
#                         self.get_logger().info(f"No valid depth value available for Person ID: {person_id}. Skipping.")
#                         continue
#                 else:
#                     # 깊이 값이 유효한 경우, 해당 값을 저장
#                     self.last_valid_depth[person_id] = depth_value
#             else:
#                 depth_value = None
#                 self.get_logger().info(f"Bbox center {bbox_center} is out of depth frame bounds.")

#             self.get_logger().info(f"Person ID: {person_id}, Bbox center: {bbox_center}, Width: {width}, Height: {height}, Depth: {depth_value}cm")


#             # 객체가 감지되었을 때 force_drive 함수 호출
#             if 1 in self.tracked_objects:  # ID 1번 객체가 추적 중일 때
#                 self.force_drive(id_to_follow=1)
#             else:
#                 self.get_logger().info("ID 1 is not being tracked. Not initiating force drive.")



#             # Update tracking information with depth value
#             self.tracked_objects[person_id] = (bbox_center, bbox, depth_value)

#             # Update the last seen time for the ID
#             self.id_last_seen[person_id] = current_time
#             active_ids.add(person_id)

#             # Define color for bounding box and text
#             color = (0, 255, 0)

#             depth_text = f"Depth: {depth_value}cm"
#             cv2.putText(self.current_color_image, depth_text, (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

#             # 커스텀 메시지 발행
#             self.publish_id_distance_info(person_id, bbox_center, width, height, depth_value)

#         # 실시간으로 처리된 이미지를 표시
#         cv2.imshow("Color Image with Tracking", self.current_color_image)
#         cv2.waitKey(1)
        



#     def publish_id_distance_info(self, id, bbox_center, width, height, depth_value):
#         self.get_logger().info(f"Preparing to publish IdDistanceInfo: ID={id}, Bbox Center={bbox_center}, Width={width}, Height={height}, Depth={depth_value}")
        
#         msg = IdDistanceInfo()
#         msg.id = id
#         msg.bbox_center = Point(x=float(bbox_center[0]), y=float(bbox_center[1]), z=0.0)
#         msg.width = float(width)
#         msg.height = float(height)
#         msg.distance = float(depth_value)
        
#         # 발행 전 메시지 정보 로그
#         self.get_logger().info(f"IdDistanceInfo message prepared: {msg}")
        
#         self.publisher_id_distance.publish(msg)
#         self.get_logger().info(f"Published IdDistanceInfo: ID={id}, Bbox Center={bbox_center}, bbox_Width={width}, bbox_Height={height}, Depth={depth_value}")


#     def extract_global_features(self, image):
#         pil_img = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#         input_img = transform(pil_img).unsqueeze(0).to(device)
#         features = predictor(input_img).cpu().numpy()
#         return features

#     def extract_local_features(self, image):
#         height, width, _ = image.shape
#         segments = [
#             image[:height//2, :],  
#             image[height//2:, :],  
#             image[:, :width//2],   
#             image[:, width//2:]    
#         ]
        
#         local_features = []
#         local_histograms = []
#         for segment in segments:
#             pil_img = PILImage.fromarray(cv2.cvtColor(segment, cv2.COLOR_BGR2RGB))
#             input_img = transform(pil_img).unsqueeze(0).to(device)
#             features = predictor(input_img).cpu().numpy()
#             hist = self.calculate_histogram(segment)
#             local_features.append(features)
#             local_histograms.append(hist)
        
#         upper_body_features = local_features[0]  
#         lower_body_features = local_features[1]  
        
#         upper_body_histogram = local_histograms[0]  
#         lower_body_histogram = local_histograms[1]  

#         return upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram

#     def calculate_histogram(self, image):
#         hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#         hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
#         cv2.normalize(hist, hist)
#         return hist.flatten()

#     def calculate_combined_similarity(self, global_features1, global_features2, upper_body_features1, upper_body_features2, lower_body_features1, lower_body_features2, upper_body_histogram1, upper_body_histogram2, lower_body_histogram1, lower_body_histogram2):
#         global_similarity = np.dot(global_features1, global_features2.T) / (np.linalg.norm(global_features1) * np.linalg.norm(global_features2))
#         upper_body_similarity = np.dot(upper_body_features1, upper_body_features2.T) / (np.linalg.norm(upper_body_features1) * np.linalg.norm(upper_body_features2))
#         lower_body_similarity = np.dot(lower_body_features1, lower_body_features2.T) / (np.linalg.norm(lower_body_features1) * np.linalg.norm(lower_body_features2))
#         upper_body_hist_similarity = cv2.compareHist(upper_body_histogram1, upper_body_histogram2, cv2.HISTCMP_CORREL)
#         lower_body_hist_similarity = cv2.compareHist(lower_body_histogram1, lower_body_histogram2, cv2.HISTCMP_CORREL)
        
#         combined_similarity = (0.2 * global_similarity +  
#                                 0.1 * upper_body_similarity +
#                                 0.4 * lower_body_similarity +
#                                 0.3 * lower_body_hist_similarity)
#         return combined_similarity


#     def force_drive(self, id_to_follow=1, duration=1.0, stop_distance=0.5):
#         """
#         로봇을 특정 방향으로 주행시키는 함수.

#         :param id_to_follow: 추종할 객체의 ID (기본값은 1)
#         :param duration: 주행 지속 시간 (초 단위)
#         :param stop_distance: 주행을 멈출 장애물까지의 거리 (미터 단위)
#         """

#         if id_to_follow not in self.tracked_objects:
#             self.get_logger().info(f'ID {id_to_follow} is not being tracked.')
#             return

#         bbox_center, _, depth_value = self.tracked_objects[id_to_follow]

#         if depth_value is None or depth_value == 0:
#             self.get_logger().info(f'No valid depth value for ID {id_to_follow}. Cannot proceed.')
#             return

#         camera_center_x = self.current_color_image.shape[1] // 2
#         angular_z = 0.0

#         # 중심 좌표가 카메라의 중심에서 얼마나 떨어져 있는지 계산
#         offset_x = bbox_center[0] - camera_center_x

#         # 회전 속도 비율을 조정할 최대값 설정 (예: 0.5로 설정)
#         max_angular_speed = 0.35

#         # 카메라 중심에서의 오프셋을 기준으로 회전 속도를 비례적으로 조정
#         angular_z = -max_angular_speed * (offset_x / camera_center_x)

#         # angular_z가 max_angular_speed보다 크거나 작지 않도록 클램핑
#         angular_z = max(-max_angular_speed, min(max_angular_speed, angular_z))

#         self.get_logger().info(f'Calculated angular_z: {angular_z:.2f} for offset: {offset_x}')





#         # # 장애물이 일정 거리 안에 있을 경우 멈춤
#         # if depth_value < stop_distance * 100:  # depth_value는 cm 단위이므로 100을 곱함
#         #     self.get_logger().info(f'Obstacle detected within {depth_value / 100:.2f} meters. Stopping force drive.')
#         #     return

#         twist_msg = Twist()
#         twist_msg.linear.x = 0.1  # 직진
#         twist_msg.angular.z = angular_z  # 회전 방향

#         self.publisher_cmd_vel.publish(twist_msg)
#         time.sleep(0.1)


#         # end_time = self.get_clock().now().seconds_nanoseconds()[0] + duration

#         # while self.get_clock().now().seconds_nanoseconds()[0] < end_time:
#         #     if self.closest_obstacle_distance < stop_distance:
#         #         self.get_logger().info(f'Obstacle detected within {self.closest_obstacle_distance:.2f} meters. Stopping force drive.')
#         #         break  # 장애물이 감지되면 주행 중지

#             # self.publisher_cmd_vel.publish(twist_msg)
#             # time.sleep(0.1)

#         # 주행 종료를 위해 속도를 0으로 설정
#         #twist_msg.linear.x = 0.0
#         #twist_msg.angular.z = 0.0
#         #self.publisher_cmd_vel.publish(twist_msg)


# def main(args=None):
#     rclpy.init(args=args)
#     node = AllInOneNode()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()


























# import collections
# if not hasattr(collections, 'Mapping'):
#     import collections.abc
#     collections.Mapping = collections.abc.Mapping
# import cv2
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# from geometry_msgs.msg import Twist, Point
# from cv_bridge import CvBridge
# import numpy as np
# import torch
# from fastreid.config import get_cfg
# from fastreid.engine import DefaultPredictor
# from fastreid.data.transforms import build_transforms
# from PIL import Image as PILImage
# from ultralytics import YOLO
# from message.msg import IdDistanceInfo
# import time


# # CUDA 디바이스 설정
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # YOLOv5 모델 로드
# model = YOLO("yolov5su.pt").to(device)

# # BotSORT 설정 파일 경로
# tracker_config = "/home/jinjuuk/.local/lib/python3.10/site-packages/ultralytics/cfg/trackers/botsort.yaml"

# # FastReID 설정
# cfg = get_cfg()
# cfg.merge_from_file("/home/jinjuuk/fast-reid/configs/Market1501/bagtricks_R50.yml")
# cfg.MODEL.WEIGHTS = "/home/jinjuuk/dev_ws/pt_files/market_mgn_R50-ibn.pth"
# cfg.MODEL.DEVICE = "cuda"  # GPU 사용 설정

# predictor = DefaultPredictor(cfg)
# transform = build_transforms(cfg, is_train=False)

# # 알려진 사람들의 특징 벡터와 ID를 저장할 딕셔너리
# known_persons = {}

# # 초기 설정값
# SIMILARITY_THRESHOLD = 0.97

# class AllInOneNode(Node):
#     def __init__(self):
#         super().__init__('all_in_one_node')
        
#         # 로그 출력: CUDA 사용 여부 확인
#         if torch.cuda.is_available():
#             cuda_device_name = torch.cuda.get_device_name(0)
#             self.get_logger().info(f"CUDA is available. Using GPU: {cuda_device_name}")
#         else:
#             self.get_logger().info("CUDA is not available. Using CPU")

#         # 구독자 설정
#         self.subscription_color = self.create_subscription(
#             Image,
#             '/camera/color/image_raw',
#             self.color_image_callback,
#             10
#         )
#         self.subscription_depth = self.create_subscription(
#             Image,
#             '/camera/depth/image_raw',
#             self.depth_image_callback,
#             10
#         )
        
#         # 발행자 설정
#         self.publisher_cmd_vel = self.create_publisher(
#             Twist,
#             'cmd_vel',
#             10
#         )
#         self.publisher_id_distance = self.create_publisher(
#             IdDistanceInfo,  
#             'id_distance_info',
#             10
#         )
        
#         # OpenCV를 위한 CVBridge 초기화
#         self.bridge = CvBridge()

#         # 트래킹 정보 저장
#         self.tracked_objects = {}  # id -> (bbox_center, bbox, depth)
#         self.id_last_seen = {}  # id -> last seen time
#         # 각 person_id에 대한 마지막 유효한 깊이 값을 저장하는 딕셔너리 초기화
#         self.last_valid_depth = {}
#         # 장애물까지의 가장 가까운 거리를 저장할 변수 초기화
#         self.closest_obstacle_distance = float('inf')

#         self.get_logger().info("AllInOneNode initialized")


#     # Function to generate a unique color based on ID
#     def get_unique_color(self, id):
#         np.random.seed(id)  # Seed with the ID for consistency
#         return tuple(np.random.randint(0, 255, 3).tolist())

#     def color_image_callback(self, msg):
#         self.get_logger().info("Received color image")

#         # 이미지 변환
#         frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
#         self.current_color_image = frame.copy()
#         self.get_logger().info("Color image converted to OpenCV format")

#         results = model.track(classes=[0], max_det=4, show=False, source=frame, stream=True, tracker=tracker_config)
#         results = list(results)  # generator를 list로 변환
#         self.get_logger().info(f"YOLO tracking complete. Detected {len(results)} results")

#         active_ids = set()
#         current_time = self.get_clock().now()

#         for result in results:
#             boxes = result.boxes.xyxy.cpu().numpy().astype(int) if result.boxes is not None else []
#             track_ids = result.boxes.id.cpu().numpy().astype(int) if result.boxes.id is not None else []

#             for box, track_id in zip(boxes, track_ids):
#                 x1, y1, x2, y2 = box
#                 person_img = frame[y1:y2, x1:x2]

#                 global_features = self.extract_global_features(person_img)
#                 upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram = self.extract_local_features(person_img)

#                 max_similarity = -1
#                 max_similarity_id = -1

#                 for known_id, (known_global_features, known_upper_body_features, known_lower_body_features, known_upper_body_histogram, known_lower_body_histogram) in known_persons.items():
#                     similarity = self.calculate_combined_similarity(global_features, known_global_features, upper_body_features, known_upper_body_features, lower_body_features, known_lower_body_features, upper_body_histogram, known_upper_body_histogram, lower_body_histogram, known_lower_body_histogram)
#                     self.get_logger().info(f"Known_Id: {known_id}, {max_similarity_id},  Calculated similarity:  {similarity}, {max_similarity}")

#                     if similarity > max_similarity:
#                         max_similarity = similarity
#                         max_similarity_id = known_id

#                 if max_similarity > SIMILARITY_THRESHOLD:
#                     if max_similarity_id not in active_ids:
#                         person_id = max_similarity_id
#                 else:
#                     person_id = int(track_id)
#                     if person_id not in known_persons:
#                         new_id = max(known_persons.keys(), default=0) + 1
#                         person_id = new_id
#                     known_persons[person_id] = (global_features, upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram)

#                 self.id_last_seen[person_id] = current_time
#                 active_ids.add(person_id)

#                 # 바운딩 박스의 중심 좌표 계산
#                 bbox_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
#                 self.tracked_objects[person_id] = (bbox_center, (x1, y1, x2, y2), None)  # depth_value는 아직 없음

#                 # Get unique color for the person ID
#                 color = self.get_unique_color(person_id)

#                 # 실시간으로 바운딩 박스 표시
#                 cv2.rectangle(self.current_color_image, (x1, y1), (x2, y2), color, 2)
#                 label = f"ID: {person_id}"
#                 cv2.putText(self.current_color_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

#                 # 추가적으로 센터 포인트 및 그 정보 표시
#                 center_x = (x1 + x2) // 2
#                 center_y = (y1 + y2) // 2
#                 centp = f"Center: ({center_x}, {center_y})"
#                 cv2.putText(self.current_color_image, centp, (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

#     def depth_image_callback(self, msg):
#         self.get_logger().info("Received depth image")
        
#         if not self.tracked_objects:
#             self.get_logger().info("No tracked objects available, skipping depth processing")
#             return

#         # 깊이 이미지를 OpenCV 형식으로 변환
#         depth_frame = self.bridge.imgmsg_to_cv2(msg, '16UC1')
#         self.get_logger().info("Depth image converted to OpenCV format")

#         # 장애물까지의 거리를 계산
#         self.closest_obstacle_distance = np.min(depth_frame) / 10.0  # 최소 깊이 값을 cm로 변환

#         active_ids = set()
#         current_time = self.get_clock().now()

#         for person_id, (bbox_center, bbox, _) in self.tracked_objects.items():
#             if bbox is None:
#                 self.get_logger().info(f"No bbox available for Person ID: {person_id}, skipping.")
#                 continue
            
#             x1, y1, x2, y2 = bbox
#             x, y = bbox_center
#             width = x2 - x1
#             height = y2 - y1

#             # 깊이 값이 0인 경우, 이전에 유효했던 깊이 값으로 대체하는 로직
#             if 0 <= x < depth_frame.shape[1] and 0 <= y < depth_frame.shape[0]:
#                 depth_value = round(depth_frame[y, x] / 10, 2)  # mm를 cm로 변환 후 소수점 2자리까지 표시
                
#                 # 깊이 값이 0인 경우
#                 if depth_value == 0:
#                     if person_id in self.last_valid_depth:
#                         depth_value = self.last_valid_depth[person_id]
#                         self.get_logger().info(f"Depth value was 0. Using last valid depth value: {depth_value}cm")
#                     else:
#                         self.get_logger().info(f"No valid depth value available for Person ID: {person_id}. Skipping.")
#                         continue
#                 else:
#                     # 깊이 값이 유효한 경우, 해당 값을 저장
#                     self.last_valid_depth[person_id] = depth_value
#             else:
#                 depth_value = None
#                 self.get_logger().info(f"Bbox center {bbox_center} is out of depth frame bounds.")

#             self.get_logger().info(f"Person ID: {person_id}, Bbox center: {bbox_center}, Width: {width}, Height: {height}, Depth: {depth_value}cm")

#             # 객체가 감지되었을 때 force_drive 함수 호출
#             if 1 in self.tracked_objects:  # ID 1번 객체가 추적 중일 때
#                 self.force_drive(id_to_follow=1)
#             else:
#                 self.get_logger().info("ID 1 is not being tracked. Not initiating force drive.")

#             # Update tracking information with depth value
#             self.tracked_objects[person_id] = (bbox_center, bbox, depth_value)

#             # Update the last seen time for the ID
#             self.id_last_seen[person_id] = current_time
#             active_ids.add(person_id)

#             # Get unique color for the person ID
#             color = self.get_unique_color(person_id)

#             depth_text = f"Depth: {depth_value}cm"
#             cv2.putText(self.current_color_image, depth_text, (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

#             # 커스텀 메시지 발행
#             self.publish_id_distance_info(person_id, bbox_center, width, height, depth_value)

#         # 실시간으로 처리된 이미지를 표시
#         cv2.imshow("Color Image with Tracking", self.current_color_image)
#         cv2.waitKey(1)
        



#     def publish_id_distance_info(self, id, bbox_center, width, height, depth_value):
#         self.get_logger().info(f"Preparing to publish IdDistanceInfo: ID={id}, Bbox Center={bbox_center}, Width={width}, Height={height}, Depth={depth_value}")
        
#         msg = IdDistanceInfo()
#         msg.id = id
#         msg.bbox_center = Point(x=float(bbox_center[0]), y=float(bbox_center[1]), z=0.0)
#         msg.width = float(width)
#         msg.height = float(height)
#         msg.distance = float(depth_value)
        
#         # 발행 전 메시지 정보 로그
#         self.get_logger().info(f"IdDistanceInfo message prepared: {msg}")
        
#         self.publisher_id_distance.publish(msg)
#         self.get_logger().info(f"Published IdDistanceInfo: ID={id}, Bbox Center={bbox_center}, bbox_Width={width}, bbox_Height={height}, Depth={depth_value}")


#     def extract_global_features(self, image):
#         pil_img = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#         input_img = transform(pil_img).unsqueeze(0).to(device)
#         features = predictor(input_img).cpu().numpy()
#         return features

#     def extract_local_features(self, image):
#         height, width, _ = image.shape
#         segments = [
#             image[:height//2, :],  
#             image[height//2:, :],  
#             image[:, :width//2],   
#             image[:, width//2:]    
#         ]
        
#         local_features = []
#         local_histograms = []
#         for segment in segments:
#             pil_img = PILImage.fromarray(cv2.cvtColor(segment, cv2.COLOR_BGR2RGB))
#             input_img = transform(pil_img).unsqueeze(0).to(device)
#             features = predictor(input_img).cpu().numpy()
#             hist = self.calculate_histogram(segment)
#             local_features.append(features)
#             local_histograms.append(hist)
        
#         upper_body_features = local_features[0]  
#         lower_body_features = local_features[1]  
        
#         upper_body_histogram = local_histograms[0]  
#         lower_body_histogram = local_histograms[1]  

#         return upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram

#     def calculate_histogram(self, image):
#         hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#         hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
#         cv2.normalize(hist, hist)
#         return hist.flatten()

#     def calculate_combined_similarity(self, global_features1, global_features2, upper_body_features1, upper_body_features2, lower_body_features1, lower_body_features2, upper_body_histogram1, upper_body_histogram2, lower_body_histogram1, lower_body_histogram2):
#         global_similarity = np.dot(global_features1, global_features2.T) / (np.linalg.norm(global_features1) * np.linalg.norm(global_features2))
#         upper_body_similarity = np.dot(upper_body_features1, upper_body_features2.T) / (np.linalg.norm(upper_body_features1) * np.linalg.norm(upper_body_features2))
#         lower_body_similarity = np.dot(lower_body_features1, lower_body_features2.T) / (np.linalg.norm(lower_body_features1) * np.linalg.norm(lower_body_features2))
#         upper_body_hist_similarity = cv2.compareHist(upper_body_histogram1, upper_body_histogram2, cv2.HISTCMP_CORREL)
#         lower_body_hist_similarity = cv2.compareHist(lower_body_histogram1, lower_body_histogram2, cv2.HISTCMP_CORREL)
        
#         combined_similarity = (0.2 * global_similarity +  
#                                 0.1 * upper_body_similarity +
#                                 0.4 * lower_body_similarity +
#                                 0.3 * lower_body_hist_similarity)
#         return combined_similarity


#     def force_drive(self, id_to_follow=1, duration=1.0, stop_distance=0.5):
#         """
#         로봇을 특정 방향으로 주행시키는 함수.

#         :param id_to_follow: 추종할 객체의 ID (기본값은 1)
#         :param duration: 주행 지속 시간 (초 단위)
#         :param stop_distance: 주행을 멈출 장애물까지의 거리 (미터 단위)
#         """

#         if id_to_follow not in self.tracked_objects:
#             self.get_logger().info(f'ID {id_to_follow} is not being tracked.')
#             return

#         bbox_center, _, depth_value = self.tracked_objects[id_to_follow]

#         if depth_value is None or depth_value == 0:
#             self.get_logger().info(f'No valid depth value for ID {id_to_follow}. Cannot proceed.')
#             return

#         camera_center_x = self.current_color_image.shape[1] // 2
#         angular_z = 0.0

#         # 중심 좌표가 카메라의 중심에서 얼마나 떨어져 있는지 계산
#         offset_x = bbox_center[0] - camera_center_x

#         # 회전 속도 비율을 조정할 최대값 설정 (예: 0.5로 설정)
#         max_angular_speed = 0.35

#         # 카메라 중심에서의 오프셋을 기준으로 회전 속도를 비례적으로 조정
#         angular_z = -max_angular_speed * (offset_x / camera_center_x)

#         # angular_z가 max_angular_speed보다 크거나 작지 않도록 클램핑
#         angular_z = max(-max_angular_speed, min(max_angular_speed, angular_z))

#         self.get_logger().info(f'Calculated angular_z: {angular_z:.2f} for offset: {offset_x}')





#         # # 장애물이 일정 거리 안에 있을 경우 멈춤
#         # if depth_value < stop_distance * 100:  # depth_value는 cm 단위이므로 100을 곱함
#         #     self.get_logger().info(f'Obstacle detected within {depth_value / 100:.2f} meters. Stopping force drive.')
#         #     return

#         twist_msg = Twist()
#         twist_msg.linear.x = 0.1  # 직진
#         twist_msg.angular.z = angular_z  # 회전 방향

#         self.publisher_cmd_vel.publish(twist_msg)
#         time.sleep(0.1)


#         # end_time = self.get_clock().now().seconds_nanoseconds()[0] + duration

#         # while self.get_clock().now().seconds_nanoseconds()[0] < end_time:
#         #     if self.closest_obstacle_distance < stop_distance:
#         #         self.get_logger().info(f'Obstacle detected within {self.closest_obstacle_distance:.2f} meters. Stopping force drive.')
#         #         break  # 장애물이 감지되면 주행 중지

#             # self.publisher_cmd_vel.publish(twist_msg)
#             # time.sleep(0.1)

#         # 주행 종료를 위해 속도를 0으로 설정
#         #twist_msg.linear.x = 0.0
#         #twist_msg.angular.z = 0.0
#         #self.publisher_cmd_vel.publish(twist_msg)


# def main(args=None):
#     rclpy.init(args=args)
#     node = AllInOneNode()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()







# import collections
# if not hasattr(collections, 'Mapping'):
#     import collections.abc
#     collections.Mapping = collections.abc.Mapping
# import cv2
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# from geometry_msgs.msg import Twist, Point
# from cv_bridge import CvBridge
# import numpy as np
# import torch
# from fastreid.config import get_cfg
# from fastreid.engine import DefaultPredictor
# from fastreid.data.transforms import build_transforms
# from PIL import Image as PILImage
# from ultralytics import YOLO
# from message.msg import IdDistanceInfo
# import time


# # CUDA 디바이스 설정
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # YOLOv5 모델 로드
# model = YOLO("yolov5su.pt").to(device)

# # BotSORT 설정 파일 경로
# tracker_config = "/home/jinjuuk/.local/lib/python3.10/site-packages/ultralytics/cfg/trackers/botsort.yaml"

# # FastReID 설정
# cfg = get_cfg()
# cfg.merge_from_file("/home/jinjuuk/fast-reid/configs/Market1501/bagtricks_R50.yml")
# cfg.MODEL.WEIGHTS = "/home/jinjuuk/dev_ws/pt_files/market_mgn_R50-ibn.pth"
# cfg.MODEL.DEVICE = "cuda"  # GPU 사용 설정

# predictor = DefaultPredictor(cfg)
# transform = build_transforms(cfg, is_train=False)

# # 알려진 사람들의 특징 벡터와 ID를 저장할 딕셔너리
# known_persons = {}

# # 초기 설정값
# SIMILARITY_THRESHOLD = 0.97

# class AllInOneNode(Node):
#     def __init__(self):
#         super().__init__('all_in_one_node')
        
#         # 로그 출력: CUDA 사용 여부 확인
#         if torch.cuda.is_available():
#             cuda_device_name = torch.cuda.get_device_name(0)
#             self.get_logger().info(f"CUDA is available. Using GPU: {cuda_device_name}")
#         else:
#             self.get_logger().info("CUDA is not available. Using CPU")

#         # 구독자 설정
#         self.subscription_color = self.create_subscription(
#             Image,
#             '/camera/color/image_raw',
#             self.color_image_callback,
#             10
#         )
#         self.subscription_depth = self.create_subscription(
#             Image,
#             '/camera/depth/image_raw',
#             self.depth_image_callback,
#             10
#         )
        
#         # 발행자 설정
#         self.publisher_cmd_vel = self.create_publisher(
#             Twist,
#             'cmd_vel',
#             10
#         )
#         self.publisher_id_distance = self.create_publisher(
#             IdDistanceInfo,  
#             'id_distance_info',
#             10
#         )
        
#         # OpenCV를 위한 CVBridge 초기화
#         self.bridge = CvBridge()

#         # 트래킹 정보 저장
#         self.tracked_objects = {}  # id -> (bbox_center, bbox, depth)
#         self.id_last_seen = {}  # id -> last seen time
#         # 각 person_id에 대한 마지막 유효한 깊이 값을 저장하는 딕셔너리 초기화
#         self.last_valid_depth = {}
#         # 장애물까지의 가장 가까운 거리를 저장할 변수 초기화
#         self.closest_obstacle_distance = float('inf')

#         self.get_logger().info("AllInOneNode initialized")


#     # Function to generate a unique color based on ID
#     def get_unique_color(self, id):
#         np.random.seed(id)  # Seed with the ID for consistency
#         return tuple(np.random.randint(0, 255, 3).tolist())

#     def color_image_callback(self, msg):
#         self.get_logger().info("Received color image")

#         # 이미지 변환
#         frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
#         self.current_color_image = frame.copy()
#         self.get_logger().info("Color image converted to OpenCV format")

#         results = model.track(classes=[0], max_det=4, show=False, source=frame, stream=True, tracker=tracker_config)
#         results = list(results)  # generator를 list로 변환
#         self.get_logger().info(f"YOLO tracking complete. Detected {len(results)} results")

#         active_ids = set()
#         current_time = self.get_clock().now()

#         for result in results:
#             boxes = result.boxes.xyxy.cpu().numpy().astype(int) if result.boxes is not None else []
#             track_ids = result.boxes.id.cpu().numpy().astype(int) if result.boxes.id is not None else []

#             for box, track_id in zip(boxes, track_ids):
#                 x1, y1, x2, y2 = box
#                 person_img = frame[y1:y2, x1:x2]

#                 global_features = self.extract_global_features(person_img)
#                 upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram = self.extract_local_features(person_img)

#                 max_similarity = -1
#                 max_similarity_id = -1

#                 for known_id, (known_global_features, known_upper_body_features, known_lower_body_features, known_upper_body_histogram, known_lower_body_histogram) in known_persons.items():
#                     similarity = self.calculate_combined_similarity(
#                         global_features,
#                         known_global_features,
#                         upper_body_features,
#                         known_upper_body_features,
#                         lower_body_features,
#                         known_lower_body_features,
#                         upper_body_histogram,
#                         known_upper_body_histogram,
#                         lower_body_histogram,
#                         known_lower_body_histogram,
#                     )
#                     self.get_logger().info(f"Known_Id: {known_id}, {max_similarity_id}, Calculated similarity: {similarity}, {max_similarity}")

#                     if similarity > max_similarity:
#                         max_similarity = similarity
#                         max_similarity_id = known_id

#                 if max_similarity > SIMILARITY_THRESHOLD:
#                     if max_similarity_id not in active_ids:
#                         person_id = max_similarity_id
#                 else:
#                     person_id = int(track_id)
#                     if person_id not in known_persons:
#                         new_id = max(known_persons.keys(), default=0) + 1
#                         person_id = new_id
#                     known_persons[person_id] = (global_features, upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram)

#                 self.id_last_seen[person_id] = current_time
#                 active_ids.add(person_id)

#                 # 바운딩 박스의 중심 좌표 계산
#                 bbox_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
#                 self.tracked_objects[person_id] = (bbox_center, (x1, y1, x2, y2), None)  # depth_value는 아직 없음

#                 # Get unique color for the person ID
#                 color = self.get_unique_color(person_id)

#                 # 실시간으로 바운딩 박스 표시
#                 cv2.rectangle(self.current_color_image, (x1, y1), (x2, y2), color, 2)
#                 label = f"ID: {person_id}"

#                 # Set text color to white
#                 text_color = (255, 255, 255)

#                 # Calculate text size for background rectangle
#                 (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)

#                 # Draw a filled rectangle as background for the text, matching the bounding box color
#                 cv2.rectangle(self.current_color_image, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, thickness=cv2.FILLED)

#                 # Draw the text on top of the rectangle
#                 cv2.putText(self.current_color_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

#                 # Additional center point and information display
#                 center_x = (x1 + x2) // 2
#                 center_y = (y1 + y2) // 2
#                 centp = f"Center: ({center_x}, {center_y})"
#                 cv2.putText(self.current_color_image, centp, (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

#     def depth_image_callback(self, msg):
#         self.get_logger().info("Received depth image")
        
#         if not self.tracked_objects:
#             self.get_logger().info("No tracked objects available, skipping depth processing")
#             return

#         # 깊이 이미지를 OpenCV 형식으로 변환
#         depth_frame = self.bridge.imgmsg_to_cv2(msg, '16UC1')
#         self.get_logger().info("Depth image converted to OpenCV format")

#         # 장애물까지의 거리를 계산
#         self.closest_obstacle_distance = np.min(depth_frame) / 10.0  # 최소 깊이 값을 cm로 변환

#         active_ids = set()
#         current_time = self.get_clock().now()

#         for person_id, (bbox_center, bbox, _) in self.tracked_objects.items():
#             if bbox is None:
#                 self.get_logger().info(f"No bbox available for Person ID: {person_id}, skipping.")
#                 continue
            
#             x1, y1, x2, y2 = bbox
#             x, y = bbox_center
#             width = x2 - x1
#             height = y2 - y1

#             # 깊이 값이 0인 경우, 이전에 유효했던 깊이 값으로 대체하는 로직
#             if 0 <= x < depth_frame.shape[1] and 0 <= y < depth_frame.shape[0]:
#                 depth_value = round(depth_frame[y, x] / 10, 2)  # mm를 cm로 변환 후 소수점 2자리까지 표시
                
#                 # 깊이 값이 0인 경우
#                 if depth_value == 0:
#                     if person_id in self.last_valid_depth:
#                         depth_value = self.last_valid_depth[person_id]
#                         self.get_logger().info(f"Depth value was 0. Using last valid depth value: {depth_value}cm")
#                     else:
#                         self.get_logger().info(f"No valid depth value available for Person ID: {person_id}. Skipping.")
#                         continue
#                 else:
#                     # 깊이 값이 유효한 경우, 해당 값을 저장
#                     self.last_valid_depth[person_id] = depth_value
#             else:
#                 depth_value = None
#                 self.get_logger().info(f"Bbox center {bbox_center} is out of depth frame bounds.")

#             self.get_logger().info(f"Person ID: {person_id}, Bbox center: {bbox_center}, Width: {width}, Height: {height}, Depth: {depth_value}cm")

#             # 객체가 감지되었을 때 force_drive 함수 호출
#             if 1 in self.tracked_objects:  # ID 1번 객체가 추적 중일 때
#                 self.force_drive(id_to_follow=1)
#             else:
#                 self.get_logger().info("ID 1 is not being tracked. Not initiating force drive.")

#             # Update tracking information with depth value
#             self.tracked_objects[person_id] = (bbox_center, bbox, depth_value)

#             # Update the last seen time for the ID
#             self.id_last_seen[person_id] = current_time
#             active_ids.add(person_id)

#             # Get unique color for the person ID
#             color = self.get_unique_color(person_id)

#             # Set text color to white
#             text_color = (255, 255, 255)

#             depth_text = f"Depth: {depth_value}cm"
#             cv2.putText(self.current_color_image, depth_text, (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

#             # 커스텀 메시지 발행
#             self.publish_id_distance_info(person_id, bbox_center, width, height, depth_value)

#         # 실시간으로 처리된 이미지를 표시
#         cv2.imshow("Color Image with Tracking", self.current_color_image)
#         cv2.waitKey(1)




#     def publish_id_distance_info(self, id, bbox_center, width, height, depth_value):
#         self.get_logger().info(f"Preparing to publish IdDistanceInfo: ID={id}, Bbox Center={bbox_center}, Width={width}, Height={height}, Depth={depth_value}")
        
#         msg = IdDistanceInfo()
#         msg.id = id
#         msg.bbox_center = Point(x=float(bbox_center[0]), y=float(bbox_center[1]), z=0.0)
#         msg.width = float(width)
#         msg.height = float(height)
#         msg.distance = float(depth_value)
        
#         # 발행 전 메시지 정보 로그
#         self.get_logger().info(f"IdDistanceInfo message prepared: {msg}")
        
#         self.publisher_id_distance.publish(msg)
#         self.get_logger().info(f"Published IdDistanceInfo: ID={id}, Bbox Center={bbox_center}, bbox_Width={width}, bbox_Height={height}, Depth={depth_value}")


#     def extract_global_features(self, image):
#         pil_img = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#         input_img = transform(pil_img).unsqueeze(0).to(device)
#         features = predictor(input_img).cpu().numpy()
#         return features

#     def extract_local_features(self, image):
#         height, width, _ = image.shape
#         segments = [
#             image[:height//2, :],  
#             image[height//2:, :],  
#             image[:, :width//2],   
#             image[:, width//2:]    
#         ]
        
#         local_features = []
#         local_histograms = []
#         for segment in segments:
#             pil_img = PILImage.fromarray(cv2.cvtColor(segment, cv2.COLOR_BGR2RGB))
#             input_img = transform(pil_img).unsqueeze(0).to(device)
#             features = predictor(input_img).cpu().numpy()
#             hist = self.calculate_histogram(segment)
#             local_features.append(features)
#             local_histograms.append(hist)
        
#         upper_body_features = local_features[0]  
#         lower_body_features = local_features[1]  
        
#         upper_body_histogram = local_histograms[0]  
#         lower_body_histogram = local_histograms[1]  

#         return upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram

#     def calculate_histogram(self, image):
#         hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#         hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
#         cv2.normalize(hist, hist)
#         return hist.flatten()

#     def calculate_combined_similarity(self, global_features1, global_features2, upper_body_features1, upper_body_features2, lower_body_features1, lower_body_features2, upper_body_histogram1, upper_body_histogram2, lower_body_histogram1, lower_body_histogram2):
#         global_similarity = np.dot(global_features1, global_features2.T) / (np.linalg.norm(global_features1) * np.linalg.norm(global_features2))
#         upper_body_similarity = np.dot(upper_body_features1, upper_body_features2.T) / (np.linalg.norm(upper_body_features1) * np.linalg.norm(upper_body_features2))
#         lower_body_similarity = np.dot(lower_body_features1, lower_body_features2.T) / (np.linalg.norm(lower_body_features1) * np.linalg.norm(lower_body_features2))
#         upper_body_hist_similarity = cv2.compareHist(upper_body_histogram1, upper_body_histogram2, cv2.HISTCMP_CORREL)
#         lower_body_hist_similarity = cv2.compareHist(lower_body_histogram1, lower_body_histogram2, cv2.HISTCMP_CORREL)
        
#         combined_similarity = (0.2 * global_similarity +  
#                                 0.1 * upper_body_similarity +
#                                 0.4 * lower_body_similarity +
#                                 0.3 * lower_body_hist_similarity)
#         return combined_similarity


#     def force_drive(self, id_to_follow=1, duration=1.0, stop_distance=0.5):
#         """
#         로봇을 특정 방향으로 주행시키는 함수.

#         :param id_to_follow: 추종할 객체의 ID (기본값은 1)
#         :param duration: 주행 지속 시간 (초 단위)
#         :param stop_distance: 주행을 멈출 장애물까지의 거리 (미터 단위)
#         """

#         if id_to_follow not in self.tracked_objects:
#             self.get_logger().info(f'ID {id_to_follow} is not being tracked.')
#             return

#         bbox_center, _, depth_value = self.tracked_objects[id_to_follow]

#         if depth_value is None or depth_value == 0:
#             self.get_logger().info(f'No valid depth value for ID {id_to_follow}. Cannot proceed.')
#             return

#         camera_center_x = self.current_color_image.shape[1] // 2
#         angular_z = 0.0

#         # 중심 좌표가 카메라의 중심에서 얼마나 떨어져 있는지 계산
#         offset_x = bbox_center[0] - camera_center_x

#         # 회전 속도 비율을 조정할 최대값 설정 (예: 0.5로 설정)
#         max_angular_speed = 0.35

#         # 카메라 중심에서의 오프셋을 기준으로 회전 속도를 비례적으로 조정
#         angular_z = -max_angular_speed * (offset_x / camera_center_x)

#         # angular_z가 max_angular_speed보다 크거나 작지 않도록 클램핑
#         angular_z = max(-max_angular_speed, min(max_angular_speed, angular_z))

#         self.get_logger().info(f'Calculated angular_z: {angular_z:.2f} for offset: {offset_x}')





#         # # 장애물이 일정 거리 안에 있을 경우 멈춤
#         # if depth_value < stop_distance * 100:  # depth_value는 cm 단위이므로 100을 곱함
#         #     self.get_logger().info(f'Obstacle detected within {depth_value / 100:.2f} meters. Stopping force drive.')
#         #     return

#         twist_msg = Twist()
#         twist_msg.linear.x = 0.1  # 직진
#         twist_msg.angular.z = angular_z  # 회전 방향

#         self.publisher_cmd_vel.publish(twist_msg)
#         time.sleep(0.1)


#         # end_time = self.get_clock().now().seconds_nanoseconds()[0] + duration

#         # while self.get_clock().now().seconds_nanoseconds()[0] < end_time:
#         #     if self.closest_obstacle_distance < stop_distance:
#         #         self.get_logger().info(f'Obstacle detected within {self.closest_obstacle_distance:.2f} meters. Stopping force drive.')
#         #         break  # 장애물이 감지되면 주행 중지

#             # self.publisher_cmd_vel.publish(twist_msg)
#             # time.sleep(0.1)

#         # 주행 종료를 위해 속도를 0으로 설정
#         #twist_msg.linear.x = 0.0
#         #twist_msg.angular.z = 0.0
#         #self.publisher_cmd_vel.publish(twist_msg)


# def main(args=None):
#     rclpy.init(args=args)
#     node = AllInOneNode()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()








# import collections
# if not hasattr(collections, 'Mapping'):
#     import collections.abc
#     collections.Mapping = collections.abc.Mapping
# import cv2
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# from geometry_msgs.msg import Twist, Point
# from cv_bridge import CvBridge
# import numpy as np
# import torch
# from fastreid.config import get_cfg
# from fastreid.engine import DefaultPredictor
# from fastreid.data.transforms import build_transforms
# from PIL import Image as PILImage
# from ultralytics import YOLO
# from message.msg import IdDistanceInfo
# import time


# # CUDA 디바이스 설정
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # YOLOv5 모델 로드
# model = YOLO("yolov8m.pt").to(device)

# # BotSORT 설정 파일 경로
# tracker_config = "/home/jinjuuk/.local/lib/python3.10/site-packages/ultralytics/cfg/trackers/botsort.yaml"

# # FastReID 설정
# cfg = get_cfg()
# cfg.merge_from_file("/home/jinjuuk/fast-reid/configs/Market1501/bagtricks_R50.yml")
# cfg.MODEL.WEIGHTS = "/home/jinjuuk/dev_ws/pt_files/market_mgn_R50-ibn.pth"
# cfg.MODEL.DEVICE = "cuda"  # GPU 사용 설정

# predictor = DefaultPredictor(cfg)
# transform = build_transforms(cfg, is_train=False)

# # 알려진 사람들의 특징 벡터와 ID를 저장할 딕셔너리
# known_persons = {}

# # 초기 설정값
# SIMILARITY_THRESHOLD = 0.97

# class AllInOneNode(Node):
#     def __init__(self):
#         super().__init__('all_in_one_node')
        
#         # 로그 출력: CUDA 사용 여부 확인
#         if torch.cuda.is_available():
#             cuda_device_name = torch.cuda.get_device_name(0)
#             self.get_logger().info(f"CUDA is available. Using GPU: {cuda_device_name}")
#         else:
#             self.get_logger().info("CUDA is not available. Using CPU")

#         # 구독자 설정
#         self.subscription_color = self.create_subscription(
#             Image,
#             '/camera/color/image_raw',
#             self.color_image_callback,
#             10
#         )
#         self.subscription_depth = self.create_subscription(
#             Image,
#             '/camera/depth/image_raw',
#             self.depth_image_callback,
#             10
#         )
        
#         # 발행자 설정
#         self.publisher_cmd_vel = self.create_publisher(
#             Twist,
#             'cmd_vel',
#             10
#         )
#         self.publisher_id_distance = self.create_publisher(
#             IdDistanceInfo,  
#             'id_distance_info',
#             10
#         )
        
#         # OpenCV를 위한 CVBridge 초기화
#         self.bridge = CvBridge()

#         # 트래킹 정보 저장
#         self.tracked_objects = {}  # id -> (bbox_center, bbox, depth)
#         self.id_last_seen = {}  # id -> last seen time
#         # 각 person_id에 대한 마지막 유효한 깊이 값을 저장하는 딕셔너리 초기화
#         self.last_valid_depth = {}
#         # 장애물까지의 가장 가까운 거리를 저장할 변수 초기화
#         self.closest_obstacle_distance = float('inf')

#         self.get_logger().info("AllInOneNode initialized")


#     # Function to generate a unique color based on ID
#     def get_unique_color(self, id):
#         np.random.seed(id)  # Seed with the ID for consistency
#         return tuple(np.random.randint(0, 255, 3).tolist())

#     def color_image_callback(self, msg):
#         self.get_logger().info("Received color image")

#         # 이미지 변환
#         frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
#         self.current_color_image = frame.copy()
#         self.get_logger().info("Color image converted to OpenCV format")

#         results = model.track(classes=[0], max_det=4, show=False, source=frame, stream=True, tracker=tracker_config)
#         results = list(results)  # generator를 list로 변환
#         self.get_logger().info(f"YOLO tracking complete. Detected {len(results)} results")

#         active_ids = set()
#         current_time = self.get_clock().now()

#         for result in results:
#             boxes = result.boxes.xyxy.cpu().numpy().astype(int) if result.boxes is not None else []
#             track_ids = result.boxes.id.cpu().numpy().astype(int) if result.boxes.id is not None else []

#             for box, track_id in zip(boxes, track_ids):
#                 x1, y1, x2, y2 = box
#                 person_img = frame[y1:y2, x1:x2]

#                 global_features = self.extract_global_features(person_img)
#                 upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram = self.extract_local_features(person_img)

#                 max_similarity = -1
#                 max_similarity_id = -1

#                 for known_id, (known_global_features, known_upper_body_features, known_lower_body_features, known_upper_body_histogram, known_lower_body_histogram) in known_persons.items():
#                     similarity = self.calculate_combined_similarity(
#                         global_features,
#                         known_global_features,
#                         upper_body_features,
#                         known_upper_body_features,
#                         lower_body_features,
#                         known_lower_body_features,
#                         upper_body_histogram,
#                         known_upper_body_histogram,
#                         lower_body_histogram,
#                         known_lower_body_histogram,
#                     )
#                     self.get_logger().info(f"Known_Id: {known_id}, {max_similarity_id}, Calculated similarity: {similarity}, {max_similarity}")

#                     if similarity > max_similarity:
#                         max_similarity = similarity
#                         max_similarity_id = known_id

#                 if max_similarity > SIMILARITY_THRESHOLD:
#                     if max_similarity_id not in active_ids:
#                         person_id = max_similarity_id
#                 else:
#                     person_id = int(track_id)
#                     if person_id not in known_persons:
#                         new_id = max(known_persons.keys(), default=0) + 1
#                         person_id = new_id
#                     known_persons[person_id] = (global_features, upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram)

#                 self.id_last_seen[person_id] = current_time
#                 active_ids.add(person_id)

#                 # 바운딩 박스의 중심 좌표 계산
#                 bbox_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
#                 self.tracked_objects[person_id] = (bbox_center, (x1, y1, x2, y2), None)  # depth_value는 아직 없음

#                 # Get unique color for the person ID
#                 color = self.get_unique_color(person_id)

#                 # 실시간으로 바운딩 박스 표시
#                 cv2.rectangle(self.current_color_image, (x1, y1), (x2, y2), color, 2)
#                 label = f"ID: {person_id}"

#                 # Set text color to white
#                 text_color = (255, 255, 255)

#                 # Calculate text size for background rectangle
#                 (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)

#                 # Draw a filled rectangle as background for the ID text, matching the bounding box color
#                 cv2.rectangle(self.current_color_image, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, thickness=cv2.FILLED)

#                 # Draw the ID text on top of the rectangle
#                 cv2.putText(self.current_color_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

#                 # Additional center point and information display
#                 center_x = (x1 + x2) // 2
#                 center_y = (y1 + y2) // 2
#                 centp = f"Center: ({center_x}, {center_y})"

#                 # Calculate text size for center point background
#                 (center_text_width, center_text_height), baseline = cv2.getTextSize(centp, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)

#                 # Draw a filled rectangle as background for the center point text, matching the bounding box color
#                 cv2.rectangle(self.current_color_image, (x1, y1 - center_text_height - 45), (x1 + center_text_width, y1 - 35), color, thickness=cv2.FILLED)

#                 # Draw the center point text
#                 cv2.putText(self.current_color_image, centp, (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

#     def depth_image_callback(self, msg):
#         self.get_logger().info("Received depth image")
        
#         if not self.tracked_objects:
#             self.get_logger().info("No tracked objects available, skipping depth processing")
#             return

#         # 깊이 이미지를 OpenCV 형식으로 변환
#         depth_frame = self.bridge.imgmsg_to_cv2(msg, '16UC1')
#         self.get_logger().info("Depth image converted to OpenCV format")

#         # 장애물까지의 거리를 계산
#         self.closest_obstacle_distance = np.min(depth_frame) / 10.0  # 최소 깊이 값을 cm로 변환

#         active_ids = set()
#         current_time = self.get_clock().now()

#         for person_id, (bbox_center, bbox, _) in self.tracked_objects.items():
#             if bbox is None:
#                 self.get_logger().info(f"No bbox available for Person ID: {person_id}, skipping.")
#                 continue
            
#             x1, y1, x2, y2 = bbox
#             x, y = bbox_center
#             width = x2 - x1
#             height = y2 - y1

#             # 깊이 값이 0인 경우, 이전에 유효했던 깊이 값으로 대체하는 로직
#             if 0 <= x < depth_frame.shape[1] and 0 <= y < depth_frame.shape[0]:
#                 depth_value = round(depth_frame[y, x] / 10, 2)  # mm를 cm로 변환 후 소수점 2자리까지 표시
                
#                 # 깊이 값이 0인 경우
#                 if depth_value == 0:
#                     if person_id in self.last_valid_depth:
#                         depth_value = self.last_valid_depth[person_id]
#                         self.get_logger().info(f"Depth value was 0. Using last valid depth value: {depth_value}cm")
#                     else:
#                         self.get_logger().info(f"No valid depth value available for Person ID: {person_id}. Skipping.")
#                         continue
#                 else:
#                     # 깊이 값이 유효한 경우, 해당 값을 저장
#                     self.last_valid_depth[person_id] = depth_value
#             else:
#                 depth_value = None
#                 self.get_logger().info(f"Bbox center {bbox_center} is out of depth frame bounds.")

#             self.get_logger().info(f"Person ID: {person_id}, Bbox center: {bbox_center}, Width: {width}, Height: {height}, Depth: {depth_value}cm")

#             # 객체가 감지되었을 때 force_drive 함수 호출
#             if 1 in self.tracked_objects:  # ID 1번 객체가 추적 중일 때
#                 self.force_drive(id_to_follow=1)
#             else:
#                 self.get_logger().info("ID 1 is not being tracked. Not initiating force drive.")

#             # Update tracking information with depth value
#             self.tracked_objects[person_id] = (bbox_center, bbox, depth_value)

#             # Update the last seen time for the ID
#             self.id_last_seen[person_id] = current_time
#             active_ids.add(person_id)

#             # Get unique color for the person ID
#             color = self.get_unique_color(person_id)

#             # Set text color to white
#             text_color = (255, 255, 255)

#             depth_text = f"Depth: {depth_value}cm"

#             # Calculate text size for depth text background
#             (depth_text_width, depth_text_height), baseline = cv2.getTextSize(depth_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)

#             # Draw a filled rectangle as background for the depth text, matching the bounding box color
#             cv2.rectangle(self.current_color_image, (x1, y1 + 10), (x1 + depth_text_width, y1 + depth_text_height + 20), color, thickness=cv2.FILLED)

#             # Draw the depth text
#             cv2.putText(self.current_color_image, depth_text, (x1, y1 + depth_text_height + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

#             # 커스텀 메시지 발행
#             self.publish_id_distance_info(person_id, bbox_center, width, height, depth_value)

#         # 실시간으로 처리된 이미지를 표시
#         cv2.imshow("Color Image with Tracking", self.current_color_image)
#         cv2.waitKey(1)




#     def publish_id_distance_info(self, id, bbox_center, width, height, depth_value):
#         self.get_logger().info(f"Preparing to publish IdDistanceInfo: ID={id}, Bbox Center={bbox_center}, Width={width}, Height={height}, Depth={depth_value}")
        
#         msg = IdDistanceInfo()
#         msg.id = id
#         msg.bbox_center = Point(x=float(bbox_center[0]), y=float(bbox_center[1]), z=0.0)
#         msg.width = float(width)
#         msg.height = float(height)
#         msg.distance = float(depth_value)
        
#         # 발행 전 메시지 정보 로그
#         self.get_logger().info(f"IdDistanceInfo message prepared: {msg}")
        
#         self.publisher_id_distance.publish(msg)
#         self.get_logger().info(f"Published IdDistanceInfo: ID={id}, Bbox Center={bbox_center}, bbox_Width={width}, bbox_Height={height}, Depth={depth_value}")


#     def extract_global_features(self, image):
#         pil_img = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#         input_img = transform(pil_img).unsqueeze(0).to(device)
#         features = predictor(input_img).cpu().numpy()
#         return features

#     def extract_local_features(self, image):
#         height, width, _ = image.shape
#         segments = [
#             image[:height//2, :],  
#             image[height//2:, :],  
#             image[:, :width//2],   
#             image[:, width//2:]    
#         ]
        
#         local_features = []
#         local_histograms = []
#         for segment in segments:
#             pil_img = PILImage.fromarray(cv2.cvtColor(segment, cv2.COLOR_BGR2RGB))
#             input_img = transform(pil_img).unsqueeze(0).to(device)
#             features = predictor(input_img).cpu().numpy()
#             hist = self.calculate_histogram(segment)
#             local_features.append(features)
#             local_histograms.append(hist)
        
#         upper_body_features = local_features[0]  
#         lower_body_features = local_features[1]  
        
#         upper_body_histogram = local_histograms[0]  
#         lower_body_histogram = local_histograms[1]  

#         return upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram

#     def calculate_histogram(self, image):
#         hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#         hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
#         cv2.normalize(hist, hist)
#         return hist.flatten()

#     def calculate_combined_similarity(self, global_features1, global_features2, upper_body_features1, upper_body_features2, lower_body_features1, lower_body_features2, upper_body_histogram1, upper_body_histogram2, lower_body_histogram1, lower_body_histogram2):
#         global_similarity = np.dot(global_features1, global_features2.T) / (np.linalg.norm(global_features1) * np.linalg.norm(global_features2))
#         upper_body_similarity = np.dot(upper_body_features1, upper_body_features2.T) / (np.linalg.norm(upper_body_features1) * np.linalg.norm(upper_body_features2))
#         lower_body_similarity = np.dot(lower_body_features1, lower_body_features2.T) / (np.linalg.norm(lower_body_features1) * np.linalg.norm(lower_body_features2))
#         upper_body_hist_similarity = cv2.compareHist(upper_body_histogram1, upper_body_histogram2, cv2.HISTCMP_CORREL)
#         lower_body_hist_similarity = cv2.compareHist(lower_body_histogram1, lower_body_histogram2, cv2.HISTCMP_CORREL)
        
#         combined_similarity = (0.2 * global_similarity +  
#                                 0.1 * upper_body_similarity +
#                                 0.4 * lower_body_similarity +
#                                 0.3 * lower_body_hist_similarity)
#         return combined_similarity


#     def force_drive(self, id_to_follow=1, duration=1.0, stop_distance=0.5):
#         """
#         로봇을 특정 방향으로 주행시키는 함수.

#         :param id_to_follow: 추종할 객체의 ID (기본값은 1)
#         :param duration: 주행 지속 시간 (초 단위)
#         :param stop_distance: 주행을 멈출 장애물까지의 거리 (미터 단위)
#         """

#         if id_to_follow not in self.tracked_objects:
#             self.get_logger().info(f'ID {id_to_follow} is not being tracked.')
#             return

#         bbox_center, _, depth_value = self.tracked_objects[id_to_follow]

#         if depth_value is None or depth_value == 0:
#             self.get_logger().info(f'No valid depth value for ID {id_to_follow}. Cannot proceed.')
#             return

#         camera_center_x = self.current_color_image.shape[1] // 2
#         angular_z = 0.0

#         # 중심 좌표가 카메라의 중심에서 얼마나 떨어져 있는지 계산
#         offset_x = bbox_center[0] - camera_center_x

#         # 회전 속도 비율을 조정할 최대값 설정 (예: 0.5로 설정)
#         max_angular_speed = 0.35

#         # 카메라 중심에서의 오프셋을 기준으로 회전 속도를 비례적으로 조정
#         angular_z = -max_angular_speed * (offset_x / camera_center_x)

#         # angular_z가 max_angular_speed보다 크거나 작지 않도록 클램핑
#         angular_z = max(-max_angular_speed, min(max_angular_speed, angular_z))

#         self.get_logger().info(f'Calculated angular_z: {angular_z:.2f} for offset: {offset_x}')





#         # # 장애물이 일정 거리 안에 있을 경우 멈춤
#         # if depth_value < stop_distance * 100:  # depth_value는 cm 단위이므로 100을 곱함
#         #     self.get_logger().info(f'Obstacle detected within {depth_value / 100:.2f} meters. Stopping force drive.')
#         #     return

#         twist_msg = Twist()
#         twist_msg.linear.x = 0.1  # 직진
#         twist_msg.angular.z = angular_z  # 회전 방향

#         self.publisher_cmd_vel.publish(twist_msg)
#         time.sleep(0.1)


#         # end_time = self.get_clock().now().seconds_nanoseconds()[0] + duration

#         # while self.get_clock().now().seconds_nanoseconds()[0] < end_time:
#         #     if self.closest_obstacle_distance < stop_distance:
#         #         self.get_logger().info(f'Obstacle detected within {self.closest_obstacle_distance:.2f} meters. Stopping force drive.')
#         #         break  # 장애물이 감지되면 주행 중지

#             # self.publisher_cmd_vel.publish(twist_msg)
#             # time.sleep(0.1)

#         # 주행 종료를 위해 속도를 0으로 설정
#         #twist_msg.linear.x = 0.0
#         #twist_msg.angular.z = 0.0
#         #self.publisher_cmd_vel.publish(twist_msg)


# def main(args=None):
#     rclpy.init(args=args)
#     node = AllInOneNode()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()











# import collections
# if not hasattr(collections, 'Mapping'):
#     import collections.abc
#     collections.Mapping = collections.abc.Mapping
# import cv2
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# from geometry_msgs.msg import Twist, Point
# from cv_bridge import CvBridge
# import numpy as np
# import torch
# from fastreid.config import get_cfg
# from fastreid.engine import DefaultPredictor
# from fastreid.data.transforms import build_transforms
# from PIL import Image as PILImage
# from ultralytics import YOLO
# from message.msg import IdDistanceInfo
# import time


# # CUDA 디바이스 설정
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # YOLOv5 모델 로드
# model = YOLO("yolov5su.pt").to(device)

# # BotSORT 설정 파일 경로
# tracker_config = "/home/jinjuuk/.local/lib/python3.10/site-packages/ultralytics/cfg/trackers/botsort.yaml"

# # FastReID 설정
# cfg = get_cfg()
# cfg.merge_from_file("/home/jinjuuk/fast-reid/configs/Market1501/bagtricks_R50.yml")
# cfg.MODEL.WEIGHTS = "/home/jinjuuk/dev_ws/pt_files/market_mgn_R50-ibn.pth"
# cfg.MODEL.DEVICE = "cuda"  # GPU 사용 설정

# predictor = DefaultPredictor(cfg)
# transform = build_transforms(cfg, is_train=False)

# # 알려진 사람들의 특징 벡터와 ID를 저장할 딕셔너리
# known_persons = {}

# # 초기 설정값
# SIMILARITY_THRESHOLD = 0.97

# class AllInOneNode(Node):
#     def __init__(self):
#         super().__init__('all_in_one_node')
        
#         # 로그 출력: CUDA 사용 여부 확인
#         if torch.cuda.is_available():
#             cuda_device_name = torch.cuda.get_device_name(0)
#             self.get_logger().info(f"CUDA is available. Using GPU: {cuda_device_name}")
#         else:
#             self.get_logger().info("CUDA is not available. Using CPU")

#         # 구독자 설정
#         self.subscription_color = self.create_subscription(
#             Image,
#             '/camera/color/image_raw',
#             self.color_image_callback,
#             10
#         )
#         self.subscription_depth = self.create_subscription(
#             Image,
#             '/camera/depth/image_raw',
#             self.depth_image_callback,
#             10
#         )
        
#         # 발행자 설정
#         self.publisher_cmd_vel = self.create_publisher(
#             Twist,
#             'cmd_vel',
#             10
#         )
#         self.publisher_id_distance = self.create_publisher(
#             IdDistanceInfo,  
#             'id_distance_info',
#             10
#         )
        
#         # OpenCV를 위한 CVBridge 초기화
#         self.bridge = CvBridge()

#         # 트래킹 정보 저장
#         self.tracked_objects = {}  # id -> (bbox_center, bbox, depth)
#         self.id_last_seen = {}  # id -> last seen time
#         # 각 person_id에 대한 마지막 유효한 깊이 값을 저장하는 딕셔너리 초기화
#         self.last_valid_depth = {}
#         # 장애물까지의 가장 가까운 거리를 저장할 변수 초기화
#         self.closest_obstacle_distance = float('inf')

#         self.get_logger().info("AllInOneNode initialized")


#     # Function to generate a unique color based on ID
#     def get_unique_color(self, id):
#         np.random.seed(id)  # Seed with the ID for consistency
#         return tuple(np.random.randint(0, 255, 3).tolist())

#     def color_image_callback(self, msg):
#         self.get_logger().info("Received color image")

#         # 이미지 변환
#         frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
#         self.current_color_image = frame.copy()
#         self.get_logger().info("Color image converted to OpenCV format")

#         results = model.track(classes=[0], max_det=4, show=False, source=frame, stream=True, tracker=tracker_config)
#         results = list(results)  # generator를 list로 변환
#         self.get_logger().info(f"YOLO tracking complete. Detected {len(results)} results")

#         active_ids = set()
#         current_time = self.get_clock().now()

#         for result in results:
#             boxes = result.boxes.xyxy.cpu().numpy().astype(int) if result.boxes is not None else []
#             track_ids = result.boxes.id.cpu().numpy().astype(int) if result.boxes.id is not None else []

#             for box, track_id in zip(boxes, track_ids):
#                 x1, y1, x2, y2 = box
#                 person_img = frame[y1:y2, x1:x2]

#                 global_features = self.extract_global_features(person_img)
#                 upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram = self.extract_local_features(person_img)

#                 max_similarity = -1
#                 max_similarity_id = -1

#                 for known_id, (known_global_features, known_upper_body_features, known_lower_body_features, known_upper_body_histogram, known_lower_body_histogram) in known_persons.items():
#                     similarity = self.calculate_combined_similarity(
#                         global_features,
#                         known_global_features,
#                         upper_body_features,
#                         known_upper_body_features,
#                         lower_body_features,
#                         known_lower_body_features,
#                         upper_body_histogram,
#                         known_upper_body_histogram,
#                         lower_body_histogram,
#                         known_lower_body_histogram,
#                     )
#                     self.get_logger().info(f"Known_Id: {known_id}, {max_similarity_id}, Calculated similarity: {similarity}, {max_similarity}")

#                     if similarity > max_similarity:
#                         max_similarity = similarity
#                         max_similarity_id = known_id

#                 if max_similarity > SIMILARITY_THRESHOLD:
#                     if max_similarity_id not in active_ids:
#                         person_id = max_similarity_id
#                 else:
#                     person_id = int(track_id)
#                     if person_id not in known_persons:
#                         new_id = max(known_persons.keys(), default=0) + 1
#                         person_id = new_id
#                     known_persons[person_id] = (global_features, upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram)

#                 self.id_last_seen[person_id] = current_time
#                 active_ids.add(person_id)

#                 # 바운딩 박스의 중심 좌표 계산
#                 bbox_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
#                 self.tracked_objects[person_id] = (bbox_center, (x1, y1, x2, y2), None)  # depth_value는 아직 없음

#                 # Get unique color for the person ID
#                 color = self.get_unique_color(person_id)

#                 # 실시간으로 바운딩 박스 표시
#                 cv2.rectangle(self.current_color_image, (x1, y1), (x2, y2), color, 2)
#                 label = f"ID: {person_id}"

#                 # Set text color to white
#                 text_color = (255, 255, 255)

#                 # Calculate text size for background rectangle for ID
#                 (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)

#                 # Draw a filled rectangle as background for the ID text, matching the bounding box color
#                 cv2.rectangle(self.current_color_image, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, thickness=cv2.FILLED)

#                 # Draw the ID text on top of the rectangle
#                 cv2.putText(self.current_color_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

#                 # Additional center point and information display
#                 center_x = (x1 + x2) // 2
#                 center_y = (y1 + y2) // 2
#                 centp = f"Center: ({center_x}, {center_y})"

#                 # Calculate text size for center point background
#                 (center_text_width, center_text_height), baseline = cv2.getTextSize(centp, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)

#                 # Draw a filled rectangle as background for the center point text, below the ID text
#                 cv2.rectangle(self.current_color_image, (x1, y1 + 5), (x1 + center_text_width, y1 + center_text_height + 15), color, thickness=cv2.FILLED)

#                 # Draw the center point text
#                 cv2.putText(self.current_color_image, centp, (x1, y1 + center_text_height + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

#     def depth_image_callback(self, msg):
#         self.get_logger().info("Received depth image")
        
#         if not self.tracked_objects:
#             self.get_logger().info("No tracked objects available, skipping depth processing")
#             return

#         # Convert depth image to OpenCV format
#         depth_frame = self.bridge.imgmsg_to_cv2(msg, '16UC1')
#         self.get_logger().info("Depth image converted to OpenCV format")

#         # Calculate distance to the closest obstacle
#         self.closest_obstacle_distance = np.min(depth_frame) / 10.0  # Convert minimum depth value to cm

#         active_ids = set()
#         current_time = self.get_clock().now()

#         for person_id, (bbox_center, bbox, _) in self.tracked_objects.items():
#             if bbox is None:
#                 self.get_logger().info(f"No bbox available for Person ID: {person_id}, skipping.")
#                 continue
            
#             x1, y1, x2, y2 = bbox
#             x, y = bbox_center
#             width = x2 - x1
#             height = y2 - y1

#             # Handling depth values, using last valid if current is zero
#             if 0 <= x < depth_frame.shape[1] and 0 <= y < depth_frame.shape[0]:
#                 depth_value = round(depth_frame[y, x] / 10, 2)  # Convert mm to cm with two decimal places
                
#                 if depth_value == 0:
#                     if person_id in self.last_valid_depth:
#                         depth_value = self.last_valid_depth[person_id]
#                         self.get_logger().info(f"Depth value was 0. Using last valid depth value: {depth_value}cm")
#                     else:
#                         self.get_logger().info(f"No valid depth value available for Person ID: {person_id}. Skipping.")
#                         continue
#                 else:
#                     self.last_valid_depth[person_id] = depth_value
#             else:
#                 if person_id in self.last_valid_depth:
#                     depth_value = self.last_valid_depth[person_id]
#                     self.get_logger().info(f"Depth value was None. Using last valid depth value: {depth_value}cm")
#                 else:
#                     depth_value = None
#                     self.get_logger().info(f"Bbox center {bbox_center} is out of depth frame bounds and no previous depth value available.")


#             self.get_logger().info(f"Person ID: {person_id}, Bbox center: {bbox_center}, Width: {width}, Height: {height}, Depth: {depth_value}cm")

#             # Handle object detection and following
#             if 1 in self.tracked_objects:
#                 self.force_drive(id_to_follow=1)
#             else:
#                 self.get_logger().info("ID 1 is not being tracked. Not initiating force drive.")

#             # Update tracking information with depth value
#             self.tracked_objects[person_id] = (bbox_center, bbox, depth_value)

#             # Update the last seen time for the ID
#             self.id_last_seen[person_id] = current_time
#             active_ids.add(person_id)

#             # Get unique color for the person ID
#             color = self.get_unique_color(person_id)

#             # Set text color to white
#             text_color = (255, 255, 255)

#             # Calculate text for center point display, assuming you need this in depth_image_callback
#             centp = f"Center: ({x}, {y})"
#             (center_text_width, center_text_height), _ = cv2.getTextSize(centp, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)

#             depth_text = f"Depth: {depth_value}cm"
#             (depth_text_width, depth_text_height), _ = cv2.getTextSize(depth_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)

#             # Adjust the drawing position if needed and draw the rectangle and text
#             cv2.rectangle(self.current_color_image, (x1, y1 + center_text_height + 25), (x1 + depth_text_width, y1 + center_text_height + depth_text_height + 35), color, thickness=cv2.FILLED)

#             # Draw the depth text
#             cv2.putText(self.current_color_image, depth_text, (x1, y1 + center_text_height + depth_text_height + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

#             # Publish custom message
#             self.publish_id_distance_info(person_id, bbox_center, width, height, depth_value)

#         # Display the processed image in real-time
#         cv2.imshow("Color Image with Tracking", self.current_color_image)
#         cv2.waitKey(1)




#     def publish_id_distance_info(self, id, bbox_center, width, height, depth_value):
#         self.get_logger().info(f"Preparing to publish IdDistanceInfo: ID={id}, Bbox Center={bbox_center}, Width={width}, Height={height}, Depth={depth_value}")
        
#         msg = IdDistanceInfo()
#         msg.id = id
#         msg.bbox_center = Point(x=float(bbox_center[0]), y=float(bbox_center[1]), z=0.0)
#         msg.width = float(width)
#         msg.height = float(height)
#         msg.distance = float(depth_value)
        
#         # 발행 전 메시지 정보 로그
#         self.get_logger().info(f"IdDistanceInfo message prepared: {msg}")
        
#         self.publisher_id_distance.publish(msg)
#         self.get_logger().info(f"Published IdDistanceInfo: ID={id}, Bbox Center={bbox_center}, bbox_Width={width}, bbox_Height={height}, Depth={depth_value}")


#     def extract_global_features(self, image):
#         pil_img = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#         input_img = transform(pil_img).unsqueeze(0).to(device)
#         features = predictor(input_img).cpu().numpy()
#         return features

#     def extract_local_features(self, image):
#         height, width, _ = image.shape
#         segments = [
#             image[:height//2, :],  
#             image[height//2:, :],  
#             image[:, :width//2],   
#             image[:, width//2:]    
#         ]
        
#         local_features = []
#         local_histograms = []
#         for segment in segments:
#             pil_img = PILImage.fromarray(cv2.cvtColor(segment, cv2.COLOR_BGR2RGB))
#             input_img = transform(pil_img).unsqueeze(0).to(device)
#             features = predictor(input_img).cpu().numpy()
#             hist = self.calculate_histogram(segment)
#             local_features.append(features)
#             local_histograms.append(hist)
        
#         upper_body_features = local_features[0]  
#         lower_body_features = local_features[1]  
        
#         upper_body_histogram = local_histograms[0]  
#         lower_body_histogram = local_histograms[1]  

#         return upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram

#     def calculate_histogram(self, image):
#         hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#         hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
#         cv2.normalize(hist, hist)
#         return hist.flatten()

#     def calculate_combined_similarity(self, global_features1, global_features2, upper_body_features1, upper_body_features2, lower_body_features1, lower_body_features2, upper_body_histogram1, upper_body_histogram2, lower_body_histogram1, lower_body_histogram2):
#         global_similarity = np.dot(global_features1, global_features2.T) / (np.linalg.norm(global_features1) * np.linalg.norm(global_features2))
#         upper_body_similarity = np.dot(upper_body_features1, upper_body_features2.T) / (np.linalg.norm(upper_body_features1) * np.linalg.norm(upper_body_features2))
#         lower_body_similarity = np.dot(lower_body_features1, lower_body_features2.T) / (np.linalg.norm(lower_body_features1) * np.linalg.norm(lower_body_features2))
#         upper_body_hist_similarity = cv2.compareHist(upper_body_histogram1, upper_body_histogram2, cv2.HISTCMP_CORREL)
#         lower_body_hist_similarity = cv2.compareHist(lower_body_histogram1, lower_body_histogram2, cv2.HISTCMP_CORREL)
        
#         combined_similarity = (0.2 * global_similarity +  
#                                 0.1 * upper_body_similarity +
#                                 0.4 * lower_body_similarity +
#                                 0.3 * lower_body_hist_similarity)
#         return combined_similarity


#     def force_drive(self, id_to_follow=1, duration=1.0, stop_distance=1): # stop_distance = 0.5는 50cm
#         """
#         로봇을 특정 방향으로 주행시키는 함수.

#         :param id_to_follow: 추종할 객체의 ID (기본값은 1)
#         :param duration: 주행 지속 시간 (초 단위)
#         :param stop_distance: 주행을 멈출 장애물까지의 거리 (미터 단위)
#         """

#         if id_to_follow not in self.tracked_objects:
#             self.get_logger().info(f'ID {id_to_follow} is not being tracked.')
#             return

#         bbox_center, _, depth_value = self.tracked_objects[id_to_follow]

#         if depth_value is None or depth_value == 0:
#             self.get_logger().info(f'No valid depth value for ID {id_to_follow}. Cannot proceed.')
#             return

#         camera_center_x = self.current_color_image.shape[1] // 2
#         angular_z = 0.0

#         # 중심 좌표가 카메라의 중심에서 얼마나 떨어져 있는지 계산
#         offset_x = bbox_center[0] - camera_center_x

#         # 회전 속도 비율을 조정할 최대값 설정 (예: 0.5로 설정)
#         max_angular_speed = 0.35

#         # 카메라 중심에서의 오프셋을 기준으로 회전 속도를 비례적으로 조정
#         angular_z = -max_angular_speed * (offset_x / camera_center_x)

#         # angular_z가 max_angular_speed보다 크거나 작지 않도록 클램핑
#         angular_z = max(-max_angular_speed, min(max_angular_speed, angular_z))

#         self.get_logger().info(f'Calculated angular_z: {angular_z:.2f} for offset: {offset_x}')


#         twist_msg = Twist()


#         # 장애물이 일정 거리 안에 있을 경우 멈춤
#         if depth_value < stop_distance * 100:  # depth_value는 cm 단위이므로 100을 곱함
#             self.get_logger().info(f'Safty distance detected within {depth_value / 100:.2f} meters. Stopping force drive.')
#             twist_msg.linear.x = 0.0
#             twist_msg.angular.z = 0.0
#             self.publisher_cmd_vel.publish(twist_msg)
#             time.sleep(0.1)

#         else:
#             safty_distance_diff =  depth_value - (stop_distance * 100)
#             self.get_logger().info(f'Object distance: {depth_value}cm,  Distance between Object and Safty_Zone: {safty_distance_diff}cm')
#             twist_msg.linear.x = 0.1  # 직진
#             twist_msg.angular.z = angular_z  # 회전 방향
#             self.publisher_cmd_vel.publish(twist_msg)
#             time.sleep(0.1)
#             return 



# def main(args=None):
#     rclpy.init(args=args)
#     node = AllInOneNode()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()













# import collections
# if not hasattr(collections, 'Mapping'):
#     import collections.abc
#     collections.Mapping = collections.abc.Mapping
# import cv2
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# from geometry_msgs.msg import Twist, Point
# from cv_bridge import CvBridge
# import numpy as np
# import torch
# from fastreid.config import get_cfg
# from fastreid.engine import DefaultPredictor
# from fastreid.data.transforms import build_transforms
# from PIL import Image as PILImage
# from ultralytics import YOLO
# from message.msg import IdDistanceInfo
# import time


# # CUDA 디바이스 설정
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # YOLOv5 모델 로드
# model = YOLO("yolov5su.pt").to(device)

# # BotSORT 설정 파일 경로
# tracker_config = "/home/jinjuuk/.local/lib/python3.10/site-packages/ultralytics/cfg/trackers/botsort.yaml"

# # FastReID 설정
# cfg = get_cfg()
# cfg.merge_from_file("/home/jinjuuk/fast-reid/configs/Market1501/bagtricks_R50.yml")
# cfg.MODEL.WEIGHTS = "/home/jinjuuk/dev_ws/pt_files/market_mgn_R50-ibn.pth"
# cfg.MODEL.DEVICE = "cuda"  # GPU 사용 설정

# predictor = DefaultPredictor(cfg)
# transform = build_transforms(cfg, is_train=False)

# # 알려진 사람들의 특징 벡터와 ID를 저장할 딕셔너리
# known_persons = {}

# # 초기 설정값
# SIMILARITY_THRESHOLD = 0.97

# class AllInOneNode(Node):
#     def __init__(self):
#         super().__init__('all_in_one_node')
        
#         # 로그 출력: CUDA 사용 여부 확인
#         if torch.cuda.is_available():
#             cuda_device_name = torch.cuda.get_device_name(0)
#             self.get_logger().info(f"CUDA is available. Using GPU: {cuda_device_name}")
#         else:
#             self.get_logger().info("CUDA is not available. Using CPU")

#         # 구독자 설정
#         self.subscription_color = self.create_subscription(
#             Image,
#             '/camera/color/image_raw',
#             self.color_image_callback,
#             10
#         )
#         self.subscription_depth = self.create_subscription(
#             Image,
#             '/camera/depth/image_raw',
#             self.depth_image_callback,
#             10
#         )
        
#         # 발행자 설정
#         self.publisher_cmd_vel = self.create_publisher(
#             Twist,
#             'cmd_vel',
#             10
#         )
#         self.publisher_id_distance = self.create_publisher(
#             IdDistanceInfo,  
#             'id_distance_info',
#             10
#         )
        
#         # OpenCV를 위한 CVBridge 초기화
#         self.bridge = CvBridge()

#         # 트래킹 정보 저장
#         self.tracked_objects = {}  # id -> (bbox_center, bbox, depth)
#         self.id_last_seen = {}  # id -> last seen time
#         # 각 person_id에 대한 마지막 유효한 깊이 값을 저장하는 딕셔너리 초기화
#         self.last_valid_depth = {}
#         # 장애물까지의 가장 가까운 거리를 저장할 변수 초기화
#         self.closest_obstacle_distance = float('inf')

#         self.get_logger().info("AllInOneNode initialized")


#     # Function to generate a unique color based on ID
#     def get_unique_color(self, id):
#         np.random.seed(id)  # Seed with the ID for consistency
#         return tuple(np.random.randint(0, 255, 3).tolist())

#     def color_image_callback(self, msg):
#         self.get_logger().info("Received color image")

#         # 이미지 변환
#         frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
#         self.current_color_image = frame.copy()
#         self.get_logger().info("Color image converted to OpenCV format")

#         results = model.track(classes=[0], max_det=4, show=False, source=frame, stream=True, tracker=tracker_config)
#         results = list(results)  # generator를 list로 변환
#         self.get_logger().info(f"YOLO tracking complete. Detected {len(results)} results")

#         active_ids = set()
#         current_time = self.get_clock().now()

#         for result in results:
#             boxes = result.boxes.xyxy.cpu().numpy().astype(int) if result.boxes is not None else []
#             track_ids = result.boxes.id.cpu().numpy().astype(int) if result.boxes.id is not None else []

#             for box, track_id in zip(boxes, track_ids):
#                 x1, y1, x2, y2 = box
#                 person_img = frame[y1:y2, x1:x2]

#                 global_features = self.extract_global_features(person_img)
#                 upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram = self.extract_local_features(person_img)

#                 max_similarity = -1
#                 max_similarity_id = -1

#                 for known_id, (known_global_features, known_upper_body_features, known_lower_body_features, known_upper_body_histogram, known_lower_body_histogram) in known_persons.items():
#                     similarity = self.calculate_combined_similarity(
#                         global_features,
#                         known_global_features,
#                         upper_body_features,
#                         known_upper_body_features,
#                         lower_body_features,
#                         known_lower_body_features,
#                         upper_body_histogram,
#                         known_upper_body_histogram,
#                         lower_body_histogram,
#                         known_lower_body_histogram,
#                     )
#                     self.get_logger().info(f"Known_Id: {known_id}, {max_similarity_id}, Calculated similarity: {similarity}, {max_similarity}")

#                     if similarity > max_similarity:
#                         max_similarity = similarity
#                         max_similarity_id = known_id

#                 if max_similarity > SIMILARITY_THRESHOLD:
#                     if max_similarity_id not in active_ids:
#                         person_id = max_similarity_id
#                 else:
#                     person_id = int(track_id)
#                     if person_id not in known_persons:
#                         new_id = max(known_persons.keys(), default=0) + 1
#                         person_id = new_id
#                     known_persons[person_id] = (global_features, upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram)

#                 self.id_last_seen[person_id] = current_time
#                 active_ids.add(person_id)

#                 # 바운딩 박스의 중심 좌표 계산
#                 bbox_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
#                 self.tracked_objects[person_id] = (bbox_center, (x1, y1, x2, y2), None)  # depth_value는 아직 없음

#                 # Get unique color for the person ID
#                 color = self.get_unique_color(person_id)

#                 # 실시간으로 바운딩 박스 표시
#                 cv2.rectangle(self.current_color_image, (x1, y1), (x2, y2), color, 2)
#                 label = f"ID: {person_id}"

#                 # Set text color to white
#                 text_color = (255, 255, 255)

#                 # Define center point text
#                 center_x = (x1 + x2) // 2
#                 center_y = (y1 + y2) // 2
#                 centp = f"Center: ({center_x}, {center_y})"

#                 # Calculate text size for background rectangle for ID and center point
#                 (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
#                 (center_text_width, center_text_height), baseline = cv2.getTextSize(centp, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)

#                 # Calculate the position for the background rectangles and text to display them at the bottom left of the bounding box
#                 label_y_position = y2 + text_height + 10
#                 center_text_y_position = label_y_position + center_text_height + 10

#                 # Draw a filled rectangle as background for the ID text, matching the bounding box color
#                 cv2.rectangle(self.current_color_image, (x1, y2), (x1 + text_width, label_y_position), color, thickness=cv2.FILLED)

#                 # Draw the ID text on top of the rectangle
#                 cv2.putText(self.current_color_image, label, (x1, y2 + text_height), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

#                 # Draw a filled rectangle as background for the center point text, below the ID text
#                 cv2.rectangle(self.current_color_image, (x1, label_y_position + 5), (x1 + center_text_width, center_text_y_position), color, thickness=cv2.FILLED)

#                 # Draw the center point text
#                 cv2.putText(self.current_color_image, centp, (x1, label_y_position + center_text_height + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)



#     def depth_image_callback(self, msg):
#         self.get_logger().info("Received depth image")
        
#         if not self.tracked_objects:
#             self.get_logger().info("No tracked objects available, skipping depth processing")
#             return

#         # Convert depth image to OpenCV format
#         depth_frame = self.bridge.imgmsg_to_cv2(msg, '16UC1')
#         self.get_logger().info("Depth image converted to OpenCV format")

#         # Calculate distance to the closest obstacle
#         self.closest_obstacle_distance = np.min(depth_frame) / 10.0  # Convert minimum depth value to cm

#         active_ids = set()
#         current_time = self.get_clock().now()

#         for person_id, (bbox_center, bbox, _) in self.tracked_objects.items():
#             if bbox is None:
#                 self.get_logger().info(f"No bbox available for Person ID: {person_id}, skipping.")
#                 continue
            
#             x1, y1, x2, y2 = bbox
#             x, y = bbox_center
#             width = x2 - x1
#             height = y2 - y1

#             # Handling depth values, using last valid if current is zero
#             if 0 <= x < depth_frame.shape[1] and 0 <= y < depth_frame.shape[0]:
#                 depth_value = round(depth_frame[y, x] / 10, 2)  # Convert mm to cm with two decimal places
                
#                 if depth_value == 0:
#                     if person_id in self.last_valid_depth:
#                         depth_value = self.last_valid_depth[person_id]
#                         self.get_logger().info(f"Depth value was 0. Using last valid depth value: {depth_value}cm")
#                     else:
#                         self.get_logger().info(f"No valid depth value available for Person ID: {person_id}. Skipping.")
#                         continue
#                 else:
#                     self.last_valid_depth[person_id] = depth_value
#             else:
#                 if person_id in self.last_valid_depth:
#                     depth_value = self.last_valid_depth[person_id]
#                     self.get_logger().info(f"Depth value was None. Using last valid depth value: {depth_value}cm")
#                 else:
#                     depth_value = None
#                     self.get_logger().info(f"Bbox center {bbox_center} is out of depth frame bounds and no previous depth value available.")

#             self.get_logger().info(f"Person ID: {person_id}, Bbox center: {bbox_center}, Width: {width}, Height: {height}, Depth: {depth_value}cm")

#             # Handle object detection and following
#             if 1 in self.tracked_objects:
#                 self.force_drive(id_to_follow=1)
#             else:
#                 self.get_logger().info("ID 1 is not being tracked. Not initiating force drive.")

#             # Update tracking information with depth value
#             self.tracked_objects[person_id] = (bbox_center, bbox, depth_value)

#             # Update the last seen time for the ID
#             self.id_last_seen[person_id] = current_time
#             active_ids.add(person_id)

#             # Get unique color for the person ID
#             color = self.get_unique_color(person_id)

#             # Set text color to white
#             text_color = (255, 255, 255)

#             # Calculate text for center point display
#             centp = f"Center: ({x}, {y})"
#             (center_text_width, center_text_height), _ = cv2.getTextSize(centp, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)

#             depth_text = f"Depth: {depth_value}cm"
#             (depth_text_width, depth_text_height), _ = cv2.getTextSize(depth_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)

#             # Adjust the drawing position to the bottom left of the bounding box, below the center text
#             depth_text_y_position = y2 + center_text_height + depth_text_height + 50

#             # Draw a filled rectangle as background for the depth text, matching the bounding box color
#             cv2.rectangle(self.current_color_image, (x1, y2 + center_text_height + 45), (x1 + depth_text_width, depth_text_y_position), color, thickness=cv2.FILLED)

#             # Draw the depth text
#             cv2.putText(self.current_color_image, depth_text, (x1, y2 + center_text_height + depth_text_height + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

#             # Publish custom message
#             self.publish_id_distance_info(person_id, bbox_center, width, height, depth_value)

#         # Display the processed image in real-time
#         cv2.imshow("Color Image with Tracking", self.current_color_image)
#         cv2.waitKey(1)

    




#     def publish_id_distance_info(self, id, bbox_center, width, height, depth_value):
#         self.get_logger().info(f"Preparing to publish IdDistanceInfo: ID={id}, Bbox Center={bbox_center}, Width={width}, Height={height}, Depth={depth_value}")
        
#         msg = IdDistanceInfo()
#         msg.id = id
#         msg.bbox_center = Point(x=float(bbox_center[0]), y=float(bbox_center[1]), z=0.0)
#         msg.width = float(width)
#         msg.height = float(height)
#         msg.distance = float(depth_value)
        
#         # 발행 전 메시지 정보 로그
#         self.get_logger().info(f"IdDistanceInfo message prepared: {msg}")
        
#         self.publisher_id_distance.publish(msg)
#         self.get_logger().info(f"Published IdDistanceInfo: ID={id}, Bbox Center={bbox_center}, bbox_Width={width}, bbox_Height={height}, Depth={depth_value}")


#     def extract_global_features(self, image):
#         pil_img = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#         input_img = transform(pil_img).unsqueeze(0).to(device)
#         features = predictor(input_img).cpu().numpy()
#         return features

#     def extract_local_features(self, image):
#         height, width, _ = image.shape
#         segments = [
#             image[:height//2, :],  
#             image[height//2:, :],  
#             image[:, :width//2],   
#             image[:, width//2:]    
#         ]
        
#         local_features = []
#         local_histograms = []
#         for segment in segments:
#             pil_img = PILImage.fromarray(cv2.cvtColor(segment, cv2.COLOR_BGR2RGB))
#             input_img = transform(pil_img).unsqueeze(0).to(device)
#             features = predictor(input_img).cpu().numpy()
#             hist = self.calculate_histogram(segment)
#             local_features.append(features)
#             local_histograms.append(hist)
        
#         upper_body_features = local_features[0]  
#         lower_body_features = local_features[1]  
        
#         upper_body_histogram = local_histograms[0]  
#         lower_body_histogram = local_histograms[1]  

#         return upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram

#     def calculate_histogram(self, image):
#         hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#         hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
#         cv2.normalize(hist, hist)
#         return hist.flatten()

#     def calculate_combined_similarity(self, global_features1, global_features2, upper_body_features1, upper_body_features2, lower_body_features1, lower_body_features2, upper_body_histogram1, upper_body_histogram2, lower_body_histogram1, lower_body_histogram2):
#         global_similarity = np.dot(global_features1, global_features2.T) / (np.linalg.norm(global_features1) * np.linalg.norm(global_features2))
#         upper_body_similarity = np.dot(upper_body_features1, upper_body_features2.T) / (np.linalg.norm(upper_body_features1) * np.linalg.norm(upper_body_features2))
#         lower_body_similarity = np.dot(lower_body_features1, lower_body_features2.T) / (np.linalg.norm(lower_body_features1) * np.linalg.norm(lower_body_features2))
#         upper_body_hist_similarity = cv2.compareHist(upper_body_histogram1, upper_body_histogram2, cv2.HISTCMP_CORREL)
#         lower_body_hist_similarity = cv2.compareHist(lower_body_histogram1, lower_body_histogram2, cv2.HISTCMP_CORREL)
        
#         combined_similarity = (0.2 * global_similarity +  
#                                 0.1 * upper_body_similarity +
#                                 0.4 * lower_body_similarity +
#                                 0.3 * lower_body_hist_similarity)
#         return combined_similarity


#     def force_drive(self, id_to_follow=1, duration=1.0, stop_distance=1): # stop_distance = 0.5는 50cm
#         """
#         로봇을 특정 방향으로 주행시키는 함수.

#         :param id_to_follow: 추종할 객체의 ID (기본값은 1)
#         :param duration: 주행 지속 시간 (초 단위)
#         :param stop_distance: 주행을 멈출 장애물까지의 거리 (미터 단위)
#         """

#         if id_to_follow not in self.tracked_objects:
#             self.get_logger().info(f'ID {id_to_follow} is not being tracked.')
#             return

#         bbox_center, _, depth_value = self.tracked_objects[id_to_follow]

#         if depth_value is None or depth_value == 0:
#             self.get_logger().info(f'No valid depth value for ID {id_to_follow}. Cannot proceed.')
#             return

#         camera_center_x = self.current_color_image.shape[1] // 2
#         angular_z = 0.0

#         # 중심 좌표가 카메라의 중심에서 얼마나 떨어져 있는지 계산
#         offset_x = bbox_center[0] - camera_center_x

#         # 회전 속도 비율을 조정할 최대값 설정 (예: 0.5로 설정)
#         max_angular_speed = 0.35

#         # 카메라 중심에서의 오프셋을 기준으로 회전 속도를 비례적으로 조정
#         angular_z = -max_angular_speed * (offset_x / camera_center_x)

#         # angular_z가 max_angular_speed보다 크거나 작지 않도록 클램핑
#         angular_z = max(-max_angular_speed, min(max_angular_speed, angular_z))

#         self.get_logger().info(f'Calculated angular_z: {angular_z:.2f} for offset: {offset_x}')


#         twist_msg = Twist()


#         # 장애물이 일정 거리 안에 있을 경우 멈춤
#         if depth_value < stop_distance * 100:  # depth_value는 cm 단위이므로 100을 곱함
#             self.get_logger().info(f'Safty distance detected within {depth_value / 100:.2f} meters. Stopping force drive.')
#             twist_msg.linear.x = 0.0
#             twist_msg.angular.z = 0.0
#             self.publisher_cmd_vel.publish(twist_msg)
#             time.sleep(0.1)

#         else:
#             safty_distance_diff =  depth_value - (stop_distance * 100)
#             self.get_logger().info(f'Object distance: {depth_value}cm,  Distance between Object and Safty_Zone: {safty_distance_diff}cm')
#             twist_msg.linear.x = 0.1  # 직진
#             twist_msg.angular.z = angular_z  # 회전 방향
#             self.publisher_cmd_vel.publish(twist_msg)
#             time.sleep(0.1)
#             return 



# def main(args=None):
#     rclpy.init(args=args)
#     node = AllInOneNode()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()




















# import collections
# if not hasattr(collections, 'Mapping'):
#     import collections.abc
#     collections.Mapping = collections.abc.Mapping
# import cv2
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# from geometry_msgs.msg import Twist, Point
# from cv_bridge import CvBridge
# import numpy as np
# import torch
# from fastreid.config import get_cfg
# from fastreid.engine import DefaultPredictor
# from fastreid.data.transforms import build_transforms
# from PIL import Image as PILImage
# from ultralytics import YOLO
# from message.msg import IdDistanceInfo
# import time


# # CUDA 디바이스 설정
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # YOLOv5 모델 로드
# model = YOLO("yolov5su.pt").to(device)

# # BotSORT 설정 파일 경로
# tracker_config = "/home/jinjuuk/.local/lib/python3.10/site-packages/ultralytics/cfg/trackers/botsort.yaml"

# # FastReID 설정
# cfg = get_cfg()
# cfg.merge_from_file("/home/jinjuuk/fast-reid/configs/Market1501/bagtricks_R50.yml")
# cfg.MODEL.WEIGHTS = "/home/jinjuuk/dev_ws/pt_files/market_mgn_R50-ibn.pth"
# cfg.MODEL.DEVICE = "cuda"  # GPU 사용 설정

# predictor = DefaultPredictor(cfg)
# transform = build_transforms(cfg, is_train=False)

# # 알려진 사람들의 특징 벡터와 ID를 저장할 딕셔너리
# known_persons = {}

# # 초기 설정값
# SIMILARITY_THRESHOLD = 0.97

# class AllInOneNode(Node):
#     def __init__(self):
#         super().__init__('all_in_one_node')
        
#         # 로그 출력: CUDA 사용 여부 확인
#         if torch.cuda.is_available():
#             cuda_device_name = torch.cuda.get_device_name(0)
#             self.get_logger().info(f"CUDA is available. Using GPU: {cuda_device_name}")
#         else:
#             self.get_logger().info("CUDA is not available. Using CPU")

#         # 구독자 설정
#         self.subscription_color = self.create_subscription(
#             Image,
#             '/camera/color/image_raw',
#             self.color_image_callback,
#             10
#         )
#         self.subscription_depth = self.create_subscription(
#             Image,
#             '/camera/depth/image_raw',
#             self.depth_image_callback,
#             10
#         )
        
#         # 발행자 설정
#         self.publisher_cmd_vel = self.create_publisher(
#             Twist,
#             'cmd_vel',
#             10
#         )
#         self.publisher_id_distance = self.create_publisher(
#             IdDistanceInfo,  
#             'id_distance_info',
#             10
#         )
        
#         # OpenCV를 위한 CVBridge 초기화
#         self.bridge = CvBridge()

#         # 트래킹 정보 저장
#         self.tracked_objects = {}  # id -> (bbox_center, bbox, depth)
#         self.id_last_seen = {}  # id -> last seen time
#         # 각 person_id에 대한 마지막 유효한 깊이 값을 저장하는 딕셔너리 초기화
#         self.last_valid_depth = {}
#         # 장애물까지의 가장 가까운 거리를 저장할 변수 초기화
#         self.closest_obstacle_distance = float('inf')

#         self.get_logger().info("AllInOneNode initialized")


#     # Function to generate a unique color based on ID
#     def get_unique_color(self, id):
#         np.random.seed(id)  # Seed with the ID for consistency
#         return tuple(np.random.randint(0, 255, 3).tolist())


#     def depth_image_callback(self, msg):
#         self.get_logger().info("Received depth image")
        
#         if not self.tracked_objects:
#             self.get_logger().info("No tracked objects available, skipping depth processing")
#             return

#         # Convert depth image to OpenCV format
#         depth_frame = self.bridge.imgmsg_to_cv2(msg, '16UC1')
#         self.get_logger().info("Depth image converted to OpenCV format")

#         # Calculate distance to the closest obstacle
#         self.closest_obstacle_distance = np.min(depth_frame) / 10.0  # Convert minimum depth value to cm

#         active_ids = set()
#         current_time = self.get_clock().now()

#         for person_id, (bbox_center, bbox, _) in self.tracked_objects.items():
#             if bbox is None:
#                 self.get_logger().info(f"No bbox available for Person ID: {person_id}, skipping.")
#                 continue
            
#             x1, y1, x2, y2 = bbox
#             x, y = bbox_center
#             width = x2 - x1
#             height = y2 - y1

#             # Handling depth values, using last valid if current is zero
#             if 0 <= x < depth_frame.shape[1] and 0 <= y < depth_frame.shape[0]:
#                 depth_value = round(depth_frame[y, x] / 10, 2)  # Convert mm to cm with two decimal places
                
#                 if depth_value == 0:
#                     if person_id in self.last_valid_depth:
#                         depth_value = self.last_valid_depth[person_id]
#                         self.get_logger().info(f"Depth value was 0. Using last valid depth value: {depth_value}cm")
#                     else:
#                         self.get_logger().info(f"No valid depth value available for Person ID: {person_id}. Skipping.")
#                         continue
#                 else:
#                     self.last_valid_depth[person_id] = depth_value
#             else:
#                 if person_id in self.last_valid_depth:
#                     depth_value = self.last_valid_depth[person_id]
#                     self.get_logger().info(f"Depth value was None. Using last valid depth value: {depth_value}cm")
#                 else:
#                     depth_value = None
#                     self.get_logger().info(f"Bbox center {bbox_center} is out of depth frame bounds and no previous depth value available.")

#             self.get_logger().info(f"Person ID: {person_id}, Bbox center: {bbox_center}, Width: {width}, Height: {height}, Depth: {depth_value}cm")

#             # Update tracking information with depth value
#             self.tracked_objects[person_id] = (bbox_center, bbox, depth_value)

#             # Update the last seen time for the ID
#             self.id_last_seen[person_id] = current_time
#             active_ids.add(person_id)

#             # Handle object detection and following
#             if 1 in self.tracked_objects:
#                 self.force_drive(id_to_follow=1)
#             else:
#                 self.get_logger().info("ID 1 is not being tracked. Not initiating force drive.")


#         #Display the processed image in real-time
#         cv2.imshow("Color Image with Tracking", self.current_color_image)
#         cv2.waitKey(1)
    

#     def color_image_callback(self, msg):
#         self.get_logger().info("Received color image")

#         # 이미지 변환
#         frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
#         self.current_color_image = frame.copy()
#         self.get_logger().info("Color image converted to OpenCV format")

#         results = model.track(classes=[0], max_det=4, show=False, source=frame, stream=True, tracker=tracker_config)
#         results = list(results)  # generator를 list로 변환
#         self.get_logger().info(f"YOLO tracking complete. Detected {len(results)} results")

#         active_ids = set()
#         current_time = self.get_clock().now()

#         for result in results:
#             boxes = result.boxes.xyxy.cpu().numpy().astype(int) if result.boxes is not None else []
#             track_ids = result.boxes.id.cpu().numpy().astype(int) if result.boxes.id is not None else []

#             for box, track_id in zip(boxes, track_ids):
#                 x1, y1, x2, y2 = box
#                 person_img = frame[y1:y2, x1:x2]

#                 global_features = self.extract_global_features(person_img)
#                 upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram = self.extract_local_features(person_img)

#                 max_similarity = -1
#                 max_similarity_id = -1

#                 for known_id, (known_global_features, known_upper_body_features, known_lower_body_features, known_upper_body_histogram, known_lower_body_histogram) in known_persons.items():
#                     similarity = self.calculate_combined_similarity(
#                         global_features,
#                         known_global_features,
#                         upper_body_features,
#                         known_upper_body_features,
#                         lower_body_features,
#                         known_lower_body_features,
#                         upper_body_histogram,
#                         known_upper_body_histogram,
#                         lower_body_histogram,
#                         known_lower_body_histogram,
#                     )
#                     self.get_logger().info(f"Known_Id: {known_id}, {max_similarity_id}, Calculated similarity: {similarity}, {max_similarity}")

#                     if similarity > max_similarity:
#                         max_similarity = similarity
#                         max_similarity_id = known_id

#                 if max_similarity > SIMILARITY_THRESHOLD:
#                     if max_similarity_id not in active_ids:
#                         person_id = max_similarity_id
#                 else:
#                     person_id = int(track_id)
#                     if person_id not in known_persons:
#                         new_id = max(known_persons.keys(), default=0) + 1
#                         person_id = new_id
#                     known_persons[person_id] = (global_features, upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram)

#                 self.id_last_seen[person_id] = current_time
#                 active_ids.add(person_id)

#                 # 바운딩 박스의 중심 좌표 계산
#                 bbox_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
#                 bbox = (x1, y1, x2, y2)
#                 self.tracked_objects[person_id] = (bbox_center, bbox, self.tracked_objects.get(person_id, (None, None, None))[2])  # Use stored depth value

#                 # Get unique color for the person ID
#                 color = self.get_unique_color(person_id)

#                 # 실시간으로 바운딩 박스 표시
#                 cv2.rectangle(self.current_color_image, (x1, y1), (x2, y2), color, 2)
#                 label = f"ID: {person_id}"

#                 # Set text color to white
#                 text_color = (255, 255, 255)

#                 # Define center point text
#                 center_x = (x1 + x2) // 2
#                 center_y = (y1 + y2) // 2
#                 centp = f"Center: ({center_x}, {center_y})"

#                 # Retrieve depth value
#                 depth_value = self.tracked_objects[person_id][2]
#                 depth_text = f"Depth: {depth_value}cm" if depth_value is not None else "Depth: N/A"

#                 # Calculate text size for background rectangles
#                 (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
#                 (center_text_width, center_text_height), _ = cv2.getTextSize(centp, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
#                 (depth_text_width, depth_text_height), _ = cv2.getTextSize(depth_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)

#                 # Calculate the position for the background rectangles and text to display them at the bottom left of the bounding box
#                 label_y_position = y2 + text_height + 10
#                 center_text_y_position = label_y_position + center_text_height + 10
#                 depth_text_y_position = center_text_y_position + depth_text_height + 10

#                 # Draw a filled rectangle as background for the ID text, matching the bounding box color
#                 cv2.rectangle(self.current_color_image, (x1, y2), (x1 + text_width, label_y_position), color, thickness=cv2.FILLED)

#                 # Draw the ID text on top of the rectangle
#                 cv2.putText(self.current_color_image, label, (x1, y2 + text_height), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

#                 # Draw a filled rectangle as background for the center point text, below the ID text
#                 cv2.rectangle(self.current_color_image, (x1, label_y_position + 5), (x1 + center_text_width, center_text_y_position), color, thickness=cv2.FILLED)

#                 # Draw the center point text
#                 cv2.putText(self.current_color_image, centp, (x1, label_y_position + center_text_height + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

#                 # Draw a filled rectangle as background for the depth text, below the center point text
#                 cv2.rectangle(self.current_color_image, (x1, center_text_y_position + 5), (x1 + depth_text_width, depth_text_y_position), color, thickness=cv2.FILLED)

#                 # Draw the depth text
#                 cv2.putText(self.current_color_image, depth_text, (x1, center_text_y_position + depth_text_height + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

            

#     def publish_id_distance_info(self, id, bbox_center, width, height, depth_value):
#         self.get_logger().info(f"Preparing to publish IdDistanceInfo: ID={id}, Bbox Center={bbox_center}, Width={width}, Height={height}, Depth={depth_value}")
        
#         msg = IdDistanceInfo()
#         msg.id = id
#         msg.bbox_center = Point(x=float(bbox_center[0]), y=float(bbox_center[1]), z=0.0)
#         msg.width = float(width)
#         msg.height = float(height)
#         msg.distance = float(depth_value)
        
#         # 발행 전 메시지 정보 로그
#         self.get_logger().info(f"IdDistanceInfo message prepared: {msg}")
        
#         self.publisher_id_distance.publish(msg)
#         self.get_logger().info(f"Published IdDistanceInfo: ID={id}, Bbox Center={bbox_center}, bbox_Width={width}, bbox_Height={height}, Depth={depth_value}")


#     def extract_global_features(self, image):
#         pil_img = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#         input_img = transform(pil_img).unsqueeze(0).to(device)
#         features = predictor(input_img).cpu().numpy()
#         return features

#     def extract_local_features(self, image):
#         height, width, _ = image.shape
#         segments = [
#             image[:height//2, :],  
#             image[height//2:, :],  
#             image[:, :width//2],   
#             image[:, width//2:]    
#         ]
        
#         local_features = []
#         local_histograms = []
#         for segment in segments:
#             pil_img = PILImage.fromarray(cv2.cvtColor(segment, cv2.COLOR_BGR2RGB))
#             input_img = transform(pil_img).unsqueeze(0).to(device)
#             features = predictor(input_img).cpu().numpy()
#             hist = self.calculate_histogram(segment)
#             local_features.append(features)
#             local_histograms.append(hist)
        
#         upper_body_features = local_features[0]  
#         lower_body_features = local_features[1]  
        
#         upper_body_histogram = local_histograms[0]  
#         lower_body_histogram = local_histograms[1]  

#         return upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram

#     def calculate_histogram(self, image):
#         hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#         hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
#         cv2.normalize(hist, hist)
#         return hist.flatten()

#     def calculate_combined_similarity(self, global_features1, global_features2, upper_body_features1, upper_body_features2, lower_body_features1, lower_body_features2, upper_body_histogram1, upper_body_histogram2, lower_body_histogram1, lower_body_histogram2):
#         global_similarity = np.dot(global_features1, global_features2.T) / (np.linalg.norm(global_features1) * np.linalg.norm(global_features2))
#         upper_body_similarity = np.dot(upper_body_features1, upper_body_features2.T) / (np.linalg.norm(upper_body_features1) * np.linalg.norm(upper_body_features2))
#         lower_body_similarity = np.dot(lower_body_features1, lower_body_features2.T) / (np.linalg.norm(lower_body_features1) * np.linalg.norm(lower_body_features2))
#         upper_body_hist_similarity = cv2.compareHist(upper_body_histogram1, upper_body_histogram2, cv2.HISTCMP_CORREL)
#         lower_body_hist_similarity = cv2.compareHist(lower_body_histogram1, lower_body_histogram2, cv2.HISTCMP_CORREL)
        
#         combined_similarity = (0.2 * global_similarity +  
#                                 0.1 * upper_body_similarity +
#                                 0.4 * lower_body_similarity +
#                                 0.3 * lower_body_hist_similarity)
#         return combined_similarity


#     def force_drive(self, id_to_follow=1, duration=1.0, stop_distance=1): # stop_distance = 0.5는 50cm
#         """
#         로봇을 특정 방향으로 주행시키는 함수.

#         :param id_to_follow: 추종할 객체의 ID (기본값은 1)
#         :param duration: 주행 지속 시간 (초 단위)
#         :param stop_distance: 주행을 멈출 장애물까지의 거리 (미터 단위)
#         """

#         if id_to_follow not in self.tracked_objects:
#             self.get_logger().info(f'ID {id_to_follow} is not being tracked.')
#             return

#         bbox_center, _, depth_value = self.tracked_objects[id_to_follow]

#         if depth_value is None or depth_value == 0:
#             self.get_logger().info(f'No valid depth value for ID {id_to_follow}. Cannot proceed.')
#             return

#         camera_center_x = self.current_color_image.shape[1] // 2
#         angular_z = 0.0

#         # 중심 좌표가 카메라의 중심에서 얼마나 떨어져 있는지 계산
#         offset_x = bbox_center[0] - camera_center_x

#         # 회전 속도 비율을 조정할 최대값 설정 (예: 0.5로 설정)
#         max_angular_speed = 0.35

#         # 카메라 중심에서의 오프셋을 기준으로 회전 속도를 비례적으로 조정
#         angular_z = -max_angular_speed * (offset_x / camera_center_x)

#         # angular_z가 max_angular_speed보다 크거나 작지 않도록 클램핑
#         angular_z = max(-max_angular_speed, min(max_angular_speed, angular_z))

#         self.get_logger().info(f'Calculated angular_z: {angular_z:.2f} for offset: {offset_x}')


#         twist_msg = Twist()


#         # 장애물이 일정 거리 안에 있을 경우 멈춤
#         if depth_value < stop_distance * 100:  # depth_value는 cm 단위이므로 100을 곱함
#             self.get_logger().info(f'Safty distance detected within {depth_value / 100:.2f} meters. Stopping force drive.')
#             twist_msg.linear.x = 0.0
#             twist_msg.angular.z = 0.0
#             self.publisher_cmd_vel.publish(twist_msg)
#             time.sleep(0.1)

#         else:
#             safty_distance_diff =  depth_value - (stop_distance * 100)
#             self.get_logger().info(f'Object distance: {depth_value}cm,  Distance between Object and Safty_Zone: {safty_distance_diff}cm')
#             twist_msg.linear.x = 0.1  # 직진
#             twist_msg.angular.z = angular_z  # 회전 방향
#             self.publisher_cmd_vel.publish(twist_msg)
#             time.sleep(0.1)
#             return 



# def main(args=None):
#     rclpy.init(args=args)
#     node = AllInOneNode()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()

















# import collections
# if not hasattr(collections, 'Mapping'):
#     import collections.abc
#     collections.Mapping = collections.abc.Mapping
# import cv2
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# from geometry_msgs.msg import Twist, Point
# from cv_bridge import CvBridge
# import numpy as np
# import torch
# from fastreid.config import get_cfg
# from fastreid.engine import DefaultPredictor
# from fastreid.data.transforms import build_transforms
# from PIL import Image as PILImage
# from ultralytics import YOLO
# from message.msg import IdDistanceInfo
# import time


# # CUDA 디바이스 설정
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # YOLOv5 모델 로드
# model = YOLO("yolov5nu.pt").to(device)

# # BotSORT 설정 파일 경로
# tracker_config = "/home/jinjuuk/.local/lib/python3.10/site-packages/ultralytics/cfg/trackers/botsort.yaml"

# # FastReID 설정
# cfg = get_cfg()
# cfg.merge_from_file("/home/jinjuuk/fast-reid/configs/Market1501/bagtricks_R50.yml")
# cfg.MODEL.WEIGHTS = "/home/jinjuuk/dev_ws/pt_files/market_mgn_R50-ibn.pth"
# cfg.MODEL.DEVICE = "cuda"  # GPU 사용 설정

# predictor = DefaultPredictor(cfg)
# transform = build_transforms(cfg, is_train=False)

# # 알려진 사람들의 특징 벡터와 ID를 저장할 딕셔너리
# known_persons = {}

# # 초기 설정값
# SIMILARITY_THRESHOLD = 0.92

# class AllInOneNode(Node):
#     def __init__(self):
#         super().__init__('all_in_one_node')
        
#         # 로그 출력: CUDA 사용 여부 확인
#         if torch.cuda.is_available():
#             cuda_device_name = torch.cuda.get_device_name(0)
#             self.get_logger().info(f"CUDA is available. Using GPU: {cuda_device_name}")
#         else:
#             self.get_logger().info("CUDA is not available. Using CPU")

#         # 구독자 설정
#         self.subscription_color = self.create_subscription(
#             Image,
#             '/camera/color/image_raw',
#             self.color_image_callback,
#             10
#         )
#         self.subscription_depth = self.create_subscription(
#             Image,
#             '/camera/depth/image_raw',
#             self.depth_image_callback,
#             10
#         )
        
#         # 발행자 설정
#         self.publisher_cmd_vel = self.create_publisher(
#             Twist,
#             'cmd_vel',
#             10
#         )
#         self.publisher_id_distance = self.create_publisher(
#             IdDistanceInfo,  
#             'id_distance_info',
#             10
#         )
        
#         # OpenCV를 위한 CVBridge 초기화
#         self.bridge = CvBridge()

#         # 트래킹 정보 저장
#         self.tracked_objects = {}  # id -> (bbox_center, bbox, depth)
#         self.id_last_seen = {}  # id -> last seen time
#         # 각 person_id에 대한 마지막 유효한 깊이 값을 저장하는 딕셔너리 초기화
#         self.last_valid_depth = {}
#         # 장애물까지의 가장 가까운 거리를 저장할 변수 초기화
#         self.closest_obstacle_distance = float('inf')

#         # self.get_logger().info("AllInOneNode initialized")


#     # Function to generate a unique color based on ID
#     def get_unique_color(self, id):
#         np.random.seed(id)  # Seed with the ID for consistency
#         return tuple(np.random.randint(0, 255, 3).tolist())


#     def depth_image_callback(self, msg):
#         # self.get_logger().info("Received depth image")
        
#         # if not self.tracked_objects:
#         #     self.get_logger().info("No tracked objects available, skipping depth processing")
#         #     return

#         # Convert depth image to OpenCV format
#         depth_frame = self.bridge.imgmsg_to_cv2(msg, '16UC1')
#         # self.get_logger().info("Depth image converted to OpenCV format")

#         # Calculate distance to the closest obstacle
#         self.closest_obstacle_distance = np.min(depth_frame) / 10.0  # Convert minimum depth value to cm

#         active_ids = set()
#         current_time = self.get_clock().now()

#         for person_id, (bbox_center, bbox, _) in self.tracked_objects.items():
#             # if bbox is None:
#             #     self.get_logger().info(f"No bbox available for Person ID: {person_id}, skipping.")
#             #     continue
            
#             x1, y1, x2, y2 = bbox
#             x, y = bbox_center
#             width = x2 - x1
#             height = y2 - y1

#             # Handling depth values, using last valid if current is zero
#             if 0 <= x < depth_frame.shape[1] and 0 <= y < depth_frame.shape[0]:
#                 depth_value = round(depth_frame[y, x] / 10, 2)  # Convert mm to cm with two decimal places
                
#                 if depth_value == 0:
#                     if person_id in self.last_valid_depth:
#                         depth_value = self.last_valid_depth[person_id]
#                     #     self.get_logger().info(f"Depth value was 0. Using last valid depth value: {depth_value}cm")
#                     # else:
#                     #     self.get_logger().info(f"No valid depth value available for Person ID: {person_id}. Skipping.")
#                     #     continue
#                 else:
#                     self.last_valid_depth[person_id] = depth_value
#             else:
#                 if person_id in self.last_valid_depth:
#                     depth_value = self.last_valid_depth[person_id]
#                     # self.get_logger().info(f"Depth value was None. Using last valid depth value: {depth_value}cm")
#                 else:
#                     depth_value = None
#                     # self.get_logger().info(f"Bbox center {bbox_center} is out of depth frame bounds and no previous depth value available.")

#             # self.get_logger().info(f"Person ID: {person_id}, Bbox center: {bbox_center}, Width: {width}, Height: {height}, Depth: {depth_value}cm")

#             # Update tracking information with depth value
#             self.tracked_objects[person_id] = (bbox_center, bbox, depth_value)

#             # Update the last seen time for the ID
#             self.id_last_seen[person_id] = current_time
#             active_ids.add(person_id)

#             # Handle object detection and following
#             if 1 in self.tracked_objects:
#                 self.force_drive(id_to_follow=1)
#             # else:
#             #     self.get_logger().info("ID 1 is not being tracked. Not initiating force drive.")


#         #Display the processed image in real-time
#         cv2.imshow("Color Image with Tracking", self.current_color_image)
#         cv2.waitKey(1)
    

#     def color_image_callback(self, msg):
#         # self.get_logger().info("Received color image")

#         # 이미지 변환
#         frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
#         self.current_color_image = frame.copy()
#         # self.get_logger().info("Color image converted to OpenCV format")

#         results = model.track(classes=[0], max_det=4, show=False, source=frame, stream=True, tracker=tracker_config)
#         results = list(results)  # generator를 list로 변환
#         # self.get_logger().info(f"YOLO tracking complete. Detected {len(results)} results")

#         active_ids = set()
#         current_time = self.get_clock().now()

#         for result in results:
#             boxes = result.boxes.xyxy.cpu().numpy().astype(int) if result.boxes is not None else []
#             track_ids = result.boxes.id.cpu().numpy().astype(int) if result.boxes.id is not None else []

#             for box, track_id in zip(boxes, track_ids):
#                 x1, y1, x2, y2 = box
#                 person_img = frame[y1:y2, x1:x2]

#                 global_features = self.extract_global_features(person_img)
#                 upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram = self.extract_local_features(person_img)

#                 max_similarity = -1
#                 max_similarity_id = -1

#                 for known_id, (known_global_features, known_upper_body_features, known_lower_body_features, known_upper_body_histogram, known_lower_body_histogram) in known_persons.items():
#                     similarity = self.calculate_combined_similarity(
#                         global_features,
#                         known_global_features,
#                         upper_body_features,
#                         known_upper_body_features,
#                         lower_body_features,
#                         known_lower_body_features,
#                         upper_body_histogram,
#                         known_upper_body_histogram,
#                         lower_body_histogram,
#                         known_lower_body_histogram,
#                     )
#                     # self.get_logger().info(f"Known_Id: {known_id}, {max_similarity_id}, Calculated similarity: {similarity}, {max_similarity}")

#                     if similarity > max_similarity:
#                         max_similarity = similarity
#                         max_similarity_id = known_id

#                 if max_similarity > SIMILARITY_THRESHOLD:
#                     if max_similarity_id not in active_ids:
#                         person_id = max_similarity_id
#                 else:
#                     person_id = int(track_id)
#                     if person_id not in known_persons:
#                         new_id = max(known_persons.keys(), default=0) + 1
#                         person_id = new_id
#                     known_persons[person_id] = (global_features, upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram)

#                 self.id_last_seen[person_id] = current_time
#                 active_ids.add(person_id)

#                 # 바운딩 박스의 중심 좌표 계산
#                 bbox_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
#                 bbox = (x1, y1, x2, y2)
#                 self.tracked_objects[person_id] = (bbox_center, bbox, self.tracked_objects.get(person_id, (None, None, None))[2])  # Use stored depth value

#                 # Get unique color for the person ID
#                 color = self.get_unique_color(person_id)

#                 # 실시간으로 바운딩 박스 표시
#                 cv2.rectangle(self.current_color_image, (x1, y1), (x2, y2), color, 2)
#                 label = f"ID: {person_id}"

#                 # Set text color to white
#                 text_color = (255, 255, 255)

#                 # Define center point text
#                 center_x = (x1 + x2) // 2
#                 center_y = (y1 + y2) // 2
#                 centp = f"Center: ({center_x}, {center_y})"

#                 # Retrieve depth value
#                 depth_value = self.tracked_objects[person_id][2]
#                 depth_text = f"Depth: {depth_value}cm" if depth_value is not None else "Depth: N/A"

#                 # Calculate text size for background rectangles
#                 (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
#                 (center_text_width, center_text_height), _ = cv2.getTextSize(centp, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
#                 (depth_text_width, depth_text_height), _ = cv2.getTextSize(depth_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)

#                 # Calculate the position for the background rectangles and text to display them at the bottom left of the bounding box
#                 label_y_position = y2 + text_height + 10
#                 center_text_y_position = label_y_position + center_text_height + 10
#                 depth_text_y_position = center_text_y_position + depth_text_height + 10

#                 # Draw a filled rectangle as background for the ID text, matching the bounding box color
#                 cv2.rectangle(self.current_color_image, (x1, y2), (x1 + text_width, label_y_position), color, thickness=cv2.FILLED)

#                 # Draw the ID text on top of the rectangle
#                 cv2.putText(self.current_color_image, label, (x1, y2 + text_height), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

#                 # Draw a filled rectangle as background for the center point text, below the ID text
#                 cv2.rectangle(self.current_color_image, (x1, label_y_position + 5), (x1 + center_text_width, center_text_y_position), color, thickness=cv2.FILLED)

#                 # Draw the center point text
#                 cv2.putText(self.current_color_image, centp, (x1, label_y_position + center_text_height + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

#                 # Draw a filled rectangle as background for the depth text, below the center point text
#                 cv2.rectangle(self.current_color_image, (x1, center_text_y_position + 5), (x1 + depth_text_width, depth_text_y_position), color, thickness=cv2.FILLED)

#                 # Draw the depth text
#                 cv2.putText(self.current_color_image, depth_text, (x1, center_text_y_position + depth_text_height + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

            

#     def publish_id_distance_info(self, id, bbox_center, width, height, depth_value):
#         # self.get_logger().info(f"Preparing to publish IdDistanceInfo: ID={id}, Bbox Center={bbox_center}, Width={width}, Height={height}, Depth={depth_value}")
        
#         msg = IdDistanceInfo()
#         msg.id = id
#         msg.bbox_center = Point(x=float(bbox_center[0]), y=float(bbox_center[1]), z=0.0)
#         msg.width = float(width)
#         msg.height = float(height)
#         msg.distance = float(depth_value)
        
#         # # 발행 전 메시지 정보 로그
#         # self.get_logger().info(f"IdDistanceInfo message prepared: {msg}")
        
#         self.publisher_id_distance.publish(msg)
#         # self.get_logger().info(f"Published IdDistanceInfo: ID={id}, Bbox Center={bbox_center}, bbox_Width={width}, bbox_Height={height}, Depth={depth_value}")


#     def extract_global_features(self, image):
#         pil_img = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#         input_img = transform(pil_img).unsqueeze(0).to(device)
#         features = predictor(input_img).cpu().numpy()
#         return features

#     def extract_local_features(self, image):
#         height, width, _ = image.shape
#         segments = [
#             image[:height//2, :],  
#             image[height//2:, :],  
#             image[:, :width//2],   
#             image[:, width//2:]    
#         ]
        
#         local_features = []
#         local_histograms = []
#         for segment in segments:
#             pil_img = PILImage.fromarray(cv2.cvtColor(segment, cv2.COLOR_BGR2RGB))
#             input_img = transform(pil_img).unsqueeze(0).to(device)
#             features = predictor(input_img).cpu().numpy()
#             hist = self.calculate_histogram(segment)
#             local_features.append(features)
#             local_histograms.append(hist)
        
#         upper_body_features = local_features[0]  
#         lower_body_features = local_features[1]  
        
#         upper_body_histogram = local_histograms[0]  
#         lower_body_histogram = local_histograms[1]  

#         return upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram

#     def calculate_histogram(self, image):
#         hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#         hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
#         cv2.normalize(hist, hist)
#         return hist.flatten()

#     def calculate_combined_similarity(self, global_features1, global_features2, upper_body_features1, upper_body_features2, lower_body_features1, lower_body_features2, upper_body_histogram1, upper_body_histogram2, lower_body_histogram1, lower_body_histogram2):
#         global_similarity = np.dot(global_features1, global_features2.T) / (np.linalg.norm(global_features1) * np.linalg.norm(global_features2))
#         upper_body_similarity = np.dot(upper_body_features1, upper_body_features2.T) / (np.linalg.norm(upper_body_features1) * np.linalg.norm(upper_body_features2))
#         lower_body_similarity = np.dot(lower_body_features1, lower_body_features2.T) / (np.linalg.norm(lower_body_features1) * np.linalg.norm(lower_body_features2))
#         upper_body_hist_similarity = cv2.compareHist(upper_body_histogram1, upper_body_histogram2, cv2.HISTCMP_CORREL)
#         lower_body_hist_similarity = cv2.compareHist(lower_body_histogram1, lower_body_histogram2, cv2.HISTCMP_CORREL)
        
#         combined_similarity = (0.3 * global_similarity +  
#                                 0.1 * upper_body_similarity +
#                                 0.3 * lower_body_similarity +
#                                 0.3 * lower_body_hist_similarity)
#         return combined_similarity


#     def force_drive(self, id_to_follow=1, duration=1.0, stop_distance=0.7):  # stop_distance is in meters
#         """
#         로봇을 특정 방향으로 주행시키는 함수.

#         :param id_to_follow: 추종할 객체의 ID (기본값은 1)
#         :param duration: 주행 지속 시간 (초 단위)
#         :param stop_distance: 주행을 멈출 장애물까지의 거리 (미터 단위)
#         """

#         # if id_to_follow not in self.tracked_objects:
#         #     self.get_logger().info(f'ID {id_to_follow} is not being tracked.')
#         #     return

#         bbox_center, _, depth_value = self.tracked_objects[id_to_follow]

#         # if depth_value is None or depth_value == 0:
#         #     self.get_logger().info(f'No valid depth value for ID {id_to_follow}. Cannot proceed.')
#         #     return



#         # Check if any objects are detected in the frame
#         if not self.tracked_objects:
#             # self.get_logger().info('No objects detected in the frame. Stopping force drive.')
#             twist_msg = Twist()
#             twist_msg.linear.x = 0.0
#             twist_msg.angular.z = 0.0
#             self.publisher_cmd_vel.publish(twist_msg)
#             time.sleep(0.1)
#             return



#         # Check if any tracked object is too close
#         for person_id, (_, _, depth) in self.tracked_objects.items():
#             if depth is not None and depth < stop_distance * 100:  # depth is in cm, convert stop_distance to cm
#                 # self.get_logger().info(f'Object ID {person_id} is too close: {depth / 100:.2f} meters. Stopping force drive.')
#                 twist_msg = Twist()
#                 twist_msg.linear.x = 0.0
#                 twist_msg.angular.z = 0.0
#                 self.publisher_cmd_vel.publish(twist_msg)
#                 time.sleep(0.1)
#                 return


#         camera_center_x = self.current_color_image.shape[1] // 2
#         angular_z = 0.0

#         # Calculate how far the center of the bbox is from the camera's center
#         offset_x = bbox_center[0] - camera_center_x

#         # Maximum angular speed setting (e.g., set to 0.35)
#         max_angular_speed = 0.35

#         # Adjust rotational speed proportionally based on the offset from the camera center
#         angular_z = -max_angular_speed * (offset_x / camera_center_x)

#         # Clamp angular_z to be no larger than max_angular_speed and no smaller than -max_angular_speed
#         angular_z = max(-max_angular_speed, min(max_angular_speed, angular_z))

#         # self.get_logger().info(f'Calculated angular_z: {angular_z:.2f} for offset: {offset_x}')

#         # If no object is too close, proceed with driving
#         safty_distance_diff = depth_value - (stop_distance * 100)
#         # self.get_logger().info(f'Object ID {id_to_follow} distance: {depth_value}cm, Distance between Object and Safety Zone: {safty_distance_diff}cm')

#         twist_msg = Twist()
#         twist_msg.linear.x = 0.1  # Move forward
#         twist_msg.angular.z = angular_z  # Adjust orientation
#         self.publisher_cmd_vel.publish(twist_msg)
#         time.sleep(0.1)




# def main(args=None):
#     rclpy.init(args=args)
#     node = AllInOneNode()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()












# # 다중 프레임 검증(단일 프레임에서의 유사도 계산 대신 여러 프레임에서의 유사도를 통합하여 결정)을 통해 순간적인 특징 변화나 추적 추가
# import collections
# if not hasattr(collections, 'Mapping'):
#     import collections.abc
#     collections.Mapping = collections.abc.Mapping
# import cv2
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# from geometry_msgs.msg import Twist, Point
# from cv_bridge import CvBridge
# import numpy as np
# import torch
# from fastreid.config import get_cfg
# from fastreid.engine import DefaultPredictor
# from fastreid.data.transforms import build_transforms
# from PIL import Image as PILImage
# from ultralytics import YOLO
# from message.msg import IdDistanceInfo
# import time


# # CUDA 디바이스 설정
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # YOLOv5 모델 로드
# model = YOLO("yolov5nu.pt").to(device)

# # BotSORT 설정 파일 경로
# tracker_config = "/home/jinjuuk/.local/lib/python3.10/site-packages/ultralytics/cfg/trackers/botsort.yaml"

# # FastReID 설정
# cfg = get_cfg()
# cfg.merge_from_file("/home/jinjuuk/fast-reid/configs/Market1501/bagtricks_R50.yml")
# cfg.MODEL.WEIGHTS = "/home/jinjuuk/dev_ws/pt_files/market_mgn_R50-ibn.pth"
# cfg.MODEL.DEVICE = "cuda"  # GPU 사용 설정

# predictor = DefaultPredictor(cfg)
# transform = build_transforms(cfg, is_train=False)

# # 알려진 사람들의 특징 벡터와 ID를 저장할 딕셔너리
# known_persons = {}

# # 초기 설정값
# SIMILARITY_THRESHOLD = 0.91

# class AllInOneNode(Node):
#     def __init__(self):
#         super().__init__('all_in_one_node')
        
#         # # 로그 출력: CUDA 사용 여부 확인
#         if torch.cuda.is_available():
#             cuda_device_name = torch.cuda.get_device_name(0)
#             self.get_logger().info(f"CUDA is available. Using GPU: {cuda_device_name}")
#         else:
#             self.get_logger().info("CUDA is not available. Using CPU")

#         # 구독자 설정
#         self.subscription_color = self.create_subscription(
#             Image,
#             '/camera/color/image_raw',
#             self.color_image_callback,
#             10
#         )
#         self.subscription_depth = self.create_subscription(
#             Image,
#             '/camera/depth/image_raw',
#             self.depth_image_callback,
#             10
#         )
        
#         # 발행자 설정
#         self.publisher_cmd_vel = self.create_publisher(
#             Twist,
#             'cmd_vel',
#             10
#         )
#         self.publisher_id_distance = self.create_publisher(
#             IdDistanceInfo,  
#             'id_distance_info',
#             10
#         )
        
#         # OpenCV를 위한 CVBridge 초기화
#         self.bridge = CvBridge()

#         # 트래킹 정보 저장
#         self.tracked_objects = {}  # id -> (bbox_center, bbox, depth)
#         self.id_last_seen = {}  # id -> last seen time
#         # 각 person_id에 대한 마지막 유효한 깊이 값을 저장하는 딕셔너리 초기화
#         self.last_valid_depth = {}
#         # 장애물까지의 가장 가까운 거리를 저장할 변수 초기화
#         self.closest_obstacle_distance = float('inf')

#         # Dictionary to store similarity scores over multiple frames
#         self.similarity_scores_buffer = collections.defaultdict(lambda: collections.deque(maxlen=5))


#         self.get_logger().info("AllInOneNode initialized")


#     # Function to generate a unique color based on ID
#     def get_unique_color(self, id):
#         np.random.seed(id)  # Seed with the ID for consistency
#         return tuple(np.random.randint(0, 255, 3).tolist())


#     def depth_image_callback(self, msg):
#         self.get_logger().info("Received depth image")
        
#         if not self.tracked_objects:
#             self.get_logger().info("No tracked objects available, skipping depth processing")
#             return

#         # Convert depth image to OpenCV format
#         depth_frame = self.bridge.imgmsg_to_cv2(msg, '16UC1')
#         self.get_logger().info("Depth image converted to OpenCV format")

#         # Calculate distance to the closest obstacle
#         self.closest_obstacle_distance = np.min(depth_frame) / 10.0  # Convert minimum depth value to cm

#         active_ids = set()
#         current_time = self.get_clock().now()

#         for person_id, (bbox_center, bbox, _) in self.tracked_objects.items():
#             if bbox is None:
#                 self.get_logger().info(f"No bbox available for Person ID: {person_id}, skipping.")
#                 continue
            
#             x1, y1, x2, y2 = bbox
#             x, y = bbox_center
#             width = x2 - x1
#             height = y2 - y1

#             # Handling depth values, using last valid if current is zero
#             if 0 <= x < depth_frame.shape[1] and 0 <= y < depth_frame.shape[0]:
#                 depth_value = round(depth_frame[y, x] / 10, 2)  # Convert mm to cm with two decimal places
                
#                 if depth_value == 0:
#                     if person_id in self.last_valid_depth:
#                         depth_value = self.last_valid_depth[person_id]
#                         self.get_logger().info(f"Depth value was 0. Using last valid depth value: {depth_value}cm")
#                     else:
#                         self.get_logger().info(f"No valid depth value available for Person ID: {person_id}. Skipping.")
#                         continue
#                 else:
#                     self.last_valid_depth[person_id] = depth_value
#             else:
#                 if person_id in self.last_valid_depth:
#                     depth_value = self.last_valid_depth[person_id]
#                     self.get_logger().info(f"Depth value was None. Using last valid depth value: {depth_value}cm")
#                 else:
#                     depth_value = None
#                     self.get_logger().info(f"Bbox center {bbox_center} is out of depth frame bounds and no previous depth value available.")

#             self.get_logger().info(f"Person ID: {person_id}, Bbox center: {bbox_center}, Width: {width}, Height: {height}, Depth: {depth_value}cm")

#             # Update tracking information with depth value
#             self.tracked_objects[person_id] = (bbox_center, bbox, depth_value)

#             # Update the last seen time for the ID
#             self.id_last_seen[person_id] = current_time
#             active_ids.add(person_id)

#             # Handle object detection and following
#             if 1 in self.tracked_objects:
#                 self.force_drive(id_to_follow=1)
#             else:
#                 self.get_logger().info("ID 1 is not being tracked. Not initiating force drive.")


#         #Display the processed image in real-time
#         cv2.imshow("Color Image with Tracking", self.current_color_image)
#         cv2.waitKey(1)
    

#     def color_image_callback(self, msg):
#         self.get_logger().info("Received color image")

#         # 이미지 변환
#         frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
#         self.current_color_image = frame.copy()
#         self.get_logger().info("Color image converted to OpenCV format")

#         results = model.track(classes=[0], max_det=4, show=False, source=frame, stream=True, tracker=tracker_config)
#         results = list(results)  # generator를 list로 변환
#         self.get_logger().info(f"YOLO tracking complete. Detected {len(results)} results")

#         active_ids = set()
#         current_time = self.get_clock().now()

#         for result in results:
#             boxes = result.boxes.xyxy.cpu().numpy().astype(int) if result.boxes is not None else []
#             track_ids = result.boxes.id.cpu().numpy().astype(int) if result.boxes.id is not None else []

#             for box, track_id in zip(boxes, track_ids):
#                 x1, y1, x2, y2 = box
#                 person_img = frame[y1:y2, x1:x2]

#                 global_features = self.extract_global_features(person_img)
#                 upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram = self.extract_local_features(person_img)

#                 max_similarity = -1
#                 max_similarity_id = -1

#                 for known_id, (known_global_features, known_upper_body_features, known_lower_body_features, known_upper_body_histogram, known_lower_body_histogram) in known_persons.items():
#                     similarity = self.calculate_combined_similarity(
#                         global_features,
#                         known_global_features,
#                         upper_body_features,
#                         known_upper_body_features,
#                         lower_body_features,
#                         known_lower_body_features,
#                         upper_body_histogram,
#                         known_upper_body_histogram,
#                         lower_body_histogram,
#                         known_lower_body_histogram,
#                     )

#                     # Store the similarity score in the buffer for this known ID
#                     self.similarity_scores_buffer[known_id].append(similarity)

#                     # Compute the average similarity over the buffer
#                     avg_similarity = np.mean(self.similarity_scores_buffer[known_id])
                    
#                     self.get_logger().info(f"Known_Id: {known_id}, {max_similarity_id}, Calculated similarity: {similarity}, Avg similarity: {avg_similarity}, Max similarity: {max_similarity}")

#                     # Update the max_similarity based on the average similarity
#                     if avg_similarity > max_similarity:
#                         max_similarity = avg_similarity
#                         max_similarity_id = known_id

#                 # Decision based on averaged similarity
#                 if max_similarity > SIMILARITY_THRESHOLD:
#                     person_id = max_similarity_id
#                 else:
#                     person_id = int(track_id)
#                     if person_id not in known_persons:
#                         new_id = max(known_persons.keys(), default=0) + 1
#                         person_id = new_id
#                     known_persons[person_id] = (global_features, upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram)

#                 self.id_last_seen[person_id] = current_time
#                 active_ids.add(person_id)



#                 # 바운딩 박스의 중심 좌표 계산
#                 bbox_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
#                 bbox = (x1, y1, x2, y2)
#                 self.tracked_objects[person_id] = (bbox_center, bbox, self.tracked_objects.get(person_id, (None, None, None))[2])  # Use stored depth value

#                 # Get unique color for the person ID
#                 color = self.get_unique_color(person_id)

#                 # 실시간으로 바운딩 박스 표시
#                 cv2.rectangle(self.current_color_image, (x1, y1), (x2, y2), color, 2)
#                 label = f"ID: {person_id}"

#                 # Set text color to white
#                 text_color = (255, 255, 255)

#                 # Define center point text
#                 center_x = (x1 + x2) // 2
#                 center_y = (y1 + y2) // 2
#                 centp = f"Center: ({center_x}, {center_y})"

#                 # Retrieve depth value
#                 depth_value = self.tracked_objects[person_id][2]
#                 depth_text = f"Depth: {depth_value}cm" if depth_value is not None else "Depth: N/A"

#                 # Calculate text size for background rectangles
#                 (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
#                 (center_text_width, center_text_height), _ = cv2.getTextSize(centp, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
#                 (depth_text_width, depth_text_height), _ = cv2.getTextSize(depth_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)

#                 # Calculate the position for the background rectangles and text to display them at the bottom left of the bounding box
#                 label_y_position = y2 + text_height + 10
#                 center_text_y_position = label_y_position + center_text_height + 10
#                 depth_text_y_position = center_text_y_position + depth_text_height + 10

#                 # Draw a filled rectangle as background for the ID text, matching the bounding box color
#                 cv2.rectangle(self.current_color_image, (x1, y2), (x1 + text_width, label_y_position), color, thickness=cv2.FILLED)

#                 # Draw the ID text on top of the rectangle
#                 cv2.putText(self.current_color_image, label, (x1, y2 + text_height), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

#                 # Draw a filled rectangle as background for the center point text, below the ID text
#                 cv2.rectangle(self.current_color_image, (x1, label_y_position + 5), (x1 + center_text_width, center_text_y_position), color, thickness=cv2.FILLED)

#                 # Draw the center point text
#                 cv2.putText(self.current_color_image, centp, (x1, label_y_position + center_text_height + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

#                 # Draw a filled rectangle as background for the depth text, below the center point text
#                 cv2.rectangle(self.current_color_image, (x1, center_text_y_position + 5), (x1 + depth_text_width, depth_text_y_position), color, thickness=cv2.FILLED)

#                 # Draw the depth text
#                 cv2.putText(self.current_color_image, depth_text, (x1, center_text_y_position + depth_text_height + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

#                 # Clean up similarity buffers for inactive IDs
#                 for known_id in list(self.similarity_scores_buffer.keys()):
#                     if known_id not in active_ids:
#                         del self.similarity_scores_buffer[known_id]


#     def publish_id_distance_info(self, id, bbox_center, width, height, depth_value):
#         self.get_logger().info(f"Preparing to publish IdDistanceInfo: ID={id}, Bbox Center={bbox_center}, Width={width}, Height={height}, Depth={depth_value}")
        
#         msg = IdDistanceInfo()
#         msg.id = id
#         msg.bbox_center = Point(x=float(bbox_center[0]), y=float(bbox_center[1]), z=0.0)
#         msg.width = float(width)
#         msg.height = float(height)
#         msg.distance = float(depth_value)
        
#         # # 발행 전 메시지 정보 로그
#         self.get_logger().info(f"IdDistanceInfo message prepared: {msg}")
        
#         self.publisher_id_distance.publish(msg)
#         self.get_logger().info(f"Published IdDistanceInfo: ID={id}, Bbox Center={bbox_center}, bbox_Width={width}, bbox_Height={height}, Depth={depth_value}")


#     def extract_global_features(self, image):
#         pil_img = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#         input_img = transform(pil_img).unsqueeze(0).to(device)
#         features = predictor(input_img).cpu().numpy()
#         return features

#     def extract_local_features(self, image):
#         height, width, _ = image.shape
#         segments = [
#             image[:height//2, :],  
#             image[height//2:, :],  
#             image[:, :width//2],   
#             image[:, width//2:]    
#         ]
        
#         local_features = []
#         local_histograms = []
#         for segment in segments:
#             pil_img = PILImage.fromarray(cv2.cvtColor(segment, cv2.COLOR_BGR2RGB))
#             input_img = transform(pil_img).unsqueeze(0).to(device)
#             features = predictor(input_img).cpu().numpy()
#             hist = self.calculate_histogram(segment)
#             local_features.append(features)
#             local_histograms.append(hist)
        
#         upper_body_features = local_features[0]  
#         lower_body_features = local_features[1]  
        
#         upper_body_histogram = local_histograms[0]  
#         lower_body_histogram = local_histograms[1]  

#         return upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram

#     def calculate_histogram(self, image):
#         hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#         hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
#         cv2.normalize(hist, hist)
#         return hist.flatten()

#     def calculate_combined_similarity(self, global_features1, global_features2, upper_body_features1, upper_body_features2, lower_body_features1, lower_body_features2, upper_body_histogram1, upper_body_histogram2, lower_body_histogram1, lower_body_histogram2):
#         global_similarity = np.dot(global_features1, global_features2.T) / (np.linalg.norm(global_features1) * np.linalg.norm(global_features2))
#         upper_body_similarity = np.dot(upper_body_features1, upper_body_features2.T) / (np.linalg.norm(upper_body_features1) * np.linalg.norm(upper_body_features2))
#         lower_body_similarity = np.dot(lower_body_features1, lower_body_features2.T) / (np.linalg.norm(lower_body_features1) * np.linalg.norm(lower_body_features2))
#         upper_body_hist_similarity = cv2.compareHist(upper_body_histogram1, upper_body_histogram2, cv2.HISTCMP_CORREL)
#         lower_body_hist_similarity = cv2.compareHist(lower_body_histogram1, lower_body_histogram2, cv2.HISTCMP_CORREL)
        
#         combined_similarity = (0.3 * global_similarity +  
#                                 0.1 * upper_body_similarity +
#                                 0.3 * lower_body_similarity +
#                                 0.3 * lower_body_hist_similarity)
#         return combined_similarity


#     def force_drive(self, id_to_follow=1, duration=1.0, stop_distance=0.7):  # stop_distance is in meters
#         """
#         로봇을 특정 방향으로 주행시키는 함수.

#         :param id_to_follow: 추종할 객체의 ID (기본값은 1)
#         :param duration: 주행 지속 시간 (초 단위)
#         :param stop_distance: 주행을 멈출 장애물까지의 거리 (미터 단위)
#         """

#         if id_to_follow not in self.tracked_objects:
#             self.get_logger().info(f'ID {id_to_follow} is not being tracked.')
#             return

#         bbox_center, _, depth_value = self.tracked_objects[id_to_follow]

#         if depth_value is None or depth_value == 0:
#             self.get_logger().info(f'No valid depth value for ID {id_to_follow}. Cannot proceed.')
#             return



#         # Check if any objects are detected in the frame
#         if not self.tracked_objects:
#             self.get_logger().info('No objects detected in the frame. Stopping force drive.')
#             twist_msg = Twist()
#             twist_msg.linear.x = 0.0
#             twist_msg.angular.z = 0.0
#             self.publisher_cmd_vel.publish(twist_msg)
#             time.sleep(0.1)
#             return



#         # Check if any tracked object is too close
#         for person_id, (_, _, depth) in self.tracked_objects.items():
#             if depth is not None and depth < stop_distance * 100:  # depth is in cm, convert stop_distance to cm
#                 self.get_logger().info(f'Object ID {person_id} is too close: {depth / 100:.2f} meters. Stopping force drive.')
#                 twist_msg = Twist()
#                 twist_msg.linear.x = 0.0
#                 twist_msg.angular.z = 0.0
#                 self.publisher_cmd_vel.publish(twist_msg)
#                 time.sleep(0.1)
#                 return


#         camera_center_x = self.current_color_image.shape[1] // 2
#         angular_z = 0.0

#         # Calculate how far the center of the bbox is from the camera's center
#         offset_x = bbox_center[0] - camera_center_x

#         # Maximum angular speed setting (e.g., set to 0.35)
#         max_angular_speed = 0.35

#         # Adjust rotational speed proportionally based on the offset from the camera center
#         angular_z = -max_angular_speed * (offset_x / camera_center_x)

#         # Clamp angular_z to be no larger than max_angular_speed and no smaller than -max_angular_speed
#         angular_z = max(-max_angular_speed, min(max_angular_speed, angular_z))

#         self.get_logger().info(f'Calculated angular_z: {angular_z:.2f} for offset: {offset_x}')

#         # If no object is too close, proceed with driving
#         safty_distance_diff = depth_value - (stop_distance * 100)
#         self.get_logger().info(f'Object ID {id_to_follow} distance: {depth_value}cm, Distance between Object and Safety Zone: {safty_distance_diff}cm')

#         twist_msg = Twist()
#         twist_msg.linear.x = 0.1  # Move forward
#         twist_msg.angular.z = angular_z  # Adjust orientation
#         self.publisher_cmd_vel.publish(twist_msg)
#         time.sleep(0.1)


# def main(args=None):
#     rclpy.init(args=args)
#     node = AllInOneNode()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()










# import collections
# if not hasattr(collections, 'Mapping'):
#     import collections.abc
#     collections.Mapping = collections.abc.Mapping

# import cv2
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import CompressedImage
# from geometry_msgs.msg import Twist
# from geometry_msgs.msg import Point
# from cv_bridge import CvBridge
# import numpy as np
# import torch
# from fastreid.config import get_cfg
# from fastreid.engine import DefaultPredictor
# from fastreid.data.transforms import build_transforms
# from PIL import Image as PILImage
# from ultralytics import YOLO
# from message.msg import IdDistanceInfo
# import time

# # CUDA device configuration
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # YOLOv5 model loading
# model = YOLO("yolov5nu.pt").to(device)

# # BotSORT configuration file path
# tracker_config = "/home/jinjuuk/.local/lib/python3.10/site-packages/ultralytics/cfg/trackers/botsort.yaml"

# # FastReID configuration
# cfg = get_cfg()
# cfg.merge_from_file("/home/jinjuuk/fast-reid/configs/Market1501/bagtricks_R50.yml")
# cfg.MODEL.WEIGHTS = "/home/jinjuuk/dev_ws/pt_files/market_mgn_R50-ibn.pth"
# cfg.MODEL.DEVICE = "cuda"  # GPU usage

# predictor = DefaultPredictor(cfg)
# transform = build_transforms(cfg, is_train=False)

# # Dictionary to store known persons' feature vectors and IDs
# known_persons = {}

# # Initial setting
# SIMILARITY_THRESHOLD = 0.97

# class AllInOneNode(Node):
#     def __init__(self):
#         super().__init__('all_in_one_node')

#         # Subscribers
#         self.subscription_color = self.create_subscription(
#             CompressedImage,
#             '/camera/color/image_raw/compressed',
#             self.color_image_callback,
#             10
#         )
#         self.subscription_depth = self.create_subscription(
#             CompressedImage,
#             '/camera/depth/image_raw/compressed',
#             self.depth_image_callback,
#             10
#         )

#         # Publishers
#         self.publisher_cmd_vel = self.create_publisher(
#             Twist,
#             'cmd_vel',
#             10
#         )
#         self.publisher_id_distance = self.create_publisher(
#             IdDistanceInfo,
#             'id_distance_info',
#             10
#         )

#         # Initialize CvBridge for OpenCV
#         self.bridge = CvBridge()

#         # Tracking information storage
#         self.tracked_objects = {}  # id -> (bbox_center, bbox, depth)
#         self.id_last_seen = {}  # id -> last seen time
#         self.last_valid_depth = {}  # To store the last valid depth value for each person_id
#         self.closest_obstacle_distance = float('inf')  # To store the closest distance to an obstacle
#         self.similarity_scores_buffer = collections.defaultdict(lambda: collections.deque(maxlen=5))  # Store similarity scores over multiple frames

#     def get_unique_color(self, id):
#         """Generate a unique color based on ID"""
#         np.random.seed(id)  # Seed with the ID for consistency
#         return tuple(np.random.randint(0, 255, 3).tolist())

#     def depth_image_callback(self, msg):
#         # Decode the compressed depth image
#         np_arr = np.frombuffer(msg.data, np.uint8)
#         depth_frame = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

#         # Calculate the closest obstacle distance
#         self.closest_obstacle_distance = np.min(depth_frame) / 10.0  # Convert minimum depth value to cm

#         active_ids = set()
#         current_time = self.get_clock().now()

#         for person_id, (bbox_center, bbox, _) in self.tracked_objects.items():
#             x1, y1, x2, y2 = bbox
#             x, y = bbox_center

#             if 0 <= x < depth_frame.shape[1] and 0 <= y < depth_frame.shape[0]:
#                 depth_value = round(depth_frame[y, x] / 10, 2)  # Convert mm to cm with two decimal places
#                 if depth_value == 0:
#                     if person_id in self.last_valid_depth:
#                         depth_value = self.last_valid_depth[person_id]
#                     else:
#                         continue
#                 else:
#                     self.last_valid_depth[person_id] = depth_value
#             else:
#                 if person_id in self.last_valid_depth:
#                     depth_value = self.last_valid_depth[person_id]
#                 else:
#                     depth_value = None

#             self.tracked_objects[person_id] = (bbox_center, bbox, depth_value)
#             self.id_last_seen[person_id] = current_time
#             active_ids.add(person_id)

#             if 1 in self.tracked_objects:
#                 self.force_drive(id_to_follow=1)

#         cv2.imshow("Depth Image with Tracking", depth_frame)
#         cv2.waitKey(1)

#     def color_image_callback(self, msg):
#         # Decode the compressed color image
#         np_arr = np.frombuffer(msg.data, np.uint8)
#         frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#         self.current_color_image = frame.copy()

#         results = model.track(classes=[0], max_det=4, show=False, source=frame, stream=True, tracker=tracker_config)
#         results = list(results)

#         active_ids = set()
#         current_time = self.get_clock().now()

#         for result in results:
#             boxes = result.boxes.xyxy.cpu().numpy().astype(int) if result.boxes is not None else []
#             track_ids = result.boxes.id.cpu().numpy().astype(int) if result.boxes.id is not None else []

#             for box, track_id in zip(boxes, track_ids):
#                 x1, y1, x2, y2 = box
#                 person_img = frame[y1:y2, x1:x2]

#                 global_features = self.extract_global_features(person_img)
#                 upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram = self.extract_local_features(person_img)

#                 max_similarity = -1
#                 max_similarity_id = -1

#                 for known_id, (known_global_features, known_upper_body_features, known_lower_body_features, known_upper_body_histogram, known_lower_body_histogram) in known_persons.items():
#                     similarity = self.calculate_combined_similarity(
#                         global_features,
#                         known_global_features,
#                         upper_body_features,
#                         known_upper_body_features,
#                         lower_body_features,
#                         known_lower_body_features,
#                         upper_body_histogram,
#                         known_upper_body_histogram,
#                         lower_body_histogram,
#                         known_lower_body_histogram,
#                     )

#                     self.similarity_scores_buffer[known_id].append(similarity)
#                     avg_similarity = np.mean(self.similarity_scores_buffer[known_id])

#                     if avg_similarity > max_similarity:
#                         max_similarity = avg_similarity
#                         max_similarity_id = known_id

#                 if max_similarity > SIMILARITY_THRESHOLD:
#                     person_id = max_similarity_id
#                 else:
#                     person_id = int(track_id)
#                     if person_id not in known_persons:
#                         new_id = max(known_persons.keys(), default=0) + 1
#                         person_id = new_id
#                     known_persons[person_id] = (global_features, upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram)

#                 self.id_last_seen[person_id] = current_time
#                 active_ids.add(person_id)

#                 bbox_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
#                 bbox = (x1, y1, x2, y2)
#                 self.tracked_objects[person_id] = (bbox_center, bbox, self.tracked_objects.get(person_id, (None, None, None))[2])

#                 color = self.get_unique_color(person_id)
#                 cv2.rectangle(self.current_color_image, (x1, y1), (x2, y2), color, 2)
#                 label = f"ID: {person_id}"
#                 text_color = (255, 255, 255)

#                 center_x = (x1 + x2) // 2
#                 center_y = (y1 + y2) // 2
#                 centp = f"Center: ({center_x}, {center_y})"

#                 depth_value = self.tracked_objects[person_id][2]
#                 depth_text = f"Depth: {depth_value}cm" if depth_value is not None else "Depth: N/A"

#                 (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
#                 (center_text_width, center_text_height), _ = cv2.getTextSize(centp, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
#                 (depth_text_width, depth_text_height), _ = cv2.getTextSize(depth_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)

#                 label_y_position = y2 + text_height + 10
#                 center_text_y_position = label_y_position + center_text_height + 10
#                 depth_text_y_position = center_text_y_position + depth_text_height + 10

#                 cv2.rectangle(self.current_color_image, (x1, y2), (x1 + text_width, label_y_position), color, thickness=cv2.FILLED)
#                 cv2.putText(self.current_color_image, label, (x1, y2 + text_height), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
#                 cv2.rectangle(self.current_color_image, (x1, label_y_position + 5), (x1 + center_text_width, center_text_y_position), color, thickness=cv2.FILLED)
#                 cv2.putText(self.current_color_image, centp, (x1, label_y_position + center_text_height + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
#                 cv2.rectangle(self.current_color_image, (x1, center_text_y_position + 5), (x1 + depth_text_width, depth_text_y_position), color, thickness=cv2.FILLED)
#                 cv2.putText(self.current_color_image, depth_text, (x1, center_text_y_position + depth_text_height + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

#         cv2.imshow("Color Image with Tracking", self.current_color_image)
#         cv2.waitKey(1)

#     def publish_id_distance_info(self, id, bbox_center, width, height, depth_value):
#         msg = IdDistanceInfo()
#         msg.id = id
#         msg.bbox_center = Point(x=float(bbox_center[0]), y=float(bbox_center[1]), z=0.0)
#         msg.width = float(width)
#         msg.height = float(height)
#         msg.distance = float(depth_value)
#         self.publisher_id_distance.publish(msg)

#     def extract_global_features(self, image):
#         pil_img = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#         input_img = transform(pil_img).unsqueeze(0).to(device)
#         features = predictor(input_img).cpu().numpy()
#         return features

#     def extract_local_features(self, image):
#         height, width, _ = image.shape
#         segments = [
#             image[:height//2, :],  
#             image[height//2:, :],  
#             image[:, :width//2],   
#             image[:, width//2:]    
#         ]
        
#         local_features = []
#         local_histograms = []
#         for segment in segments:
#             pil_img = PILImage.fromarray(cv2.cvtColor(segment, cv2.COLOR_BGR2RGB))
#             input_img = transform(pil_img).unsqueeze(0).to(device)
#             features = predictor(input_img).cpu().numpy()
#             hist = self.calculate_histogram(segment)
#             local_features.append(features)
#             local_histograms.append(hist)
        
#         upper_body_features = local_features[0]  
#         lower_body_features = local_features[1]  
        
#         upper_body_histogram = local_histograms[0]  
#         lower_body_histogram = local_histograms[1]  

#         return upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram

#     def calculate_histogram(self, image):
#         hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#         hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
#         cv2.normalize(hist, hist)
#         return hist.flatten()

#     def calculate_combined_similarity(self, global_features1, global_features2, upper_body_features1, upper_body_features2, lower_body_features1, lower_body_features2, upper_body_histogram1, upper_body_histogram2, lower_body_histogram1, lower_body_histogram2):
#         global_similarity = np.dot(global_features1, global_features2.T) / (np.linalg.norm(global_features1) * np.linalg.norm(global_features2))
#         upper_body_similarity = np.dot(upper_body_features1, upper_body_features2.T) / (np.linalg.norm(upper_body_features1) * np.linalg.norm(upper_body_features2))
#         lower_body_similarity = np.dot(lower_body_features1, lower_body_features2.T) / (np.linalg.norm(lower_body_features1) * np.linalg.norm(lower_body_features2))
#         upper_body_hist_similarity = cv2.compareHist(upper_body_histogram1, upper_body_histogram2, cv2.HISTCMP_CORREL)
#         lower_body_hist_similarity = cv2.compareHist(lower_body_histogram1, lower_body_histogram2, cv2.HISTCMP_CORREL)
        
#         combined_similarity = (0.3 * global_similarity +  
#                                 0.1 * upper_body_similarity +
#                                 0.3 * lower_body_similarity +
#                                 0.3 * lower_body_hist_similarity)
#         return combined_similarity

#     def force_drive(self, id_to_follow=1, duration=1.0, stop_distance=0.7):
#         if id_to_follow not in self.tracked_objects:
#             return

#         bbox_center, _, depth_value = self.tracked_objects[id_to_follow]

#         if not self.tracked_objects:
#             twist_msg = Twist()
#             twist_msg.linear.x = 0.0
#             twist_msg.angular.z = 0.0
#             self.publisher_cmd_vel.publish(twist_msg)
#             time.sleep(0.1)
#             return

#         for person_id, (_, _, depth) in self.tracked_objects.items():
#             if depth is not None and depth < stop_distance * 100:
#                 twist_msg = Twist()
#                 twist_msg.linear.x = 0.0
#                 twist_msg.angular.z = 0.0
#                 self.publisher_cmd_vel.publish(twist_msg)
#                 time.sleep(0.1)
#                 return

#         camera_center_x = self.current_color_image.shape[1] // 2
#         angular_z = 0.0

#         offset_x = bbox_center[0] - camera_center_x

#         max_angular_speed = 0.35
#         angular_z = -max_angular_speed * (offset_x / camera_center_x)
#         angular_z = max(-max_angular_speed, min(max_angular_speed, angular_z))

#         safty_distance_diff = depth_value - (stop_distance * 100)

#         twist_msg = Twist()
#         twist_msg.linear.x = 0.1
#         twist_msg.angular.z = angular_z
#         self.publisher_cmd_vel.publish(twist_msg)
#         time.sleep(0.1)

# def main(args=None):
#     rclpy.init(args=args)
#     node = AllInOneNode()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()








# 압축, 해상도 적용된 원본 코드 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# import collections
# if not hasattr(collections, 'Mapping'):
#     import collections.abc
#     collections.Mapping = collections.abc.Mapping

# import cv2
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import CompressedImage
# from geometry_msgs.msg import Twist, Point
# from cv_bridge import CvBridge
# import numpy as np
# import torch
# from fastreid.config import get_cfg
# from fastreid.engine import DefaultPredictor
# from fastreid.data.transforms import build_transforms
# from PIL import Image as PILImage
# from ultralytics import YOLO
# from message.msg import IdDistanceInfo
# import time

# # CUDA device configuration
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # YOLOv5 model loading
# model = YOLO("yolov5su.pt").to(device)

# # BotSORT configuration file path
# tracker_config = "/home/jinjuuk/.local/lib/python3.10/site-packages/ultralytics/cfg/trackers/botsort.yaml"

# # FastReID configuration
# cfg = get_cfg()
# cfg.merge_from_file("/home/jinjuuk/fast-reid/configs/Market1501/bagtricks_R50.yml")
# cfg.MODEL.WEIGHTS = "/home/jinjuuk/dev_ws/pt_files/market_mgn_R50-ibn.pth"
# cfg.MODEL.DEVICE = "cuda"  # GPU usage

# predictor = DefaultPredictor(cfg)
# transform = build_transforms(cfg, is_train=False)

# # Dictionary to store known persons' feature vectors and IDs
# known_persons = {}

# # Initial setting
# SIMILARITY_THRESHOLD = 0.90 #0.97

# class AllInOneNode(Node):
#     def __init__(self):
#         super().__init__('all_in_one_node')

#         # Subscribers
#         self.subscription_color = self.create_subscription(
#             CompressedImage,
#             '/camera/color/image_raw/compressed',
#             self.color_image_callback,
#             10
#         )
#         self.subscription_depth = self.create_subscription(
#             CompressedImage,
#             '/camera/depth/image_raw/compressed',
#             self.depth_image_callback,
#             10
#         )

#         # Publishers
#         self.publisher_cmd_vel = self.create_publisher(
#             Twist,
#             'cmd_vel',
#             10
#         )
#         self.publisher_id_distance = self.create_publisher(
#             IdDistanceInfo,
#             'id_distance_info',
#             10
#         )

#         # Initialize CvBridge for OpenCV
#         self.bridge = CvBridge()

#         # Tracking information storage
#         self.tracked_objects = {}  # id -> (bbox_center, bbox, depth)
#         self.id_last_seen = {}  # id -> last seen time
#         self.last_valid_depth = {}  # To store the last valid depth value for each person_id
#         self.closest_obstacle_distance = float('inf')  # To store the closest distance to an obstacle
#         self.similarity_scores_buffer = collections.defaultdict(lambda: collections.deque(maxlen=5))  # Store similarity scores over multiple frames

#     def get_unique_color(self, id):
#         """Generate a unique color based on ID"""
#         np.random.seed(id)  # Seed with the ID for consistency
#         return tuple(np.random.randint(0, 255, 3).tolist())

#     def depth_image_callback(self, msg):
#         # Decode the compressed depth image
#         np_arr = np.frombuffer(msg.data, np.uint8)
#         depth_frame = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

#         # Ensure the image size is 320x240
#         if depth_frame.shape != (320, 320):
#             self.get_logger().warning(f"Unexpected depth image size: {depth_frame.shape}. Expected (320, 320).")

#         # Calculate the closest obstacle distance
#         self.closest_obstacle_distance = np.min(depth_frame) / 10.0  # Convert minimum depth value to cm

#         active_ids = set()
#         current_time = self.get_clock().now()

#         for person_id, (bbox_center, bbox, _) in self.tracked_objects.items():
#             x1, y1, x2, y2 = bbox
#             x, y = bbox_center

#             if 0 <= x < depth_frame.shape[1] and 0 <= y < depth_frame.shape[0]:
#                 depth_value = round(depth_frame[y, x] / 10, 2)  # Convert mm to cm with two decimal places
#                 if depth_value == 0:
#                     if person_id in self.last_valid_depth:
#                         depth_value = self.last_valid_depth[person_id]
#                     else:
#                         continue
#                 else:
#                     self.last_valid_depth[person_id] = depth_value
#             else:
#                 if person_id in self.last_valid_depth:
#                     depth_value = self.last_valid_depth[person_id]
#                 else:
#                     depth_value = None

#             self.tracked_objects[person_id] = (bbox_center, bbox, depth_value)
#             self.id_last_seen[person_id] = current_time
#             active_ids.add(person_id)

#             if 1 in self.tracked_objects:
#                 self.force_drive(id_to_follow=1)

#         # cv2.imshow("Depth Image with Tracking", depth_frame)
#         # cv2.waitKey(1)

#     def color_image_callback(self, msg):
#         # Decode the compressed color image
#         np_arr = np.frombuffer(msg.data, np.uint8)
#         frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

#         # Ensure the image size is 320x240
#         if frame.shape != (320, 320, 3):
#             self.get_logger().warning(f"Unexpected color image size: {frame.shape}. Expected (320, 320, 3).")

#         self.current_color_image = frame.copy()

#         results = model.track(classes=[0], max_det=3, show=False, source=frame, stream=True, tracker=tracker_config, conf = 0.70, iou = 0.75)
#         results = list(results)

#         active_ids = set()
#         current_time = self.get_clock().now()

#         for result in results:
#             boxes = result.boxes.xyxy.cpu().numpy().astype(int) if result.boxes is not None else []
#             track_ids = result.boxes.id.cpu().numpy().astype(int) if result.boxes.id is not None else []

#             for box, track_id in zip(boxes, track_ids):
#                 x1, y1, x2, y2 = box
#                 person_img = frame[y1:y2, x1:x2]

#                 global_features = self.extract_global_features(person_img)
#                 upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram = self.extract_local_features(person_img)

#                 max_similarity = -1
#                 max_similarity_id = -1

#                 for known_id, (known_global_features, known_upper_body_features, known_lower_body_features, known_upper_body_histogram, known_lower_body_histogram) in known_persons.items():
#                     similarity = self.calculate_combined_similarity(
#                         global_features,
#                         known_global_features,
#                         upper_body_features,
#                         known_upper_body_features,
#                         lower_body_features,
#                         known_lower_body_features,
#                         upper_body_histogram,
#                         known_upper_body_histogram,
#                         lower_body_histogram,
#                         known_lower_body_histogram,
#                     )

#                     self.similarity_scores_buffer[known_id].append(similarity)
#                     avg_similarity = np.mean(self.similarity_scores_buffer[known_id])

#                     if avg_similarity > max_similarity:
#                         max_similarity = avg_similarity
#                         max_similarity_id = known_id

#                 if max_similarity > SIMILARITY_THRESHOLD:
#                     person_id = max_similarity_id
#                 else:
#                     person_id = int(track_id)
#                     if person_id not in known_persons:
#                         new_id = max(known_persons.keys(), default=0) + 1
#                         person_id = new_id
#                     known_persons[person_id] = (global_features, upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram)

#                 self.id_last_seen[person_id] = current_time
#                 active_ids.add(person_id)

#                 bbox_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
#                 bbox = (x1, y1, x2, y2)
#                 self.tracked_objects[person_id] = (bbox_center, bbox, self.tracked_objects.get(person_id, (None, None, None))[2])

#                 color = self.get_unique_color(person_id)
#                 cv2.rectangle(self.current_color_image, (x1, y1), (x2, y2), color, 2)
#                 label = f"ID: {person_id}"
#                 text_color = (255, 255, 255)

#                 center_x = (x1 + x2) // 2
#                 center_y = (y1 + y2) // 2
#                 centp = f"Center: ({center_x}, {center_y})"

#                 depth_value = self.tracked_objects[person_id][2]
#                 depth_text = f"Depth: {depth_value}cm" if depth_value is not None else "Depth: N/A"

#                 (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
#                 (center_text_width, center_text_height), _ = cv2.getTextSize(centp, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
#                 (depth_text_width, depth_text_height), _ = cv2.getTextSize(depth_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)

#                 label_y_position = y2 + text_height + 10
#                 center_text_y_position = label_y_position + center_text_height + 10
#                 depth_text_y_position = center_text_y_position + depth_text_height + 10

#                 cv2.rectangle(self.current_color_image, (x1, y2), (x1 + text_width, label_y_position), color, thickness=cv2.FILLED)
#                 cv2.putText(self.current_color_image, label, (x1, y2 + text_height), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
#                 cv2.rectangle(self.current_color_image, (x1, label_y_position + 5), (x1 + center_text_width, center_text_y_position), color, thickness=cv2.FILLED)
#                 cv2.putText(self.current_color_image, centp, (x1, label_y_position + center_text_height + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
#                 cv2.rectangle(self.current_color_image, (x1, center_text_y_position + 5), (x1 + depth_text_width, depth_text_y_position), color, thickness=cv2.FILLED)
#                 cv2.putText(self.current_color_image, depth_text, (x1, center_text_y_position + depth_text_height + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

#         cv2.imshow("Color Image with Tracking", self.current_color_image)
#         cv2.waitKey(1)

#     def publish_id_distance_info(self, id, bbox_center, width, height, depth_value):
#         msg = IdDistanceInfo()
#         msg.id = id
#         msg.bbox_center = Point(x=float(bbox_center[0]), y=float(bbox_center[1]), z=0.0)
#         msg.width = float(width)
#         msg.height = float(height)
#         msg.distance = float(depth_value)
#         self.publisher_id_distance.publish(msg)

#     def extract_global_features(self, image):
#         pil_img = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#         input_img = transform(pil_img).unsqueeze(0).to(device)
#         features = predictor(input_img).cpu().numpy()
#         return features

#     def extract_local_features(self, image):
#         height, width, _ = image.shape
#         segments = [
#             image[:height//2, :],  
#             image[height//2:, :],  
#             image[:, :width//2],   
#             image[:, width//2:]    
#         ]
        
#         local_features = []
#         local_histograms = []
#         for segment in segments:
#             pil_img = PILImage.fromarray(cv2.cvtColor(segment, cv2.COLOR_BGR2RGB))
#             input_img = transform(pil_img).unsqueeze(0).to(device)
#             features = predictor(input_img).cpu().numpy()
#             hist = self.calculate_histogram(segment)
#             local_features.append(features)
#             local_histograms.append(hist)
        
#         upper_body_features = local_features[0]  
#         lower_body_features = local_features[1]  
        
#         upper_body_histogram = local_histograms[0]  
#         lower_body_histogram = local_histograms[1]  

#         return upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram

#     def calculate_histogram(self, image):
#         hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#         hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
#         cv2.normalize(hist, hist)
#         return hist.flatten()

#     def calculate_combined_similarity(self, global_features1, global_features2, upper_body_features1, upper_body_features2, lower_body_features1, lower_body_features2, upper_body_histogram1, upper_body_histogram2, lower_body_histogram1, lower_body_histogram2):
#         global_similarity = np.dot(global_features1, global_features2.T) / (np.linalg.norm(global_features1) * np.linalg.norm(global_features2))
#         upper_body_similarity = np.dot(upper_body_features1, upper_body_features2.T) / (np.linalg.norm(upper_body_features1) * np.linalg.norm(upper_body_features2))
#         lower_body_similarity = np.dot(lower_body_features1, lower_body_features2.T) / (np.linalg.norm(lower_body_features1) * np.linalg.norm(lower_body_features2))
#         upper_body_hist_similarity = cv2.compareHist(upper_body_histogram1, upper_body_histogram2, cv2.HISTCMP_CORREL)
#         lower_body_hist_similarity = cv2.compareHist(lower_body_histogram1, lower_body_histogram2, cv2.HISTCMP_CORREL)
        
#         combined_similarity = (0.2 * global_similarity +  # 0.3
#                                 0.2 * upper_body_similarity + # 0.2
#                                 0.2 * lower_body_similarity + # 0.25
#                                 0.4 * lower_body_hist_similarity) # 0.25
#         return combined_similarity

#     def force_drive(self, id_to_follow=1, duration=1.0, stop_distance=0.7):
#         if id_to_follow not in self.tracked_objects:
#             return

#         bbox_center, _, depth_value = self.tracked_objects[id_to_follow]

#         if not self.tracked_objects:
#             twist_msg = Twist()
#             twist_msg.linear.x = 0.0
#             twist_msg.angular.z = 0.0
#             self.publisher_cmd_vel.publish(twist_msg)
#             time.sleep(0.1)
#             return

#         for person_id, (_, _, depth) in self.tracked_objects.items():
#             if depth is not None and depth < stop_distance * 100:
#                 twist_msg = Twist()
#                 twist_msg.linear.x = 0.0
#                 twist_msg.angular.z = 0.0
#                 self.publisher_cmd_vel.publish(twist_msg)
#                 time.sleep(0.1)
#                 return

#         camera_center_x = self.current_color_image.shape[1] // 2
#         angular_z = 0.0

#         offset_x = bbox_center[0] - camera_center_x

#         max_angular_speed = 0.35
#         angular_z = -max_angular_speed * (offset_x / camera_center_x)
#         angular_z = max(-max_angular_speed, min(max_angular_speed, angular_z))

#         safty_distance_diff = depth_value - (stop_distance * 100)

#         twist_msg = Twist()
#         twist_msg.linear.x = 0.1
#         twist_msg.angular.z = angular_z
#         self.publisher_cmd_vel.publish(twist_msg)
#         time.sleep(0.1)

# def main(args=None):
#     rclpy.init(args=args)
#     node = AllInOneNode()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()







# 압축, 해상도 + 상의 히스토그램 + 트랙바!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
import collections
if not hasattr(collections, 'Mapping'):
    import collections.abc
    collections.Mapping = collections.abc.Mapping

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist, Point
from cv_bridge import CvBridge
import numpy as np
import torch
from fastreid.config import get_cfg
from fastreid.engine import DefaultPredictor
from fastreid.data.transforms import build_transforms
from PIL import Image as PILImage
from ultralytics import YOLO
from message.msg import IdDistanceInfo
import time

# CUDA device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# YOLOv5 model loading
model = YOLO("yolov5su.pt").to(device)

# BotSORT configuration file path
tracker_config = "/home/jinjuuk/.local/lib/python3.10/site-packages/ultralytics/cfg/trackers/botsort.yaml"

# FastReID configuration
cfg = get_cfg()
cfg.merge_from_file("/home/jinjuuk/fast-reid/configs/Market1501/bagtricks_R50.yml")
cfg.MODEL.WEIGHTS = "/home/jinjuuk/dev_ws/pt_files/market_mgn_R50-ibn.pth"
cfg.MODEL.DEVICE = "cuda"  # GPU usage

predictor = DefaultPredictor(cfg)
transform = build_transforms(cfg, is_train=False)

# Dictionary to store known persons' feature vectors and IDs
known_persons = {}

# Initial setting
SIMILARITY_THRESHOLD = 90  # 0.90 in percentage
CONF_THRESHOLD = 70  # 0.70 in percentage
IOU_THRESHOLD = 75  # 0.75 in percentage

# Weights for combined similarity
WEIGHT_GLOBAL = 20  # 20%
WEIGHT_UPPER_BODY = 15  # 15%
WEIGHT_LOWER_BODY = 15  # 15%
WEIGHT_UPPER_BODY_HIST = 25  # 25%
WEIGHT_LOWER_BODY_HIST = 25  # 25%

class AllInOneNode(Node):
    def __init__(self):
        super().__init__('all_in_one_node')

        # Subscribers
        self.subscription_color = self.create_subscription(
            CompressedImage,
            '/camera/color/image_raw/compressed',
            self.color_image_callback,
            10
        )
        self.subscription_depth = self.create_subscription(
            CompressedImage,
            '/camera/depth/image_raw/compressed',
            self.depth_image_callback,
            10
        )

        # Publishers
        self.publisher_cmd_vel = self.create_publisher(
            Twist,
            'cmd_vel',
            10
        )
        self.publisher_id_distance = self.create_publisher(
            IdDistanceInfo,
            'id_distance_info',
            10
        )

        # Initialize CvBridge for OpenCV
        self.bridge = CvBridge()

        # Tracking information storage
        self.tracked_objects = {}  # id -> (bbox_center, bbox, depth)
        self.id_last_seen = {}  # id -> last seen time
        self.last_valid_depth = {}  # To store the last valid depth value for each person_id
        self.closest_obstacle_distance = float('inf')  # To store the closest distance to an obstacle
        self.similarity_scores_buffer = collections.defaultdict(lambda: collections.deque(maxlen=5))  # Store similarity scores over multiple frames

        # Create trackbars for parameter adjustment
        self.create_trackbars()

    def create_trackbars(self):
        """Create OpenCV windows with trackbars for adjusting parameters."""
        cv2.namedWindow("Trackbars")

        cv2.createTrackbar("Confidence", "Trackbars", CONF_THRESHOLD, 100, self.on_conf_trackbar)
        cv2.createTrackbar("IoU", "Trackbars", IOU_THRESHOLD, 100, self.on_iou_trackbar)
        cv2.createTrackbar("Similarity Threshold", "Trackbars", SIMILARITY_THRESHOLD, 100, self.on_similarity_threshold_trackbar)
        cv2.createTrackbar("Weight Global", "Trackbars", WEIGHT_GLOBAL, 100, self.on_weight_trackbar)
        cv2.createTrackbar("Weight Upper Body", "Trackbars", WEIGHT_UPPER_BODY, 100, self.on_weight_trackbar)
        cv2.createTrackbar("Weight Lower Body", "Trackbars", WEIGHT_LOWER_BODY, 100, self.on_weight_trackbar)
        cv2.createTrackbar("Weight Upper Body Hist", "Trackbars", WEIGHT_UPPER_BODY_HIST, 100, self.on_weight_trackbar)
        cv2.createTrackbar("Weight Lower Body Hist", "Trackbars", WEIGHT_LOWER_BODY_HIST, 100, self.on_weight_trackbar)

    def on_conf_trackbar(self, value):
        global CONF_THRESHOLD
        CONF_THRESHOLD = value

    def on_iou_trackbar(self, value):
        global IOU_THRESHOLD
        IOU_THRESHOLD = value

    def on_similarity_threshold_trackbar(self, value):
        global SIMILARITY_THRESHOLD
        SIMILARITY_THRESHOLD = value / 100.0

    def on_weight_trackbar(self, value):
        """Callback function for trackbar to adjust weights and ensure they sum up to 100%."""
        global WEIGHT_GLOBAL, WEIGHT_UPPER_BODY, WEIGHT_LOWER_BODY, WEIGHT_UPPER_BODY_HIST, WEIGHT_LOWER_BODY_HIST

        # Get the current trackbar positions
        WEIGHT_GLOBAL = cv2.getTrackbarPos("Weight Global", "Trackbars")
        WEIGHT_UPPER_BODY = cv2.getTrackbarPos("Weight Upper Body", "Trackbars")
        WEIGHT_LOWER_BODY = cv2.getTrackbarPos("Weight Lower Body", "Trackbars")
        WEIGHT_UPPER_BODY_HIST = cv2.getTrackbarPos("Weight Upper Body Hist", "Trackbars")
        WEIGHT_LOWER_BODY_HIST = cv2.getTrackbarPos("Weight Lower Body Hist", "Trackbars")

        # Normalize weights to sum up to 100%
        total_weight = WEIGHT_GLOBAL + WEIGHT_UPPER_BODY + WEIGHT_LOWER_BODY + WEIGHT_UPPER_BODY_HIST + WEIGHT_LOWER_BODY_HIST
        if total_weight > 0:
            WEIGHT_GLOBAL = int((WEIGHT_GLOBAL / total_weight) * 100)
            WEIGHT_UPPER_BODY = int((WEIGHT_UPPER_BODY / total_weight) * 100)
            WEIGHT_LOWER_BODY = int((WEIGHT_LOWER_BODY / total_weight) * 100)
            WEIGHT_UPPER_BODY_HIST = int((WEIGHT_UPPER_BODY_HIST / total_weight) * 100)
            WEIGHT_LOWER_BODY_HIST = int((WEIGHT_LOWER_BODY_HIST / total_weight) * 100)

        # Update trackbar positions
        cv2.setTrackbarPos("Weight Global", "Trackbars", WEIGHT_GLOBAL)
        cv2.setTrackbarPos("Weight Upper Body", "Trackbars", WEIGHT_UPPER_BODY)
        cv2.setTrackbarPos("Weight Lower Body", "Trackbars", WEIGHT_LOWER_BODY)
        cv2.setTrackbarPos("Weight Upper Body Hist", "Trackbars", WEIGHT_UPPER_BODY_HIST)
        cv2.setTrackbarPos("Weight Lower Body Hist", "Trackbars", WEIGHT_LOWER_BODY_HIST)

    def get_unique_color(self, id):
        """Generate a unique color based on ID"""
        np.random.seed(id)  # Seed with the ID for consistency
        return tuple(np.random.randint(0, 255, 3).tolist())

    def depth_image_callback(self, msg):
        # Decode the compressed depth image
        np_arr = np.frombuffer(msg.data, np.uint8)
        depth_frame = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

        # Ensure the image size is 320x240
        if depth_frame.shape != (320, 320):
            self.get_logger().warning(f"Unexpected depth image size: {depth_frame.shape}. Expected (320, 320).")

        # Calculate the closest obstacle distance
        self.closest_obstacle_distance = np.min(depth_frame) / 10.0  # Convert minimum depth value to cm

        active_ids = set()
        current_time = self.get_clock().now()

        for person_id, (bbox_center, bbox, _) in self.tracked_objects.items():
            x1, y1, x2, y2 = bbox
            x, y = bbox_center

            if 0 <= x < depth_frame.shape[1] and 0 <= y < depth_frame.shape[0]:
                depth_value = round(depth_frame[y, x] / 10, 2)  # Convert mm to cm with two decimal places
                if depth_value == 0:
                    if person_id in self.last_valid_depth:
                        depth_value = self.last_valid_depth[person_id]
                    else:
                        continue
                else:
                    self.last_valid_depth[person_id] = depth_value
            else:
                if person_id in self.last_valid_depth:
                    depth_value = self.last_valid_depth[person_id]
                else:
                    depth_value = None

            self.tracked_objects[person_id] = (bbox_center, bbox, depth_value)
            self.id_last_seen[person_id] = current_time
            active_ids.add(person_id)

            if 1 in self.tracked_objects:
                self.force_drive(id_to_follow=1)

    def color_image_callback(self, msg):
        # Decode the compressed color image
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Ensure the image size is 320x240
        if frame.shape != (320, 320, 3):
            self.get_logger().warning(f"Unexpected color image size: {frame.shape}. Expected (320, 320, 3).")

        self.current_color_image = frame.copy()

        # Use current values of conf and iou thresholds from trackbars
        results = model.track(classes=[0], max_det=3, show=False, source=frame, stream=True, tracker=tracker_config, conf=CONF_THRESHOLD / 100, iou=IOU_THRESHOLD / 100)
        results = list(results)

        active_ids = set()
        current_time = self.get_clock().now()

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy().astype(int) if result.boxes is not None else []
            track_ids = result.boxes.id.cpu().numpy().astype(int) if result.boxes.id is not None else []

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
                person_img = frame[y1:y2, x1:x2]

                global_features = self.extract_global_features(person_img)
                upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram = self.extract_local_features(person_img)

                max_similarity = -1
                max_similarity_id = -1

                for known_id, (known_global_features, known_upper_body_features, known_lower_body_features, known_upper_body_histogram, known_lower_body_histogram) in known_persons.items():
                    similarity = self.calculate_combined_similarity(
                        global_features,
                        known_global_features,
                        upper_body_features,
                        known_upper_body_features,
                        lower_body_features,
                        known_lower_body_features,
                        upper_body_histogram,
                        known_upper_body_histogram,
                        lower_body_histogram,
                        known_lower_body_histogram,
                    )

                    self.similarity_scores_buffer[known_id].append(similarity)
                    avg_similarity = np.mean(self.similarity_scores_buffer[known_id])

                    if avg_similarity > max_similarity:
                        max_similarity = avg_similarity
                        max_similarity_id = known_id

                if max_similarity > SIMILARITY_THRESHOLD / 100:
                    person_id = max_similarity_id
                else:
                    person_id = int(track_id)
                    if person_id not in known_persons:
                        new_id = max(known_persons.keys(), default=0) + 1
                        person_id = new_id
                    known_persons[person_id] = (global_features, upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram)

                self.id_last_seen[person_id] = current_time
                active_ids.add(person_id)

                bbox_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                bbox = (x1, y1, x2, y2)
                self.tracked_objects[person_id] = (bbox_center, bbox, self.tracked_objects.get(person_id, (None, None, None))[2])

                color = self.get_unique_color(person_id)
                cv2.rectangle(self.current_color_image, (x1, y1), (x2, y2), color, 2)
                label = f"ID: {person_id}"
                text_color = (255, 255, 255)

                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                centp = f"Center: ({center_x}, {center_y})"

                depth_value = self.tracked_objects[person_id][2]
                depth_text = f"Depth: {depth_value}cm" if depth_value is not None else "Depth: N/A"

                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                (center_text_width, center_text_height), _ = cv2.getTextSize(centp, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                (depth_text_width, depth_text_height), _ = cv2.getTextSize(depth_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)

                label_y_position = y2 + text_height + 10
                center_text_y_position = label_y_position + center_text_height + 10
                depth_text_y_position = center_text_y_position + depth_text_height + 10

                cv2.rectangle(self.current_color_image, (x1, y2), (x1 + text_width, label_y_position), color, thickness=cv2.FILLED)
                cv2.putText(self.current_color_image, label, (x1, y2 + text_height), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
                cv2.rectangle(self.current_color_image, (x1, label_y_position + 5), (x1 + center_text_width, center_text_y_position), color, thickness=cv2.FILLED)
                cv2.putText(self.current_color_image, centp, (x1, label_y_position + center_text_height + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
                cv2.rectangle(self.current_color_image, (x1, center_text_y_position + 5), (x1 + depth_text_width, depth_text_y_position), color, thickness=cv2.FILLED)
                cv2.putText(self.current_color_image, depth_text, (x1, center_text_y_position + depth_text_height + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

        cv2.imshow("Color Image with Tracking", self.current_color_image)
        cv2.waitKey(1)

    def publish_id_distance_info(self, id, bbox_center, width, height, depth_value):
        msg = IdDistanceInfo()
        msg.id = id
        msg.bbox_center = Point(x=float(bbox_center[0]), y=float(bbox_center[1]), z=0.0)
        msg.width = float(width)
        msg.height = float(height)
        msg.distance = float(depth_value)
        self.publisher_id_distance.publish(msg)

    def extract_global_features(self, image):
        pil_img = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        input_img = transform(pil_img).unsqueeze(0).to(device)
        features = predictor(input_img).cpu().numpy()
        return features

    def extract_local_features(self, image):
        height, width, _ = image.shape
        segments = [
            image[:height//2, :],  
            image[height//2:, :],  
            image[:, :width//2],   
            image[:, width//2:]    
        ]
        
        local_features = []
        local_histograms = []
        for segment in segments:
            pil_img = PILImage.fromarray(cv2.cvtColor(segment, cv2.COLOR_BGR2RGB))
            input_img = transform(pil_img).unsqueeze(0).to(device)
            features = predictor(input_img).cpu().numpy()
            hist = self.calculate_histogram(segment)
            local_features.append(features)
            local_histograms.append(hist)
        
        upper_body_features = local_features[0]  
        lower_body_features = local_features[1]  
        
        upper_body_histogram = local_histograms[0]  
        lower_body_histogram = local_histograms[1]  

        return upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram

    def calculate_histogram(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()

    def calculate_combined_similarity(self, global_features1, global_features2, upper_body_features1, upper_body_features2, lower_body_features1, lower_body_features2, upper_body_histogram1, upper_body_histogram2, lower_body_histogram1, lower_body_histogram2):
        global_similarity = np.dot(global_features1, global_features2.T) / (np.linalg.norm(global_features1) * np.linalg.norm(global_features2))
        upper_body_similarity = np.dot(upper_body_features1, upper_body_features2.T) / (np.linalg.norm(upper_body_features1) * np.linalg.norm(upper_body_features2))
        lower_body_similarity = np.dot(lower_body_features1, lower_body_features2.T) / (np.linalg.norm(lower_body_features1) * np.linalg.norm(lower_body_features2))
        upper_body_hist_similarity = cv2.compareHist(upper_body_histogram1, upper_body_histogram2, cv2.HISTCMP_CORREL)
        lower_body_hist_similarity = cv2.compareHist(lower_body_histogram1, lower_body_histogram2, cv2.HISTCMP_CORREL)

        # Normalize weights
        total_weight = WEIGHT_GLOBAL + WEIGHT_UPPER_BODY + WEIGHT_LOWER_BODY + WEIGHT_UPPER_BODY_HIST + WEIGHT_LOWER_BODY_HIST
        normalized_global_weight = WEIGHT_GLOBAL / total_weight
        normalized_upper_body_weight = WEIGHT_UPPER_BODY / total_weight
        normalized_lower_body_weight = WEIGHT_LOWER_BODY / total_weight
        normalized_upper_body_hist_weight = WEIGHT_UPPER_BODY_HIST / total_weight
        normalized_lower_body_hist_weight = WEIGHT_LOWER_BODY_HIST / total_weight
        
        combined_similarity = (normalized_global_weight * global_similarity +
                               normalized_upper_body_weight * upper_body_similarity +
                               normalized_lower_body_weight * lower_body_similarity +
                               normalized_upper_body_hist_weight * upper_body_hist_similarity +
                               normalized_lower_body_hist_weight * lower_body_hist_similarity)
        return combined_similarity

    def force_drive(self, id_to_follow=1, duration=1.0, stop_distance=0.7):
        if id_to_follow not in self.tracked_objects:
            return

        bbox_center, _, depth_value = self.tracked_objects[id_to_follow]

        if not self.tracked_objects:
            twist_msg = Twist()
            twist_msg.linear.x = 0.0
            twist_msg.angular.z = 0.0
            self.publisher_cmd_vel.publish(twist_msg)
            time.sleep(0.1)
            return

        for person_id, (_, _, depth) in self.tracked_objects.items():
            if depth is not None and depth < stop_distance * 100:
                twist_msg = Twist()
                twist_msg.linear.x = 0.0
                twist_msg.angular.z = 0.0
                self.publisher_cmd_vel.publish(twist_msg)
                time.sleep(0.1)
                return

        camera_center_x = self.current_color_image.shape[1] // 2
        angular_z = 0.0

        offset_x = bbox_center[0] - camera_center_x

        max_angular_speed = 0.35
        angular_z = -max_angular_speed * (offset_x / camera_center_x)
        angular_z = max(-max_angular_speed, min(max_angular_speed, angular_z))

        safty_distance_diff = depth_value - (stop_distance * 100)

        twist_msg = Twist()
        twist_msg.linear.x = 0.1
        twist_msg.angular.z = angular_z
        self.publisher_cmd_vel.publish(twist_msg)
        time.sleep(0.1)

def main(args=None):
    rclpy.init(args=args)
    node = AllInOneNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()







# 압축, 해상도 + 상의 히스토그램 + 트랙바 + ID 1번의 속도와 방향 적용!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
import collections
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist, Point
from cv_bridge import CvBridge
import numpy as np
import torch
from fastreid.config import get_cfg
from fastreid.engine import DefaultPredictor
from fastreid.data.transforms import build_transforms
from PIL import Image as PILImage
from ultralytics import YOLO
from message.msg import IdDistanceInfo
import time
import math

# CUDA device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# YOLOv5 model loading
model = YOLO("yolov5su.pt").to(device)

# BotSORT configuration file path
tracker_config = "/home/jinjuuk/.local/lib/python3.10/site-packages/ultralytics/cfg/trackers/botsort.yaml"

# FastReID configuration
cfg = get_cfg()
cfg.merge_from_file("/home/jinjuuk/fast-reid/configs/Market1501/bagtricks_R50.yml")
cfg.MODEL.WEIGHTS = "/home/jinjuuk/dev_ws/pt_files/market_mgn_R50-ibn.pth"
cfg.MODEL.DEVICE = "cuda"  # GPU usage

predictor = DefaultPredictor(cfg)
transform = build_transforms(cfg, is_train=False)

# Dictionary to store known persons' feature vectors and IDs
known_persons = {}

# Initial setting
SIMILARITY_THRESHOLD = 90  # 0.90 in percentage
CONF_THRESHOLD = 70  # 0.70 in percentage
IOU_THRESHOLD = 75  # 0.75 in percentage

# Weights for combined similarity
WEIGHT_GLOBAL = 20  # 20%
WEIGHT_UPPER_BODY = 15  # 15%
WEIGHT_LOWER_BODY = 15  # 15%
WEIGHT_UPPER_BODY_HIST = 25  # 25%
WEIGHT_LOWER_BODY_HIST = 25  # 25%

# Frame buffer size for averaging similarity scores
FRAME_BUFFER_SIZE = 5  # Default value

class AllInOneNode(Node):
    def __init__(self):
        super().__init__('all_in_one_node')

        # Subscribers
        self.subscription_color = self.create_subscription(
            CompressedImage,
            '/camera/color/image_raw/compressed',
            self.color_image_callback,
            10
        )
        self.subscription_depth = self.create_subscription(
            CompressedImage,
            '/camera/depth/image_raw/compressed',
            self.depth_image_callback,
            10
        )

        # Publishers
        self.publisher_cmd_vel = self.create_publisher(
            Twist,
            'cmd_vel',
            10
        )
        self.publisher_id_distance = self.create_publisher(
            IdDistanceInfo,
            'id_distance_info',
            10
        )

        # Initialize CvBridge for OpenCV
        self.bridge = CvBridge()

        # Tracking information storage
        self.tracked_objects = {}  # id -> (bbox_center, bbox, depth)
        self.previous_positions = {}  # id -> previous bbox_center
        self.id_last_seen = {}  # id -> last seen time
        self.last_valid_depth = {}  # To store the last valid depth value for each person_id
        self.closest_obstacle_distance = float('inf')  # To store the closest distance to an obstacle
        self.similarity_scores_buffer = collections.defaultdict(lambda: collections.deque(maxlen=FRAME_BUFFER_SIZE))  # Store similarity scores over multiple frames

        # Create trackbars for parameter adjustment
        self.create_trackbars()

    def create_trackbars(self):
        """Create OpenCV windows with trackbars for adjusting parameters."""
        cv2.namedWindow("Trackbars")

        cv2.createTrackbar("Confidence", "Trackbars", CONF_THRESHOLD, 100, self.on_conf_trackbar)
        cv2.createTrackbar("IoU", "Trackbars", IOU_THRESHOLD, 100, self.on_iou_trackbar)
        cv2.createTrackbar("Similarity Threshold", "Trackbars", SIMILARITY_THRESHOLD, 100, self.on_similarity_threshold_trackbar)
        cv2.createTrackbar("Weight Global", "Trackbars", WEIGHT_GLOBAL, 100, self.on_weight_trackbar)
        cv2.createTrackbar("Weight Upper Body", "Trackbars", WEIGHT_UPPER_BODY, 100, self.on_weight_trackbar)
        cv2.createTrackbar("Weight Lower Body", "Trackbars", WEIGHT_LOWER_BODY, 100, self.on_weight_trackbar)
        cv2.createTrackbar("Weight Upper Body Hist", "Trackbars", WEIGHT_UPPER_BODY_HIST, 100, self.on_weight_trackbar)
        cv2.createTrackbar("Weight Lower Body Hist", "Trackbars", WEIGHT_LOWER_BODY_HIST, 100, self.on_weight_trackbar)
        cv2.createTrackbar("Frame Buffer Size", "Trackbars", FRAME_BUFFER_SIZE, 20, self.on_frame_buffer_size_trackbar)

    def on_conf_trackbar(self, value):
        global CONF_THRESHOLD
        CONF_THRESHOLD = value

    def on_iou_trackbar(self, value):
        global IOU_THRESHOLD
        IOU_THRESHOLD = value

    def on_similarity_threshold_trackbar(self, value):
        global SIMILARITY_THRESHOLD
        SIMILARITY_THRESHOLD = value

    def on_frame_buffer_size_trackbar(self, value):
        global FRAME_BUFFER_SIZE
        FRAME_BUFFER_SIZE = max(1, value)  # Ensure at least 1 frame for averaging
        self.similarity_scores_buffer = collections.defaultdict(lambda: collections.deque(maxlen=FRAME_BUFFER_SIZE))  # Update buffer size

    def on_weight_trackbar(self, value):
        """Callback function for trackbar to adjust weights and ensure they sum up to 100%."""
        global WEIGHT_GLOBAL, WEIGHT_UPPER_BODY, WEIGHT_LOWER_BODY, WEIGHT_UPPER_BODY_HIST, WEIGHT_LOWER_BODY_HIST

        # Get the current trackbar positions
        WEIGHT_GLOBAL = cv2.getTrackbarPos("Weight Global", "Trackbars")
        WEIGHT_UPPER_BODY = cv2.getTrackbarPos("Weight Upper Body", "Trackbars")
        WEIGHT_LOWER_BODY = cv2.getTrackbarPos("Weight Lower Body", "Trackbars")
        WEIGHT_UPPER_BODY_HIST = cv2.getTrackbarPos("Weight Upper Body Hist", "Trackbars")
        WEIGHT_LOWER_BODY_HIST = cv2.getTrackbarPos("Weight Lower Body Hist", "Trackbars")

        # Normalize weights to sum up to 100%
        total_weight = WEIGHT_GLOBAL + WEIGHT_UPPER_BODY + WEIGHT_LOWER_BODY + WEIGHT_UPPER_BODY_HIST + WEIGHT_LOWER_BODY_HIST
        if total_weight > 0:
            WEIGHT_GLOBAL = int((WEIGHT_GLOBAL / total_weight) * 100)
            WEIGHT_UPPER_BODY = int((WEIGHT_UPPER_BODY / total_weight) * 100)
            WEIGHT_LOWER_BODY = int((WEIGHT_LOWER_BODY / total_weight) * 100)
            WEIGHT_UPPER_BODY_HIST = int((WEIGHT_UPPER_BODY_HIST / total_weight) * 100)
            WEIGHT_LOWER_BODY_HIST = int((WEIGHT_LOWER_BODY_HIST / total_weight) * 100)

        # Update trackbar positions
        cv2.setTrackbarPos("Weight Global", "Trackbars", WEIGHT_GLOBAL)
        cv2.setTrackbarPos("Weight Upper Body", "Trackbars", WEIGHT_UPPER_BODY)
        cv2.setTrackbarPos("Weight Lower Body", "Trackbars", WEIGHT_LOWER_BODY)
        cv2.setTrackbarPos("Weight Upper Body Hist", "Trackbars", WEIGHT_UPPER_BODY_HIST)
        cv2.setTrackbarPos("Weight Lower Body Hist", "Trackbars", WEIGHT_LOWER_BODY_HIST)

    def get_unique_color(self, id):
        """Generate a unique color based on ID"""
        np.random.seed(id)  # Seed with the ID for consistency
        return tuple(np.random.randint(0, 255, 3).tolist())

    def depth_image_callback(self, msg):
        # Decode the compressed depth image
        np_arr = np.frombuffer(msg.data, np.uint8)
        depth_frame = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

        # Ensure the image size is 320x240
        if depth_frame.shape != (320, 320):
            self.get_logger().warning(f"Unexpected depth image size: {depth_frame.shape}. Expected (320, 320).")

        # Calculate the closest obstacle distance
        self.closest_obstacle_distance = np.min(depth_frame) / 10.0  # Convert minimum depth value to cm

        active_ids = set()
        current_time = self.get_clock().now()

        for person_id, (bbox_center, bbox, _) in self.tracked_objects.items():
            x1, y1, x2, y2 = bbox
            x, y = bbox_center

            if 0 <= x < depth_frame.shape[1] and 0 <= y < depth_frame.shape[0]:
                depth_value = round(depth_frame[y, x] / 10, 2)  # Convert mm to cm with two decimal places
                if depth_value == 0:
                    if person_id in self.last_valid_depth:
                        depth_value = self.last_valid_depth[person_id]
                    else:
                        continue
                else:
                    self.last_valid_depth[person_id] = depth_value
            else:
                if person_id in self.last_valid_depth:
                    depth_value = self.last_valid_depth[person_id]
                else:
                    depth_value = None

            self.tracked_objects[person_id] = (bbox_center, bbox, depth_value)
            self.id_last_seen[person_id] = current_time
            active_ids.add(person_id)

            if 1 in self.tracked_objects:
                self.force_drive(id_to_follow=1)

    def color_image_callback(self, msg):
        # Decode the compressed color image
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Ensure the image size is 320x240
        if frame.shape != (320, 320, 3):
            self.get_logger().warning(f"Unexpected color image size: {frame.shape}. Expected (320, 320, 3).")

        self.current_color_image = frame.copy()

        # Use current values of conf and iou thresholds from trackbars
        results = model.track(classes=[0], max_det=3, show=False, source=frame, stream=True, tracker=tracker_config, conf=CONF_THRESHOLD / 100, iou=IOU_THRESHOLD / 100)
        results = list(results)

        active_ids = set()
        current_time = self.get_clock().now()

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy().astype(int) if result.boxes is not None else []
            track_ids = result.boxes.id.cpu().numpy().astype(int) if result.boxes.id is not None else []

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
                person_img = frame[y1:y2, x1:x2]

                global_features = self.extract_global_features(person_img)
                upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram = self.extract_local_features(person_img)

                max_similarity = -1
                max_similarity_id = -1

                for known_id, (known_global_features, known_upper_body_features, known_lower_body_features, known_upper_body_histogram, known_lower_body_histogram) in known_persons.items():
                    similarity = self.calculate_combined_similarity(
                        global_features,
                        known_global_features,
                        upper_body_features,
                        known_upper_body_features,
                        lower_body_features,
                        known_lower_body_features,
                        upper_body_histogram,
                        known_upper_body_histogram,
                        lower_body_histogram,
                        known_lower_body_histogram,
                    )

                    self.similarity_scores_buffer[known_id].append(similarity)
                    avg_similarity = np.mean(list(self.similarity_scores_buffer[known_id])[-FRAME_BUFFER_SIZE:])

                    if avg_similarity > max_similarity:
                        max_similarity = avg_similarity
                        max_similarity_id = known_id

                if max_similarity > SIMILARITY_THRESHOLD / 100:
                    person_id = max_similarity_id
                else:
                    person_id = int(track_id)
                    if person_id not in known_persons:
                        new_id = max(known_persons.keys(), default=0) + 1
                        person_id = new_id
                    known_persons[person_id] = (global_features, upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram)

                self.id_last_seen[person_id] = current_time
                active_ids.add(person_id)

                bbox_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                bbox = (x1, y1, x2, y2)
                self.tracked_objects[person_id] = (bbox_center, bbox, self.tracked_objects.get(person_id, (None, None, None))[2])

                # Calculate direction and speed
                if person_id in self.previous_positions:
                    prev_x, prev_y = self.previous_positions[person_id]
                    dx = bbox_center[0] - prev_x
                    dy = bbox_center[1] - prev_y
                    direction = (dx, dy)
                    speed = math.sqrt(dx ** 2 + dy ** 2)
                else:
                    direction = (0, 0)
                    speed = 0

                # Update previous position
                self.previous_positions[person_id] = bbox_center

                # Display direction and speed on the image
                direction_text = f"Direction: ({direction[0]}, {direction[1]})"
                speed_text = f"Speed: {speed:.2f}px/frame"

                # Draw bounding box, labels, and additional info
                color = self.get_unique_color(person_id)
                cv2.rectangle(self.current_color_image, (x1, y1), (x2, y2), color, 2)
                label = f"ID: {person_id}"
                text_color = (255, 255, 255)

                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                centp = f"Center: ({center_x}, {center_y})"

                depth_value = self.tracked_objects[person_id][2]
                depth_text = f"Depth: {depth_value}cm" if depth_value is not None else "Depth: N/A"

                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                (center_text_width, center_text_height), _ = cv2.getTextSize(centp, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                (depth_text_width, depth_text_height), _ = cv2.getTextSize(depth_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                (direction_text_width, direction_text_height), _ = cv2.getTextSize(direction_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                (speed_text_width, speed_text_height), _ = cv2.getTextSize(speed_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

                label_y_position = y2 + text_height + 10
                center_text_y_position = label_y_position + center_text_height + 10
                depth_text_y_position = center_text_y_position + depth_text_height + 10
                direction_text_y_position = depth_text_y_position + direction_text_height + 10
                speed_text_y_position = direction_text_y_position + speed_text_height + 10

                cv2.rectangle(self.current_color_image, (x1, y2), (x1 + text_width, label_y_position), color, thickness=cv2.FILLED)
                cv2.putText(self.current_color_image, label, (x1, y2 + text_height), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
                cv2.rectangle(self.current_color_image, (x1, label_y_position + 5), (x1 + center_text_width, center_text_y_position), color, thickness=cv2.FILLED)
                cv2.putText(self.current_color_image, centp, (x1, label_y_position + center_text_height + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
                cv2.rectangle(self.current_color_image, (x1, center_text_y_position + 5), (x1 + depth_text_width, depth_text_y_position), color, thickness=cv2.FILLED)
                cv2.putText(self.current_color_image, depth_text, (x1, center_text_y_position + depth_text_height + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
                cv2.rectangle(self.current_color_image, (x1, depth_text_y_position + 5), (x1 + direction_text_width, direction_text_y_position), color, thickness=cv2.FILLED)
                cv2.putText(self.current_color_image, direction_text, (x1, depth_text_y_position + direction_text_height + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
                cv2.rectangle(self.current_color_image, (x1, direction_text_y_position + 5), (x1 + speed_text_width, speed_text_y_position), color, thickness=cv2.FILLED)
                cv2.putText(self.current_color_image, speed_text, (x1, direction_text_y_position + speed_text_height + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

        cv2.imshow("Color Image with Tracking", self.current_color_image)
        cv2.waitKey(1)


    def publish_id_distance_info(self, id, bbox_center, width, height, depth_value):
            msg = IdDistanceInfo()
            msg.id = id
            msg.bbox_center = Point(x=float(bbox_center[0]), y=float(bbox_center[1]), z=0.0)
            msg.width = float(width)
            msg.height = float(height)
            msg.distance = float(depth_value)
            self.publisher_id_distance.publish(msg)

    def extract_global_features(self, image):
        pil_img = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        input_img = transform(pil_img).unsqueeze(0).to(device)
        features = predictor(input_img).cpu().numpy()
        return features

    def extract_local_features(self, image):
        height, width, _ = image.shape
        segments = [
            image[:height//2, :],  
            image[height//2:, :],  
            image[:, :width//2],   
            image[:, width//2:]    
        ]
        
        local_features = []
        local_histograms = []
        for segment in segments:
            pil_img = PILImage.fromarray(cv2.cvtColor(segment, cv2.COLOR_BGR2RGB))
            input_img = transform(pil_img).unsqueeze(0).to(device)
            features = predictor(input_img).cpu().numpy()
            hist = self.calculate_histogram(segment)
            local_features.append(features)
            local_histograms.append(hist)
        
        upper_body_features = local_features[0]  
        lower_body_features = local_features[1]  
        
        upper_body_histogram = local_histograms[0]  
        lower_body_histogram = local_histograms[1]  

        return upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram

    def calculate_histogram(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()

    def calculate_combined_similarity(self, global_features1, global_features2, upper_body_features1, upper_body_features2, lower_body_features1, lower_body_features2, upper_body_histogram1, upper_body_histogram2, lower_body_histogram1, lower_body_histogram2):
        global_similarity = np.dot(global_features1, global_features2.T) / (np.linalg.norm(global_features1) * np.linalg.norm(global_features2))
        upper_body_similarity = np.dot(upper_body_features1, upper_body_features2.T) / (np.linalg.norm(upper_body_features1) * np.linalg.norm(upper_body_features2))
        lower_body_similarity = np.dot(lower_body_features1, lower_body_features2.T) / (np.linalg.norm(lower_body_features1) * np.linalg.norm(lower_body_features2))
        upper_body_hist_similarity = cv2.compareHist(upper_body_histogram1, upper_body_histogram2, cv2.HISTCMP_CORREL)
        lower_body_hist_similarity = cv2.compareHist(lower_body_histogram1, lower_body_histogram2, cv2.HISTCMP_CORREL)

        # Normalize weights
        total_weight = WEIGHT_GLOBAL + WEIGHT_UPPER_BODY + WEIGHT_LOWER_BODY + WEIGHT_UPPER_BODY_HIST + WEIGHT_LOWER_BODY_HIST
        normalized_global_weight = WEIGHT_GLOBAL / total_weight
        normalized_upper_body_weight = WEIGHT_UPPER_BODY / total_weight
        normalized_lower_body_weight = WEIGHT_LOWER_BODY / total_weight
        normalized_upper_body_hist_weight = WEIGHT_UPPER_BODY_HIST / total_weight
        normalized_lower_body_hist_weight = WEIGHT_LOWER_BODY_HIST / total_weight
        
        combined_similarity = (normalized_global_weight * global_similarity +
                            normalized_upper_body_weight * upper_body_similarity +
                            normalized_lower_body_weight * lower_body_similarity +
                            normalized_upper_body_hist_weight * upper_body_hist_similarity +
                            normalized_lower_body_hist_weight * lower_body_hist_similarity)
        return combined_similarity

    def force_drive(self, id_to_follow=1, duration=1.0, stop_distance=0.7):
        if id_to_follow not in self.tracked_objects:
            return

        bbox_center, _, depth_value = self.tracked_objects[id_to_follow]

        if not self.tracked_objects:
            twist_msg = Twist()
            twist_msg.linear.x = 0.0
            twist_msg.angular.z = 0.0
            self.publisher_cmd_vel.publish(twist_msg)
            time.sleep(0.1)
            return

        for person_id, (_, _, depth) in self.tracked_objects.items():
            if depth is not None and depth < stop_distance * 100:
                twist_msg = Twist()
                twist_msg.linear.x = 0.0
                twist_msg.angular.z = 0.0
                self.publisher_cmd_vel.publish(twist_msg)
                time.sleep(0.1)
                return

        camera_center_x = self.current_color_image.shape[1] // 2
        angular_z = 0.0

        offset_x = bbox_center[0] - camera_center_x

        max_angular_speed = 0.35
        angular_z = -max_angular_speed * (offset_x / camera_center_x)
        angular_z = max(-max_angular_speed, min(max_angular_speed, angular_z))

        safty_distance_diff = depth_value - (stop_distance * 100)

        twist_msg = Twist()
        twist_msg.linear.x = 0.1
        twist_msg.angular.z = angular_z
        self.publisher_cmd_vel.publish(twist_msg)
        time.sleep(0.1)

def main(args=None):
    rclpy.init(args=args)
    node = AllInOneNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()













# # 다중프레임 적용!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# import collections
# if not hasattr(collections, 'Mapping'):
#     import collections.abc
#     collections.Mapping = collections.abc.Mapping

# import cv2
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import CompressedImage
# from geometry_msgs.msg import Twist, Point
# from cv_bridge import CvBridge
# import numpy as np
# import torch
# from fastreid.config import get_cfg
# from fastreid.engine import DefaultPredictor
# from fastreid.data.transforms import build_transforms
# from PIL import Image as PILImage
# from ultralytics import YOLO
# from message.msg import IdDistanceInfo
# import time

# # CUDA device configuration
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # YOLOv5 model loading
# model = YOLO("yolov5su.pt").to(device)

# # BotSORT configuration file path
# tracker_config = "/home/jinjuuk/.local/lib/python3.10/site-packages/ultralytics/cfg/trackers/botsort.yaml"

# # FastReID configuration
# cfg = get_cfg()
# cfg.merge_from_file("/home/jinjuuk/fast-reid/configs/Market1501/bagtricks_R50.yml")
# cfg.MODEL.WEIGHTS = "/home/jinjuuk/dev_ws/pt_files/market_mgn_R50-ibn.pth"
# cfg.MODEL.DEVICE = "cuda"  # GPU usage

# predictor = DefaultPredictor(cfg)
# transform = build_transforms(cfg, is_train=False)

# # Dictionary to store known persons' feature vectors and IDs
# known_persons = {}

# # Initial setting
# SIMILARITY_THRESHOLD = 90  # 0.90 in percentage
# CONF_THRESHOLD = 70  # 0.70 in percentage
# IOU_THRESHOLD = 75  # 0.75 in percentage

# # Weights for combined similarity
# WEIGHT_GLOBAL = 20  # 20%
# WEIGHT_UPPER_BODY = 15  # 15%
# WEIGHT_LOWER_BODY = 15  # 15%
# WEIGHT_UPPER_BODY_HIST = 25  # 25%
# WEIGHT_LOWER_BODY_HIST = 25  # 25%

# # Frame buffer size for averaging similarity scores
# FRAME_BUFFER_SIZE = 5  # Default value

# class AllInOneNode(Node):
#     def __init__(self):
#         super().__init__('all_in_one_node')

#         # Subscribers
#         self.subscription_color = self.create_subscription(
#             CompressedImage,
#             '/camera/color/image_raw/compressed',
#             self.color_image_callback,
#             10
#         )
#         self.subscription_depth = self.create_subscription(
#             CompressedImage,
#             '/camera/depth/image_raw/compressed',
#             self.depth_image_callback,
#             10
#         )

#         # Publishers
#         self.publisher_cmd_vel = self.create_publisher(
#             Twist,
#             'cmd_vel',
#             10
#         )
#         self.publisher_id_distance = self.create_publisher(
#             IdDistanceInfo,
#             'id_distance_info',
#             10
#         )

#         # Initialize CvBridge for OpenCV
#         self.bridge = CvBridge()

#         # Tracking information storage
#         self.tracked_objects = {}  # id -> (bbox_center, bbox, depth)
#         self.id_last_seen = {}  # id -> last seen time
#         self.last_valid_depth = {}  # To store the last valid depth value for each person_id
#         self.closest_obstacle_distance = float('inf')  # To store the closest distance to an obstacle
#         self.similarity_scores_buffer = collections.defaultdict(lambda: collections.deque(maxlen=FRAME_BUFFER_SIZE))  # Store similarity scores over multiple frames

#         # Create trackbars for parameter adjustment
#         self.create_trackbars()

#     def create_trackbars(self):
#         """Create OpenCV windows with trackbars for adjusting parameters."""
#         cv2.namedWindow("Trackbars")

#         cv2.createTrackbar("Confidence", "Trackbars", CONF_THRESHOLD, 100, self.on_conf_trackbar)
#         cv2.createTrackbar("IoU", "Trackbars", IOU_THRESHOLD, 100, self.on_iou_trackbar)
#         cv2.createTrackbar("Similarity Threshold", "Trackbars", SIMILARITY_THRESHOLD, 100, self.on_similarity_threshold_trackbar)
#         cv2.createTrackbar("Weight Global", "Trackbars", WEIGHT_GLOBAL, 100, self.on_weight_trackbar)
#         cv2.createTrackbar("Weight Upper Body", "Trackbars", WEIGHT_UPPER_BODY, 100, self.on_weight_trackbar)
#         cv2.createTrackbar("Weight Lower Body", "Trackbars", WEIGHT_LOWER_BODY, 100, self.on_weight_trackbar)
#         cv2.createTrackbar("Weight Upper Body Hist", "Trackbars", WEIGHT_UPPER_BODY_HIST, 100, self.on_weight_trackbar)
#         cv2.createTrackbar("Weight Lower Body Hist", "Trackbars", WEIGHT_LOWER_BODY_HIST, 100, self.on_weight_trackbar)
#         cv2.createTrackbar("Frame Buffer Size", "Trackbars", FRAME_BUFFER_SIZE, 20, self.on_frame_buffer_size_trackbar)

#     def on_conf_trackbar(self, value):
#         global CONF_THRESHOLD
#         CONF_THRESHOLD = value

#     def on_iou_trackbar(self, value):
#         global IOU_THRESHOLD
#         IOU_THRESHOLD = value

#     def on_similarity_threshold_trackbar(self, value):
#         global SIMILARITY_THRESHOLD
#         SIMILARITY_THRESHOLD = value

#     def on_frame_buffer_size_trackbar(self, value):
#         global FRAME_BUFFER_SIZE
#         FRAME_BUFFER_SIZE = max(1, value)  # Ensure at least 1 frame for averaging
#         self.similarity_scores_buffer = collections.defaultdict(lambda: collections.deque(maxlen=FRAME_BUFFER_SIZE))  # Update buffer size

#     def on_weight_trackbar(self, value):
#         """Callback function for trackbar to adjust weights and ensure they sum up to 100%."""
#         global WEIGHT_GLOBAL, WEIGHT_UPPER_BODY, WEIGHT_LOWER_BODY, WEIGHT_UPPER_BODY_HIST, WEIGHT_LOWER_BODY_HIST

#         # Get the current trackbar positions
#         WEIGHT_GLOBAL = cv2.getTrackbarPos("Weight Global", "Trackbars")
#         WEIGHT_UPPER_BODY = cv2.getTrackbarPos("Weight Upper Body", "Trackbars")
#         WEIGHT_LOWER_BODY = cv2.getTrackbarPos("Weight Lower Body", "Trackbars")
#         WEIGHT_UPPER_BODY_HIST = cv2.getTrackbarPos("Weight Upper Body Hist", "Trackbars")
#         WEIGHT_LOWER_BODY_HIST = cv2.getTrackbarPos("Weight Lower Body Hist", "Trackbars")

#         # Normalize weights to sum up to 100%
#         total_weight = WEIGHT_GLOBAL + WEIGHT_UPPER_BODY + WEIGHT_LOWER_BODY + WEIGHT_UPPER_BODY_HIST + WEIGHT_LOWER_BODY_HIST
#         if total_weight > 0:
#             WEIGHT_GLOBAL = int((WEIGHT_GLOBAL / total_weight) * 100)
#             WEIGHT_UPPER_BODY = int((WEIGHT_UPPER_BODY / total_weight) * 100)
#             WEIGHT_LOWER_BODY = int((WEIGHT_LOWER_BODY / total_weight) * 100)
#             WEIGHT_UPPER_BODY_HIST = int((WEIGHT_UPPER_BODY_HIST / total_weight) * 100)
#             WEIGHT_LOWER_BODY_HIST = int((WEIGHT_LOWER_BODY_HIST / total_weight) * 100)

#         # Update trackbar positions
#         cv2.setTrackbarPos("Weight Global", "Trackbars", WEIGHT_GLOBAL)
#         cv2.setTrackbarPos("Weight Upper Body", "Trackbars", WEIGHT_UPPER_BODY)
#         cv2.setTrackbarPos("Weight Lower Body", "Trackbars", WEIGHT_LOWER_BODY)
#         cv2.setTrackbarPos("Weight Upper Body Hist", "Trackbars", WEIGHT_UPPER_BODY_HIST)
#         cv2.setTrackbarPos("Weight Lower Body Hist", "Trackbars", WEIGHT_LOWER_BODY_HIST)

#     def get_unique_color(self, id):
#         """Generate a unique color based on ID"""
#         np.random.seed(id)  # Seed with the ID for consistency
#         return tuple(np.random.randint(0, 255, 3).tolist())

#     def depth_image_callback(self, msg):
#         # Decode the compressed depth image
#         np_arr = np.frombuffer(msg.data, np.uint8)
#         depth_frame = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

#         # Ensure the image size is 320x240
#         if depth_frame.shape != (320, 320):
#             self.get_logger().warning(f"Unexpected depth image size: {depth_frame.shape}. Expected (320, 320).")

#         # Calculate the closest obstacle distance
#         self.closest_obstacle_distance = np.min(depth_frame) / 10.0  # Convert minimum depth value to cm

#         active_ids = set()
#         current_time = self.get_clock().now()

#         for person_id, (bbox_center, bbox, _) in self.tracked_objects.items():
#             x1, y1, x2, y2 = bbox
#             x, y = bbox_center

#             if 0 <= x < depth_frame.shape[1] and 0 <= y < depth_frame.shape[0]:
#                 depth_value = round(depth_frame[y, x] / 10, 2)  # Convert mm to cm with two decimal places
#                 if depth_value == 0:
#                     if person_id in self.last_valid_depth:
#                         depth_value = self.last_valid_depth[person_id]
#                     else:
#                         continue
#                 else:
#                     self.last_valid_depth[person_id] = depth_value
#             else:
#                 if person_id in self.last_valid_depth:
#                     depth_value = self.last_valid_depth[person_id]
#                 else:
#                     depth_value = None

#             self.tracked_objects[person_id] = (bbox_center, bbox, depth_value)
#             self.id_last_seen[person_id] = current_time
#             active_ids.add(person_id)

#             if 1 in self.tracked_objects:
#                 self.force_drive(id_to_follow=1)

#     def color_image_callback(self, msg):
#         # Decode the compressed color image
#         np_arr = np.frombuffer(msg.data, np.uint8)
#         frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

#         # Ensure the image size is 320x240
#         if frame.shape != (320, 320, 3):
#             self.get_logger().warning(f"Unexpected color image size: {frame.shape}. Expected (320, 320, 3).")

#         self.current_color_image = frame.copy()

#         # Use current values of conf and iou thresholds from trackbars
#         results = model.track(classes=[0], max_det=3, show=False, source=frame, stream=True, tracker=tracker_config, conf=CONF_THRESHOLD / 100, iou=IOU_THRESHOLD / 100)
#         results = list(results)

#         active_ids = set()
#         current_time = self.get_clock().now()

#         for result in results:
#             boxes = result.boxes.xyxy.cpu().numpy().astype(int) if result.boxes is not None else []
#             track_ids = result.boxes.id.cpu().numpy().astype(int) if result.boxes.id is not None else []

#             for box, track_id in zip(boxes, track_ids):
#                 x1, y1, x2, y2 = box
#                 person_img = frame[y1:y2, x1:x2]

#                 global_features = self.extract_global_features(person_img)
#                 upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram = self.extract_local_features(person_img)

#                 max_similarity = -1
#                 max_similarity_id = -1

#                 for known_id, (known_global_features, known_upper_body_features, known_lower_body_features, known_upper_body_histogram, known_lower_body_histogram) in known_persons.items():
#                     similarity = self.calculate_combined_similarity(
#                         global_features,
#                         known_global_features,
#                         upper_body_features,
#                         known_upper_body_features,
#                         lower_body_features,
#                         known_lower_body_features,
#                         upper_body_histogram,
#                         known_upper_body_histogram,
#                         lower_body_histogram,
#                         known_lower_body_histogram,
#                     )

#                     self.similarity_scores_buffer[known_id].append(similarity)
#                     avg_similarity = np.mean(list(self.similarity_scores_buffer[known_id])[-FRAME_BUFFER_SIZE:])

#                     if avg_similarity > max_similarity:
#                         max_similarity = avg_similarity
#                         max_similarity_id = known_id

#                 if max_similarity > SIMILARITY_THRESHOLD / 100:
#                     person_id = max_similarity_id
#                 else:
#                     person_id = int(track_id)
#                     if person_id not in known_persons:
#                         new_id = max(known_persons.keys(), default=0) + 1
#                         person_id = new_id
#                     known_persons[person_id] = (global_features, upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram)

#                 self.id_last_seen[person_id] = current_time
#                 active_ids.add(person_id)

#                 bbox_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
#                 bbox = (x1, y1, x2, y2)
#                 self.tracked_objects[person_id] = (bbox_center, bbox, self.tracked_objects.get(person_id, (None, None, None))[2])

#                 color = self.get_unique_color(person_id)
#                 cv2.rectangle(self.current_color_image, (x1, y1), (x2, y2), color, 2)
#                 label = f"ID: {person_id}"
#                 text_color = (255, 255, 255)

#                 center_x = (x1 + x2) // 2
#                 center_y = (y1 + y2) // 2
#                 centp = f"Center: ({center_x}, {center_y})"

#                 depth_value = self.tracked_objects[person_id][2]
#                 depth_text = f"Depth: {depth_value}cm" if depth_value is not None else "Depth: N/A"

#                 (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
#                 (center_text_width, center_text_height), _ = cv2.getTextSize(centp, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
#                 (depth_text_width, depth_text_height), _ = cv2.getTextSize(depth_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)

#                 label_y_position = y2 + text_height + 10
#                 center_text_y_position = label_y_position + center_text_height + 10
#                 depth_text_y_position = center_text_y_position + depth_text_height + 10

#                 cv2.rectangle(self.current_color_image, (x1, y2), (x1 + text_width, label_y_position), color, thickness=cv2.FILLED)
#                 cv2.putText(self.current_color_image, label, (x1, y2 + text_height), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
#                 cv2.rectangle(self.current_color_image, (x1, label_y_position + 5), (x1 + center_text_width, center_text_y_position), color, thickness=cv2.FILLED)
#                 cv2.putText(self.current_color_image, centp, (x1, label_y_position + center_text_height + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
#                 cv2.rectangle(self.current_color_image, (x1, center_text_y_position + 5), (x1 + depth_text_width, depth_text_y_position), color, thickness=cv2.FILLED)
#                 cv2.putText(self.current_color_image, depth_text, (x1, center_text_y_position + depth_text_height + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

#         cv2.imshow("Color Image with Tracking", self.current_color_image)
#         cv2.waitKey(1)

#     def publish_id_distance_info(self, id, bbox_center, width, height, depth_value):
#         msg = IdDistanceInfo()
#         msg.id = id
#         msg.bbox_center = Point(x=float(bbox_center[0]), y=float(bbox_center[1]), z=0.0)
#         msg.width = float(width)
#         msg.height = float(height)
#         msg.distance = float(depth_value)
#         self.publisher_id_distance.publish(msg)

#     def extract_global_features(self, image):
#         pil_img = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#         input_img = transform(pil_img).unsqueeze(0).to(device)
#         features = predictor(input_img).cpu().numpy()
#         return features

#     def extract_local_features(self, image):
#         height, width, _ = image.shape
#         segments = [
#             image[:height//2, :],  
#             image[height//2:, :],  
#             image[:, :width//2],   
#             image[:, width//2:]    
#         ]
        
#         local_features = []
#         local_histograms = []
#         for segment in segments:
#             pil_img = PILImage.fromarray(cv2.cvtColor(segment, cv2.COLOR_BGR2RGB))
#             input_img = transform(pil_img).unsqueeze(0).to(device)
#             features = predictor(input_img).cpu().numpy()
#             hist = self.calculate_histogram(segment)
#             local_features.append(features)
#             local_histograms.append(hist)
        
#         upper_body_features = local_features[0]  
#         lower_body_features = local_features[1]  
        
#         upper_body_histogram = local_histograms[0]  
#         lower_body_histogram = local_histograms[1]  

#         return upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram

#     def calculate_histogram(self, image):
#         hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#         hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
#         cv2.normalize(hist, hist)
#         return hist.flatten()

#     def calculate_combined_similarity(self, global_features1, global_features2, upper_body_features1, upper_body_features2, lower_body_features1, lower_body_features2, upper_body_histogram1, upper_body_histogram2, lower_body_histogram1, lower_body_histogram2):
#         global_similarity = np.dot(global_features1, global_features2.T) / (np.linalg.norm(global_features1) * np.linalg.norm(global_features2))
#         upper_body_similarity = np.dot(upper_body_features1, upper_body_features2.T) / (np.linalg.norm(upper_body_features1) * np.linalg.norm(upper_body_features2))
#         lower_body_similarity = np.dot(lower_body_features1, lower_body_features2.T) / (np.linalg.norm(lower_body_features1) * np.linalg.norm(lower_body_features2))
#         upper_body_hist_similarity = cv2.compareHist(upper_body_histogram1, upper_body_histogram2, cv2.HISTCMP_CORREL)
#         lower_body_hist_similarity = cv2.compareHist(lower_body_histogram1, lower_body_histogram2, cv2.HISTCMP_CORREL)

#         # Normalize weights
#         total_weight = WEIGHT_GLOBAL + WEIGHT_UPPER_BODY + WEIGHT_LOWER_BODY + WEIGHT_UPPER_BODY_HIST + WEIGHT_LOWER_BODY_HIST
#         normalized_global_weight = WEIGHT_GLOBAL / total_weight
#         normalized_upper_body_weight = WEIGHT_UPPER_BODY / total_weight
#         normalized_lower_body_weight = WEIGHT_LOWER_BODY / total_weight
#         normalized_upper_body_hist_weight = WEIGHT_UPPER_BODY_HIST / total_weight
#         normalized_lower_body_hist_weight = WEIGHT_LOWER_BODY_HIST / total_weight
        
#         combined_similarity = (normalized_global_weight * global_similarity +
#                                normalized_upper_body_weight * upper_body_similarity +
#                                normalized_lower_body_weight * lower_body_similarity +
#                                normalized_upper_body_hist_weight * upper_body_hist_similarity +
#                                normalized_lower_body_hist_weight * lower_body_hist_similarity)
#         return combined_similarity

#     def force_drive(self, id_to_follow=1, duration=1.0, stop_distance=0.7):
#         if id_to_follow not in self.tracked_objects:
#             return

#         bbox_center, _, depth_value = self.tracked_objects[id_to_follow]

#         if not self.tracked_objects:
#             twist_msg = Twist()
#             twist_msg.linear.x = 0.0
#             twist_msg.angular.z = 0.0
#             self.publisher_cmd_vel.publish(twist_msg)
#             time.sleep(0.1)
#             return

#         for person_id, (_, _, depth) in self.tracked_objects.items():
#             if depth is not None and depth < stop_distance * 100:
#                 twist_msg = Twist()
#                 twist_msg.linear.x = 0.0
#                 twist_msg.angular.z = 0.0
#                 self.publisher_cmd_vel.publish(twist_msg)
#                 time.sleep(0.1)
#                 return

#         camera_center_x = self.current_color_image.shape[1] // 2
#         angular_z = 0.0

#         offset_x = bbox_center[0] - camera_center_x

#         max_angular_speed = 0.35
#         angular_z = -max_angular_speed * (offset_x / camera_center_x)
#         angular_z = max(-max_angular_speed, min(max_angular_speed, angular_z))

#         safty_distance_diff = depth_value - (stop_distance * 100)

#         twist_msg = Twist()
#         twist_msg.linear.x = 0.1
#         twist_msg.angular.z = angular_z
#         self.publisher_cmd_vel.publish(twist_msg)
#         time.sleep(0.1)

# def main(args=None):
#     rclpy.init(args=args)
#     node = AllInOneNode()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()










# # # 다중프레임에 객체 인지 유지 보완 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# import collections
# if not hasattr(collections, 'Mapping'):
#     import collections.abc
#     collections.Mapping = collections.abc.Mapping

# import cv2
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import CompressedImage
# from geometry_msgs.msg import Twist, Point
# from cv_bridge import CvBridge
# import numpy as np
# import torch
# from fastreid.config import get_cfg
# from fastreid.engine import DefaultPredictor
# from fastreid.data.transforms import build_transforms
# from PIL import Image as PILImage
# from ultralytics import YOLO
# from message.msg import IdDistanceInfo
# import time

# # CUDA device configuration
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # YOLOv5 model loading
# model = YOLO("yolov8n.pt").to(device)

# # BotSORT configuration file path
# tracker_config = "/home/jinjuuk/.local/lib/python3.10/site-packages/ultralytics/cfg/trackers/botsort.yaml"

# # FastReID configuration
# cfg = get_cfg()
# cfg.merge_from_file("/home/jinjuuk/fast-reid/configs/Market1501/bagtricks_R50.yml")
# cfg.MODEL.WEIGHTS = "/home/jinjuuk/dev_ws/pt_files/market_mgn_R50-ibn.pth"
# cfg.MODEL.DEVICE = "cuda"  # GPU usage

# predictor = DefaultPredictor(cfg)
# transform = build_transforms(cfg, is_train=False)

# # Dictionary to store known persons' feature vectors and IDs
# known_persons = {}

# # Initial setting
# SIMILARITY_THRESHOLD = 90  # 0.90 in percentage
# CONF_THRESHOLD = 70  # 0.70 in percentage
# IOU_THRESHOLD = 75  # 0.75 in percentage

# # Weights for combined similarity
# WEIGHT_GLOBAL = 20  # 20%
# WEIGHT_UPPER_BODY = 15  # 15%
# WEIGHT_LOWER_BODY = 15  # 15%
# WEIGHT_UPPER_BODY_HIST = 25  # 25%
# WEIGHT_LOWER_BODY_HIST = 25  # 25%

# # Frame buffer size for averaging similarity scores
# FRAME_BUFFER_SIZE = 5  # Default value

# class AllInOneNode(Node):
#     def __init__(self):
#         super().__init__('all_in_one_node')

#         # Subscribers
#         self.subscription_color = self.create_subscription(
#             CompressedImage,
#             '/camera/color/image_raw/compressed',
#             self.color_image_callback,
#             10
#         )
#         self.subscription_depth = self.create_subscription(
#             CompressedImage,
#             '/camera/depth/image_raw/compressed',
#             self.depth_image_callback,
#             10
#         )

#         # Publishers
#         self.publisher_cmd_vel = self.create_publisher(
#             Twist,
#             'cmd_vel',
#             10
#         )
#         self.publisher_id_distance = self.create_publisher(
#             IdDistanceInfo,
#             'id_distance_info',
#             10
#         )

#         # Initialize CvBridge for OpenCV
#         self.bridge = CvBridge()

#         # Tracking information storage
#         self.tracked_objects = {}  # id -> (bbox_center, bbox, depth)
#         self.id_last_seen = {}  # id -> last seen time
#         self.last_valid_depth = {}  # To store the last valid depth value for each person_id
#         self.closest_obstacle_distance = float('inf')  # To store the closest distance to an obstacle
#         self.similarity_scores_buffer = collections.defaultdict(lambda: collections.deque(maxlen=FRAME_BUFFER_SIZE))  # Store similarity scores over multiple frames

#         # Create trackbars for parameter adjustment
#         self.create_trackbars()

#     def create_trackbars(self):
#         """Create OpenCV windows with trackbars for adjusting parameters."""
#         cv2.namedWindow("Trackbars")

#         cv2.createTrackbar("Confidence", "Trackbars", CONF_THRESHOLD, 100, self.on_conf_trackbar)
#         cv2.createTrackbar("IoU", "Trackbars", IOU_THRESHOLD, 100, self.on_iou_trackbar)
#         cv2.createTrackbar("Similarity Threshold", "Trackbars", SIMILARITY_THRESHOLD, 100, self.on_similarity_threshold_trackbar)
#         cv2.createTrackbar("Weight Global", "Trackbars", WEIGHT_GLOBAL, 100, self.on_weight_trackbar)
#         cv2.createTrackbar("Weight Upper Body", "Trackbars", WEIGHT_UPPER_BODY, 100, self.on_weight_trackbar)
#         cv2.createTrackbar("Weight Lower Body", "Trackbars", WEIGHT_LOWER_BODY, 100, self.on_weight_trackbar)
#         cv2.createTrackbar("Weight Upper Body Hist", "Trackbars", WEIGHT_UPPER_BODY_HIST, 100, self.on_weight_trackbar)
#         cv2.createTrackbar("Weight Lower Body Hist", "Trackbars", WEIGHT_LOWER_BODY_HIST, 100, self.on_weight_trackbar)
#         cv2.createTrackbar("Frame Buffer Size", "Trackbars", FRAME_BUFFER_SIZE, 20, self.on_frame_buffer_size_trackbar)

#     def on_conf_trackbar(self, value):
#         global CONF_THRESHOLD
#         CONF_THRESHOLD = value

#     def on_iou_trackbar(self, value):
#         global IOU_THRESHOLD
#         IOU_THRESHOLD = value

#     def on_similarity_threshold_trackbar(self, value):
#         global SIMILARITY_THRESHOLD
#         SIMILARITY_THRESHOLD = value

#     def on_frame_buffer_size_trackbar(self, value):
#         global FRAME_BUFFER_SIZE
#         FRAME_BUFFER_SIZE = max(1, value)  # Ensure at least 1 frame for averaging
#         self.similarity_scores_buffer = collections.defaultdict(lambda: collections.deque(maxlen=FRAME_BUFFER_SIZE))  # Update buffer size

#     def on_weight_trackbar(self, value):
#         """Callback function for trackbar to adjust weights and ensure they sum up to 100%."""
#         global WEIGHT_GLOBAL, WEIGHT_UPPER_BODY, WEIGHT_LOWER_BODY, WEIGHT_UPPER_BODY_HIST, WEIGHT_LOWER_BODY_HIST

#         # Get the current trackbar positions
#         WEIGHT_GLOBAL = cv2.getTrackbarPos("Weight Global", "Trackbars")
#         WEIGHT_UPPER_BODY = cv2.getTrackbarPos("Weight Upper Body", "Trackbars")
#         WEIGHT_LOWER_BODY = cv2.getTrackbarPos("Weight Lower Body", "Trackbars")
#         WEIGHT_UPPER_BODY_HIST = cv2.getTrackbarPos("Weight Upper Body Hist", "Trackbars")
#         WEIGHT_LOWER_BODY_HIST = cv2.getTrackbarPos("Weight Lower Body Hist", "Trackbars")

#         # Normalize weights to sum up to 100%
#         total_weight = WEIGHT_GLOBAL + WEIGHT_UPPER_BODY + WEIGHT_LOWER_BODY + WEIGHT_UPPER_BODY_HIST + WEIGHT_LOWER_BODY_HIST
#         if total_weight > 0:
#             WEIGHT_GLOBAL = int((WEIGHT_GLOBAL / total_weight) * 100)
#             WEIGHT_UPPER_BODY = int((WEIGHT_UPPER_BODY / total_weight) * 100)
#             WEIGHT_LOWER_BODY = int((WEIGHT_LOWER_BODY / total_weight) * 100)
#             WEIGHT_UPPER_BODY_HIST = int((WEIGHT_UPPER_BODY_HIST / total_weight) * 100)
#             WEIGHT_LOWER_BODY_HIST = int((WEIGHT_LOWER_BODY_HIST / total_weight) * 100)

#         # Update trackbar positions
#         cv2.setTrackbarPos("Weight Global", "Trackbars", WEIGHT_GLOBAL)
#         cv2.setTrackbarPos("Weight Upper Body", "Trackbars", WEIGHT_UPPER_BODY)
#         cv2.setTrackbarPos("Weight Lower Body", "Trackbars", WEIGHT_LOWER_BODY)
#         cv2.setTrackbarPos("Weight Upper Body Hist", "Trackbars", WEIGHT_UPPER_BODY_HIST)
#         cv2.setTrackbarPos("Weight Lower Body Hist", "Trackbars", WEIGHT_LOWER_BODY_HIST)

#     def get_unique_color(self, id):
#         """Generate a unique color based on ID"""
#         np.random.seed(id)  # Seed with the ID for consistency
#         return tuple(np.random.randint(0, 255, 3).tolist())

#     def depth_image_callback(self, msg):
#         # Decode the compressed depth image
#         np_arr = np.frombuffer(msg.data, np.uint8)
#         depth_frame = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

#         # Ensure the image size is 320x240
#         if depth_frame.shape != (320, 320):
#             self.get_logger().warning(f"Unexpected depth image size: {depth_frame.shape}. Expected (320, 320).")

#         # Calculate the closest obstacle distance
#         self.closest_obstacle_distance = np.min(depth_frame) / 10.0  # Convert minimum depth value to cm

#         active_ids = set()
#         current_time = self.get_clock().now()

#         for person_id, (bbox_center, bbox, _) in self.tracked_objects.items():
#             x1, y1, x2, y2 = bbox
#             x, y = bbox_center

#             if 0 <= x < depth_frame.shape[1] and 0 <= y < depth_frame.shape[0]:
#                 depth_value = round(depth_frame[y, x] / 10, 2)  # Convert mm to cm with two decimal places
#                 if depth_value == 0:
#                     if person_id in self.last_valid_depth:
#                         depth_value = self.last_valid_depth[person_id]
#                     else:
#                         continue
#                 else:
#                     self.last_valid_depth[person_id] = depth_value
#             else:
#                 if person_id in self.last_valid_depth:
#                     depth_value = self.last_valid_depth[person_id]
#                 else:
#                     depth_value = None

#             self.tracked_objects[person_id] = (bbox_center, bbox, depth_value)
#             self.id_last_seen[person_id] = current_time
#             active_ids.add(person_id)

#             if 1 in self.tracked_objects:
#                 self.force_drive(id_to_follow=1)

#     def color_image_callback(self, msg):
#         # Decode the compressed color image
#         np_arr = np.frombuffer(msg.data, np.uint8)
#         frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

#         # Ensure the image size is 320x320
#         if frame.shape != (320, 320, 3):
#             self.get_logger().warning(f"Unexpected color image size: {frame.shape}. Expected (320, 320, 3).")

#         self.current_color_image = frame.copy()

#         # Use current values of conf and iou thresholds from trackbars
#         results = model.track(classes=[0], max_det=3, show=False, source=frame, stream=True, tracker=tracker_config, conf=CONF_THRESHOLD / 100, iou=IOU_THRESHOLD / 100)
#         results = list(results)

#         active_ids = set()
#         current_time = self.get_clock().now()

#         for result in results:
#             boxes = result.boxes.xyxy.cpu().numpy().astype(int) if result.boxes is not None else []
#             track_ids = result.boxes.id.cpu().numpy().astype(int) if result.boxes.id is not None else []

#             for box, track_id in zip(boxes, track_ids):
#                 x1, y1, x2, y2 = box
#                 person_img = frame[y1:y2, x1:x2]

#                 if track_id == 1:
#                     # Extract global features based on person detected in ID 1 bounding box
#                     global_features = self.extract_global_features(person_img)

#                     # Extract local features based on color histogram
#                     upper_body_histogram = self.calculate_histogram(person_img[:(y2-y1)//2, :])  # Upper half
#                     lower_body_histogram = self.calculate_histogram(person_img[(y2-y1)//2:, :])  # Lower half

#                     # Store features
#                     current_features = global_features

#                     # Calculate average features over 5 frames
#                     self.similarity_scores_buffer[track_id].append(current_features)
#                     avg_features = np.mean(list(self.similarity_scores_buffer[track_id])[-FRAME_BUFFER_SIZE:], axis=0)

#                     # Check if current average features are within max-min range
#                     max_feature = np.max(list(self.similarity_scores_buffer[track_id]), axis=0)
#                     min_feature = np.min(list(self.similarity_scores_buffer[track_id]), axis=0)

#                     if np.all(min_feature <= avg_features) and np.all(avg_features <= max_feature):
#                         # Use similarity threshold for final ID decision
#                         max_similarity = -1
#                         max_similarity_id = -1

#                         for known_id, known_features in known_persons.items():
#                             # Calculate similarity using global and color histogram features
#                             similarity = self.calculate_combined_similarity(
#                                 avg_features, known_features[0],  # Global features
#                                 None, None,  # No need for upper/lower body features
#                                 None, None,  # No need for upper/lower body features
#                                 upper_body_histogram, known_features[1],  # Upper body histogram
#                                 lower_body_histogram, known_features[2]   # Lower body histogram
#                             )
#                             if similarity > max_similarity:
#                                 max_similarity = similarity
#                                 max_similarity_id = known_id

#                         if max_similarity > SIMILARITY_THRESHOLD / 100:
#                             person_id = max_similarity_id
#                         else:
#                             new_id = max(known_persons.keys(), default=0) + 1
#                             person_id = new_id
#                             known_persons[person_id] = (avg_features, upper_body_histogram, lower_body_histogram)

#                     else:
#                         new_id = max(known_persons.keys(), default=0) + 1
#                         person_id = new_id
#                         known_persons[person_id] = (avg_features, upper_body_histogram, lower_body_histogram)

#                 else:
#                     # Handle other track_ids similarly but only storing global features
#                     person_id = int(track_id)
#                     if person_id not in known_persons:
#                         new_id = max(known_persons.keys(), default=0) + 1
#                         person_id = new_id
#                     known_persons[person_id] = (self.extract_global_features(person_img), None, None)

#                 self.id_last_seen[person_id] = current_time
#                 active_ids.add(person_id)

#                 bbox_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
#                 bbox = (x1, y1, x2, y2)
#                 self.tracked_objects[person_id] = (bbox_center, bbox, self.tracked_objects.get(person_id, (None, None, None))[2])

#                 color = self.get_unique_color(person_id)
#                 cv2.rectangle(self.current_color_image, (x1, y1), (x2, y2), color, 2)
#                 label = f"ID: {person_id}"
#                 text_color = (255, 255, 255)

#                 center_x = (x1 + x2) // 2
#                 center_y = (y1 + y2) // 2
#                 centp = f"Center: ({center_x}, {center_y})"

#                 depth_value = self.tracked_objects[person_id][2]
#                 depth_text = f"Depth: {depth_value}cm" if depth_value is not None else "Depth: N/A"

#                 (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
#                 (center_text_width, center_text_height), _ = cv2.getTextSize(centp, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
#                 (depth_text_width, depth_text_height), _ = cv2.getTextSize(depth_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)

#                 label_y_position = y2 + text_height + 10
#                 center_text_y_position = label_y_position + center_text_height + 10
#                 depth_text_y_position = center_text_y_position + depth_text_height + 10

#                 cv2.rectangle(self.current_color_image, (x1, y2), (x1 + text_width, label_y_position), color, thickness=cv2.FILLED)
#                 cv2.putText(self.current_color_image, label, (x1, y2 + text_height), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
#                 cv2.rectangle(self.current_color_image, (x1, label_y_position + 5), (x1 + center_text_width, center_text_y_position), color, thickness=cv2.FILLED)
#                 cv2.putText(self.current_color_image, centp, (x1, label_y_position + center_text_height + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
#                 cv2.rectangle(self.current_color_image, (x1, center_text_y_position + 5), (x1 + depth_text_width, depth_text_y_position), color, thickness=cv2.FILLED)
#                 cv2.putText(self.current_color_image, depth_text, (x1, center_text_y_position + depth_text_height + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

#         cv2.imshow("Color Image with Tracking", self.current_color_image)
#         cv2.waitKey(1)


#     def publish_id_distance_info(self, id, bbox_center, width, height, depth_value):
#         msg = IdDistanceInfo()
#         msg.id = id
#         msg.bbox_center = Point(x=float(bbox_center[0]), y=float(bbox_center[1]), z=0.0)
#         msg.width = float(width)
#         msg.height = float(height)
#         msg.distance = float(depth_value)
#         self.publisher_id_distance.publish(msg)

#     def extract_global_features(self, image):
#         """
#         Extracts global features from an image using a pre-trained model.
        
#         Args:
#             image (numpy.ndarray): The input image.
            
#         Returns:
#             numpy.ndarray: The extracted global features.
#         """
#         pil_img = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#         input_img = transform(pil_img).unsqueeze(0).to(device)
#         features = predictor(input_img).cpu().numpy()
#         return features

#     def extract_local_features(self, image):
#         """
#         Extracts color histograms from the upper and lower halves of the image to use as local features.
        
#         Args:
#             image (numpy.ndarray): The input image.
            
#         Returns:
#             tuple: Histograms for the upper body and lower body.
#         """
#         height, _ = image.shape[:2]
        
#         # Split image into upper and lower halves
#         upper_body_image = image[:height // 2, :]  # Upper half
#         lower_body_image = image[height // 2:, :]  # Lower half
        
#         # Calculate histograms for each region
#         upper_body_histogram = self.calculate_histogram(upper_body_image)
#         lower_body_histogram = self.calculate_histogram(lower_body_image)
        
#         return upper_body_histogram, lower_body_histogram

#     def calculate_histogram(self, image):
#         """
#         Calculates the color histogram for an image in the HSV color space.

#         Args:
#             image (numpy.ndarray): The input image.

#         Returns:
#             numpy.ndarray: The normalized histogram flattened into a 1D array.
#         """
#         hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#         hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
#         cv2.normalize(hist, hist)

#         # Convert histogram to float32 for compatibility with cv2.compareHist
#         hist = hist.astype('float32')
        
#         return hist.flatten()

#     def calculate_combined_similarity(self, global_features1, global_features2, _, __, ___, ____, upper_body_histogram1, upper_body_histogram2, lower_body_histogram1, lower_body_histogram2):
#         """
#         Calculates a combined similarity score between two sets of features.

#         Args:
#             global_features1, global_features2 (numpy.ndarray): Global features for comparison.
#             upper_body_histogram1, upper_body_histogram2 (numpy.ndarray): Histograms of the upper body.
#             lower_body_histogram1, lower_body_histogram2 (numpy.ndarray): Histograms of the lower body.

#         Returns:
#             float: The combined similarity score.
#         """

#         # Check and convert histograms to float32 type if not None
#         if upper_body_histogram1 is not None and upper_body_histogram2 is not None:
#             upper_body_histogram1 = upper_body_histogram1.astype('float32')
#             upper_body_histogram2 = upper_body_histogram2.astype('float32')
#         else:
#             # Set a default similarity score when histograms are None
#             upper_body_hist_similarity = 0.15

#         if lower_body_histogram1 is not None and lower_body_histogram2 is not None:
#             lower_body_histogram1 = lower_body_histogram1.astype('float32')
#             lower_body_histogram2 = lower_body_histogram2.astype('float32')
#         else:
#             # Set a default similarity score when histograms are None
#             lower_body_hist_similarity = 0.15

#         # Debugging: Check histogram types
#         if upper_body_histogram1 is not None and upper_body_histogram2 is not None:
#             upper_body_hist_similarity = cv2.compareHist(upper_body_histogram1, upper_body_histogram2, cv2.HISTCMP_CORREL)
#         if lower_body_histogram1 is not None and lower_body_histogram2 is not None:
#             lower_body_hist_similarity = cv2.compareHist(lower_body_histogram1, lower_body_histogram2, cv2.HISTCMP_CORREL)

#         # Calculate global similarity
#         global_similarity = np.dot(global_features1, global_features2.T) / (np.linalg.norm(global_features1) * np.linalg.norm(global_features2))

#         # Normalize weights
#         total_weight = WEIGHT_GLOBAL + WEIGHT_UPPER_BODY_HIST + WEIGHT_LOWER_BODY_HIST
#         normalized_global_weight = WEIGHT_GLOBAL / total_weight
#         normalized_upper_body_hist_weight = WEIGHT_UPPER_BODY_HIST / total_weight
#         normalized_lower_body_hist_weight = WEIGHT_LOWER_BODY_HIST / total_weight

#         # Calculate combined similarity
#         combined_similarity = (normalized_global_weight * global_similarity +
#                             normalized_upper_body_hist_weight * upper_body_hist_similarity +
#                             normalized_lower_body_hist_weight * lower_body_hist_similarity)

#         return combined_similarity


#     def force_drive(self, id_to_follow=1, duration=1.0, stop_distance=0.7):
#         if id_to_follow not in self.tracked_objects:
#             return

#         bbox_center, _, depth_value = self.tracked_objects[id_to_follow]

#         if not self.tracked_objects:
#             twist_msg = Twist()
#             twist_msg.linear.x = 0.0
#             twist_msg.angular.z = 0.0
#             self.publisher_cmd_vel.publish(twist_msg)
#             time.sleep(0.1)
#             return

#         for person_id, (_, _, depth) in self.tracked_objects.items():
#             if depth is not None and depth < stop_distance * 100:
#                 twist_msg = Twist()
#                 twist_msg.linear.x = 0.0
#                 twist_msg.angular.z = 0.0
#                 self.publisher_cmd_vel.publish(twist_msg)
#                 time.sleep(0.1)
#                 return

#         camera_center_x = self.current_color_image.shape[1] // 2
#         angular_z = 0.0

#         offset_x = bbox_center[0] - camera_center_x

#         max_angular_speed = 0.35
#         angular_z = -max_angular_speed * (offset_x / camera_center_x)
#         angular_z = max(-max_angular_speed, min(max_angular_speed, angular_z))

#         safty_distance_diff = depth_value - (stop_distance * 100)

#         twist_msg = Twist()
#         twist_msg.linear.x = 0.1
#         twist_msg.angular.z = angular_z
#         self.publisher_cmd_vel.publish(twist_msg)
#         time.sleep(0.1)

# def main(args=None):
#     rclpy.init(args=args)
#     node = AllInOneNode()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()










# # 다중 + 속도 + 방향 적용!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# import collections
# if not hasattr(collections, 'Mapping'):
#     import collections.abc
#     collections.Mapping = collections.abc.Mapping

# import cv2
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import CompressedImage
# from geometry_msgs.msg import Twist, Point
# from cv_bridge import CvBridge
# import numpy as np
# import torch
# from fastreid.config import get_cfg
# from fastreid.engine import DefaultPredictor
# from fastreid.data.transforms import build_transforms
# from PIL import Image as PILImage
# from ultralytics import YOLO
# from message.msg import IdDistanceInfo
# import time

# # CUDA device configuration
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # YOLOv5 model loading
# model = YOLO("yolov5su.pt").to(device)

# # BotSORT configuration file path
# tracker_config = "/home/jinjuuk/.local/lib/python3.10/site-packages/ultralytics/cfg/trackers/botsort.yaml"

# # FastReID configuration
# cfg = get_cfg()
# cfg.merge_from_file("/home/jinjuuk/fast-reid/configs/Market1501/bagtricks_R50.yml")
# cfg.MODEL.WEIGHTS = "/home/jinjuuk/dev_ws/pt_files/market_mgn_R50-ibn.pth"
# cfg.MODEL.DEVICE = "cuda"  # GPU usage

# predictor = DefaultPredictor(cfg)
# transform = build_transforms(cfg, is_train=False)

# # Dictionary to store known persons' feature vectors and IDs
# known_persons = {}

# # Initial setting
# SIMILARITY_THRESHOLD = 97  # 0.97 in percentage
# CONF_THRESHOLD = 70  # 0.70 in percentage
# IOU_THRESHOLD = 75  # 0.75 in percentage

# # Initial setting for Weights of combined similarity
# WEIGHT_GLOBAL = 59  #디폴트 값 59와 나머지는 다 10
# WEIGHT_UPPER_BODY = 10
# WEIGHT_LOWER_BODY = 10
# WEIGHT_UPPER_BODY_HIST = 10
# WEIGHT_LOWER_BODY_HIST = 10  

# # Frame buffer size for averaging similarity scores
# FRAME_BUFFER_SIZE = 5  # Default value

# class AllInOneNode(Node):
#     def __init__(self):
#         super().__init__('all_in_one_node')

#         # Subscribers
#         self.subscription_color = self.create_subscription(
#             CompressedImage,
#             '/camera/color/image_raw/compressed',
#             self.color_image_callback,
#             10
#         )
#         self.subscription_depth = self.create_subscription(
#             CompressedImage,
#             '/camera/depth/image_raw/compressed',
#             self.depth_image_callback,
#             10
#         )

#         # Publishers
#         self.publisher_cmd_vel = self.create_publisher(
#             Twist,
#             'cmd_vel',
#             10
#         )
#         self.publisher_id_distance = self.create_publisher(
#             IdDistanceInfo,
#             'id_distance_info',
#             10
#         )

#         # Initialize CvBridge for OpenCV
#         self.bridge = CvBridge()

#         # Tracking information storage
#         self.tracked_objects = {}  # id -> (bbox_center, bbox, depth)
#         self.id_last_seen = {}  # id -> last seen time
#         self.last_valid_depth = {}  # To store the last valid depth value for each person_id
#         self.closest_obstacle_distance = float('inf')  # To store the closest distance to an obstacle
#         self.similarity_scores_buffer = collections.defaultdict(lambda: collections.deque(maxlen=FRAME_BUFFER_SIZE))  # Store similarity scores over multiple frames
#         self.last_positions = {}  # id -> (last_bbox_center, last_time)
#         self.velocities = {}  # id -> (velocity_x, velocity_y)
#         self.id_last_seen = {}  # id -> last seen time



#         # Create trackbars for parameter adjustment
#         self.create_trackbars()

#     def create_trackbars(self):
#         """Create OpenCV windows with trackbars for adjusting parameters."""
#         cv2.namedWindow("Trackbars")

#         cv2.createTrackbar("Confidence", "Trackbars", CONF_THRESHOLD, 100, self.on_conf_trackbar)
#         cv2.createTrackbar("IoU", "Trackbars", IOU_THRESHOLD, 100, self.on_iou_trackbar)
#         cv2.createTrackbar("Similarity Threshold", "Trackbars", SIMILARITY_THRESHOLD, 100, self.on_similarity_threshold_trackbar)
#         cv2.createTrackbar("Weight Global", "Trackbars", WEIGHT_GLOBAL, 100, self.on_weight_trackbar)
#         cv2.createTrackbar("Weight Upper Body", "Trackbars", WEIGHT_UPPER_BODY, 100, self.on_weight_trackbar)
#         cv2.createTrackbar("Weight Lower Body", "Trackbars", WEIGHT_LOWER_BODY, 100, self.on_weight_trackbar)
#         cv2.createTrackbar("Weight Upper Body Hist", "Trackbars", WEIGHT_UPPER_BODY_HIST, 100, self.on_weight_trackbar)
#         cv2.createTrackbar("Weight Lower Body Hist", "Trackbars", WEIGHT_LOWER_BODY_HIST, 100, self.on_weight_trackbar)
#         cv2.createTrackbar("Frame Buffer Size", "Trackbars", FRAME_BUFFER_SIZE, 20, self.on_frame_buffer_size_trackbar)

#     def on_conf_trackbar(self, value):
#         global CONF_THRESHOLD
#         CONF_THRESHOLD = value

#     def on_iou_trackbar(self, value):
#         global IOU_THRESHOLD
#         IOU_THRESHOLD = value

#     def on_similarity_threshold_trackbar(self, value):
#         global SIMILARITY_THRESHOLD
#         SIMILARITY_THRESHOLD = value

#     def on_frame_buffer_size_trackbar(self, value):
#         global FRAME_BUFFER_SIZE
#         FRAME_BUFFER_SIZE = max(1, value)  # Ensure at least 1 frame for averaging
#         self.similarity_scores_buffer = collections.defaultdict(lambda: collections.deque(maxlen=FRAME_BUFFER_SIZE))  # Update buffer size

#     def on_weight_trackbar(self, value):
#         """Callback function for trackbar to adjust weights and ensure they sum up to 100%."""
#         global WEIGHT_GLOBAL, WEIGHT_UPPER_BODY, WEIGHT_LOWER_BODY, WEIGHT_UPPER_BODY_HIST, WEIGHT_LOWER_BODY_HIST

#         # Get the current trackbar positions
#         WEIGHT_GLOBAL = cv2.getTrackbarPos("Weight Global", "Trackbars")
#         WEIGHT_UPPER_BODY = cv2.getTrackbarPos("Weight Upper Body", "Trackbars")
#         WEIGHT_LOWER_BODY = cv2.getTrackbarPos("Weight Lower Body", "Trackbars")
#         WEIGHT_UPPER_BODY_HIST = cv2.getTrackbarPos("Weight Upper Body Hist", "Trackbars")
#         WEIGHT_LOWER_BODY_HIST = cv2.getTrackbarPos("Weight Lower Body Hist", "Trackbars")

#         # Normalize weights to sum up to 100%
#         total_weight = WEIGHT_GLOBAL + WEIGHT_UPPER_BODY + WEIGHT_LOWER_BODY + WEIGHT_UPPER_BODY_HIST + WEIGHT_LOWER_BODY_HIST
#         if total_weight > 0:
#             WEIGHT_GLOBAL = int((WEIGHT_GLOBAL / total_weight) * 100)
#             WEIGHT_UPPER_BODY = int((WEIGHT_UPPER_BODY / total_weight) * 100)
#             WEIGHT_LOWER_BODY = int((WEIGHT_LOWER_BODY / total_weight) * 100)
#             WEIGHT_UPPER_BODY_HIST = int((WEIGHT_UPPER_BODY_HIST / total_weight) * 100)
#             WEIGHT_LOWER_BODY_HIST = int((WEIGHT_LOWER_BODY_HIST / total_weight) * 100)

#         # Update trackbar positions
#         cv2.setTrackbarPos("Weight Global", "Trackbars", WEIGHT_GLOBAL)
#         cv2.setTrackbarPos("Weight Upper Body", "Trackbars", WEIGHT_UPPER_BODY)
#         cv2.setTrackbarPos("Weight Lower Body", "Trackbars", WEIGHT_LOWER_BODY)
#         cv2.setTrackbarPos("Weight Upper Body Hist", "Trackbars", WEIGHT_UPPER_BODY_HIST)
#         cv2.setTrackbarPos("Weight Lower Body Hist", "Trackbars", WEIGHT_LOWER_BODY_HIST)

#     def get_unique_color(self, id):
#         """Generate a unique color based on ID"""
#         np.random.seed(id)  # Seed with the ID for consistency
#         return tuple(np.random.randint(0, 255, 3).tolist())

#     def depth_image_callback(self, msg):
#         # Decode the compressed depth image
#         np_arr = np.frombuffer(msg.data, np.uint8)
#         depth_frame = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

#         # Ensure the image size is 320x240
#         if depth_frame.shape != (320, 320):
#             self.get_logger().warning(f"Unexpected depth image size: {depth_frame.shape}. Expected (320, 320).")

#         # Calculate the closest obstacle distance
#         self.closest_obstacle_distance = np.min(depth_frame) / 10.0  # Convert minimum depth value to cm

#         active_ids = set()
#         current_time = self.get_clock().now()

#         for person_id, (bbox_center, bbox, _) in self.tracked_objects.items():
#             x1, y1, x2, y2 = bbox
#             x, y = bbox_center

#             if 0 <= x < depth_frame.shape[1] and 0 <= y < depth_frame.shape[0]:
#                 depth_value = round(depth_frame[y, x] / 10, 2)  # Convert mm to cm with two decimal places
#                 if depth_value == 0:
#                     if person_id in self.last_valid_depth:
#                         depth_value = self.last_valid_depth[person_id]
#                     else:
#                         continue
#                 else:
#                     self.last_valid_depth[person_id] = depth_value
#             else:
#                 if person_id in self.last_valid_depth:
#                     depth_value = self.last_valid_depth[person_id]
#                 else:
#                     depth_value = None

#             self.tracked_objects[person_id] = (bbox_center, bbox, depth_value)
#             self.id_last_seen[person_id] = current_time
#             active_ids.add(person_id)

#             if 1 in self.tracked_objects:
#                 self.force_drive(id_to_follow=1)

#     def color_image_callback(self, msg):
#         # Decode the compressed color image
#         np_arr = np.frombuffer(msg.data, np.uint8)
#         frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

#         # Ensure the image size is 320x240
#         if frame.shape != (320, 320, 3):
#             self.get_logger().warning(f"Unexpected color image size: {frame.shape}. Expected (320, 320, 3).")

#         self.current_color_image = frame.copy()

#         # Use current values of conf and iou thresholds from trackbars
#         results = model.track(classes=[0], max_det=3, show=False, source=frame, stream=True, tracker=tracker_config, conf=CONF_THRESHOLD / 100, iou=IOU_THRESHOLD / 100)
#         results = list(results)

#         active_ids = set()
#         current_time = self.get_clock().now().to_msg()

#         for result in results:
#             boxes = result.boxes.xyxy.cpu().numpy().astype(int) if result.boxes is not None else []
#             track_ids = result.boxes.id.cpu().numpy().astype(int) if result.boxes.id is not None else []

#             for box, track_id in zip(boxes, track_ids):
#                 x1, y1, x2, y2 = box
#                 person_img = frame[y1:y2, x1:x2]

#                 global_features = self.extract_global_features(person_img)
#                 upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram = self.extract_local_features(person_img)

#                 max_similarity = -1
#                 max_similarity_id = -1

#                 for known_id, (known_global_features, known_upper_body_features, known_lower_body_features, known_upper_body_histogram, known_lower_body_histogram) in known_persons.items():
#                     similarity = self.calculate_combined_similarity(
#                         global_features,
#                         known_global_features,
#                         upper_body_features,
#                         known_upper_body_features,
#                         lower_body_features,
#                         known_lower_body_features,
#                         upper_body_histogram,
#                         known_upper_body_histogram,
#                         lower_body_histogram,
#                         known_lower_body_histogram,
#                     )

#                     self.similarity_scores_buffer[known_id].append(similarity)
#                     avg_similarity = np.mean(list(self.similarity_scores_buffer[known_id])[-FRAME_BUFFER_SIZE:])

#                     if avg_similarity > max_similarity:
#                         max_similarity = avg_similarity
#                         max_similarity_id = known_id

#                 if max_similarity > SIMILARITY_THRESHOLD / 100:
#                     person_id = max_similarity_id
#                 else:
#                     person_id = int(track_id)
#                     if person_id not in known_persons:
#                         new_id = max(known_persons.keys(), default=0) + 1
#                         person_id = new_id
#                     known_persons[person_id] = (global_features, upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram)

#                 self.id_last_seen[person_id] = current_time
#                 active_ids.add(person_id)

#                 bbox_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
#                 bbox = (x1, y1, x2, y2)
                
#                 # Calculate velocity if last position exists
#                 if person_id in self.last_positions:
#                     last_bbox_center, last_time = self.last_positions[person_id]
#                     time_diff = current_time.sec - last_time.sec + (current_time.nanosec - last_time.nanosec) / 1e9
#                     if time_diff > 0:
#                         velocity_x = (bbox_center[0] - last_bbox_center[0]) / time_diff
#                         velocity_y = (bbox_center[1] - last_bbox_center[1]) / time_diff
#                         self.velocities[person_id] = (velocity_x, velocity_y)
#                 else:
#                     self.velocities[person_id] = (0, 0)  # No velocity initially

#                 # Update last position
#                 self.last_positions[person_id] = (bbox_center, current_time)

#                 self.tracked_objects[person_id] = (bbox_center, bbox, self.tracked_objects.get(person_id, (None, None, None))[2])

#                 color = self.get_unique_color(person_id)
#                 cv2.rectangle(self.current_color_image, (x1, y1), (x2, y2), color, 2)
#                 label = f"ID: {person_id}"
#                 text_color = (255, 255, 255)

#                 center_x = (x1 + x2) // 2
#                 center_y = (y1 + y2) // 2
#                 centp = f"Center: ({center_x}, {center_y})"

#                 depth_value = self.tracked_objects[person_id][2]
#                 depth_text = f"Depth: {depth_value}cm" if depth_value is not None else "Depth: N/A"

#                 velocity_text = f"Velocity: ({self.velocities[person_id][0]:.2f}, {self.velocities[person_id][1]:.2f}) px/s"

#                 (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
#                 (center_text_width, center_text_height), _ = cv2.getTextSize(centp, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
#                 (depth_text_width, depth_text_height), _ = cv2.getTextSize(depth_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
#                 (velocity_text_width, velocity_text_height), _ = cv2.getTextSize(velocity_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)

#                 label_y_position = y2 + text_height + 10
#                 center_text_y_position = label_y_position + center_text_height + 10
#                 depth_text_y_position = center_text_y_position + depth_text_height + 10
#                 velocity_text_y_position = depth_text_y_position + velocity_text_height + 10

#                 cv2.rectangle(self.current_color_image, (x1, y2), (x1 + text_width, label_y_position), color, thickness=cv2.FILLED)
#                 cv2.putText(self.current_color_image, label, (x1, y2 + text_height), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
#                 cv2.rectangle(self.current_color_image, (x1, label_y_position + 5), (x1 + center_text_width, center_text_y_position), color, thickness=cv2.FILLED)
#                 cv2.putText(self.current_color_image, centp, (x1, label_y_position + center_text_height + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
#                 cv2.rectangle(self.current_color_image, (x1, center_text_y_position + 5), (x1 + depth_text_width, depth_text_y_position), color, thickness=cv2.FILLED)
#                 cv2.putText(self.current_color_image, depth_text, (x1, center_text_y_position + depth_text_height + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
#                 cv2.rectangle(self.current_color_image, (x1, depth_text_y_position + 5), (x1 + velocity_text_width, velocity_text_y_position), color, thickness=cv2.FILLED)
#                 cv2.putText(self.current_color_image, velocity_text, (x1, depth_text_y_position + velocity_text_height + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

#         cv2.imshow("Color Image with Tracking", self.current_color_image)
#         cv2.waitKey(1)

#     def publish_id_distance_info(self, id, bbox_center, width, height, depth_value):
#         msg = IdDistanceInfo()
#         msg.id = id
#         msg.bbox_center = Point(x=float(bbox_center[0]), y=float(bbox_center[1]), z=0.0)
#         msg.width = float(width)
#         msg.height = float(height)
#         msg.distance = float(depth_value)
#         self.publisher_id_distance.publish(msg)

#     def extract_global_features(self, image):
#         pil_img = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#         input_img = transform(pil_img).unsqueeze(0).to(device)
#         features = predictor(input_img).cpu().numpy()
#         return features

#     def extract_local_features(self, image):
#         height, width, _ = image.shape
#         segments = [
#             image[:height//2, :],  
#             image[height//2:, :],  
#             image[:, :width//2],   
#             image[:, width//2:]    
#         ]
        
#         local_features = []
#         local_histograms = []
#         for segment in segments:
#             pil_img = PILImage.fromarray(cv2.cvtColor(segment, cv2.COLOR_BGR2RGB))
#             input_img = transform(pil_img).unsqueeze(0).to(device)
#             features = predictor(input_img).cpu().numpy()
#             hist = self.calculate_histogram(segment)
#             local_features.append(features)
#             local_histograms.append(hist)
        
#         upper_body_features = local_features[0]  
#         lower_body_features = local_features[1]  
        
#         upper_body_histogram = local_histograms[0]  
#         lower_body_histogram = local_histograms[1]  

#         return upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram

#     def calculate_histogram(self, image):
#         hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#         hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
#         cv2.normalize(hist, hist)
#         return hist.flatten()

#     def calculate_combined_similarity(self, global_features1, global_features2, upper_body_features1, upper_body_features2, lower_body_features1, lower_body_features2, upper_body_histogram1, upper_body_histogram2, lower_body_histogram1, lower_body_histogram2):
#         global_similarity = np.dot(global_features1, global_features2.T) / (np.linalg.norm(global_features1) * np.linalg.norm(global_features2))
#         upper_body_similarity = np.dot(upper_body_features1, upper_body_features2.T) / (np.linalg.norm(upper_body_features1) * np.linalg.norm(upper_body_features2))
#         lower_body_similarity = np.dot(lower_body_features1, lower_body_features2.T) / (np.linalg.norm(lower_body_features1) * np.linalg.norm(lower_body_features2))
#         upper_body_hist_similarity = cv2.compareHist(upper_body_histogram1, upper_body_histogram2, cv2.HISTCMP_CORREL)
#         lower_body_hist_similarity = cv2.compareHist(lower_body_histogram1, lower_body_histogram2, cv2.HISTCMP_CORREL)

#         # Normalize weights
#         total_weight = WEIGHT_GLOBAL + WEIGHT_UPPER_BODY + WEIGHT_LOWER_BODY + WEIGHT_UPPER_BODY_HIST + WEIGHT_LOWER_BODY_HIST
#         normalized_global_weight = WEIGHT_GLOBAL / total_weight
#         normalized_upper_body_weight = WEIGHT_UPPER_BODY / total_weight
#         normalized_lower_body_weight = WEIGHT_LOWER_BODY / total_weight
#         normalized_upper_body_hist_weight = WEIGHT_UPPER_BODY_HIST / total_weight
#         normalized_lower_body_hist_weight = WEIGHT_LOWER_BODY_HIST / total_weight
        
#         combined_similarity = (normalized_global_weight * global_similarity +
#                                normalized_upper_body_weight * upper_body_similarity +
#                                normalized_lower_body_weight * lower_body_similarity +
#                                normalized_upper_body_hist_weight * upper_body_hist_similarity +
#                                normalized_lower_body_hist_weight * lower_body_hist_similarity)
#         return combined_similarity

#     def force_drive(self, id_to_follow=1, duration=1.0, stop_distance=0.7):
#         if id_to_follow not in self.tracked_objects:
#             return

#         bbox_center, _, depth_value = self.tracked_objects[id_to_follow]

#         if not self.tracked_objects:
#             twist_msg = Twist()
#             twist_msg.linear.x = 0.0
#             twist_msg.angular.z = 0.0
#             self.publisher_cmd_vel.publish(twist_msg)
#             time.sleep(0.1)
#             return

#         for person_id, (_, _, depth) in self.tracked_objects.items():
#             if depth is not None and depth < stop_distance * 100:
#                 twist_msg = Twist()
#                 twist_msg.linear.x = 0.0
#                 twist_msg.angular.z = 0.0
#                 self.publisher_cmd_vel.publish(twist_msg)
#                 time.sleep(0.1)
#                 return

#         camera_center_x = self.current_color_image.shape[1] // 2
#         angular_z = 0.0

#         offset_x = bbox_center[0] - camera_center_x

#         max_angular_speed = 0.35
#         angular_z = -max_angular_speed * (offset_x / camera_center_x)
#         angular_z = max(-max_angular_speed, min(max_angular_speed, angular_z))

#         safty_distance_diff = depth_value - (stop_distance * 100)

#         twist_msg = Twist()
#         twist_msg.linear.x = 0.1
#         twist_msg.angular.z = angular_z
#         self.publisher_cmd_vel.publish(twist_msg)
#         time.sleep(0.1)

# def main(args=None):
#     rclpy.init(args=args)
#     node = AllInOneNode()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()












# # 다중 + 속도 + 방향 적용 + 디스플레이에 적용 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# import collections
# if not hasattr(collections, 'Mapping'):
#     import collections.abc
#     collections.Mapping = collections.abc.Mapping

# import cv2
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import CompressedImage
# from geometry_msgs.msg import Twist, Point
# from cv_bridge import CvBridge
# import numpy as np
# import torch
# from fastreid.config import get_cfg
# from fastreid.engine import DefaultPredictor
# from fastreid.data.transforms import build_transforms
# from PIL import Image as PILImage
# from ultralytics import YOLO
# from message.msg import IdDistanceInfo
# import time

# # CUDA device configuration
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # YOLOv5 model loading
# model = YOLO("yolov8s.pt").to(device)

# # BotSORT configuration file path
# tracker_config = "/home/jinjuuk/.local/lib/python3.10/site-packages/ultralytics/cfg/trackers/botsort.yaml"

# # FastReID configuration
# cfg = get_cfg()
# cfg.merge_from_file("/home/jinjuuk/fast-reid/configs/Market1501/bagtricks_R50.yml")
# cfg.MODEL.WEIGHTS = "/home/jinjuuk/dev_ws/pt_files/market_mgn_R50-ibn.pth"
# cfg.MODEL.DEVICE = "cuda"  # GPU usage

# predictor = DefaultPredictor(cfg)
# transform = build_transforms(cfg, is_train=False)

# # Dictionary to store known persons' feature vectors and IDs
# known_persons = {}

# # Initial setting
# SIMILARITY_THRESHOLD = 97  # 0.97 in percentage
# CONF_THRESHOLD = 70  # 0.70 in percentage
# IOU_THRESHOLD = 75  # 0.75 in percentage

# # Initial setting for Weights of combined similarity
# WEIGHT_GLOBAL = 59  
# WEIGHT_UPPER_BODY = 10
# WEIGHT_LOWER_BODY = 10
# WEIGHT_UPPER_BODY_HIST = 10
# WEIGHT_LOWER_BODY_HIST = 10  

# # Frame buffer size for averaging similarity scores
# FRAME_BUFFER_SIZE = 5  # Default value

# class AllInOneNode(Node):
#     def __init__(self):
#         super().__init__('all_in_one_node')

#         # Subscribers
#         self.subscription_color = self.create_subscription(
#             CompressedImage,
#             '/camera/color/image_raw/compressed',
#             self.color_image_callback,
#             10
#         )
#         self.subscription_depth = self.create_subscription(
#             CompressedImage,
#             '/camera/depth/image_raw/compressed',
#             self.depth_image_callback,
#             10
#         )

#         # Publishers
#         self.publisher_cmd_vel = self.create_publisher(
#             Twist,
#             'cmd_vel',
#             10
#         )
#         self.publisher_id_distance = self.create_publisher(
#             IdDistanceInfo,
#             'id_distance_info',
#             10
#         )

#         # Initialize CvBridge for OpenCV
#         self.bridge = CvBridge()

#         # Tracking information storage
#         self.tracked_objects = {}  # id -> (bbox_center, bbox, depth)
#         self.id_last_seen = {}  # id -> last seen time
#         self.last_valid_depth = {}  # To store the last valid depth value for each person_id
#         self.closest_obstacle_distance = float('inf')  # To store the closest distance to an obstacle
#         self.similarity_scores_buffer = collections.defaultdict(lambda: collections.deque(maxlen=FRAME_BUFFER_SIZE))  # Store similarity scores over multiple frames
#         self.trajectory = collections.deque(maxlen=30)  # To store ID 1's trajectory points


#         # Create trackbars for parameter adjustment
#         self.create_trackbars()

#     def create_trackbars(self):
#         """Create OpenCV windows with trackbars for adjusting parameters."""
#         cv2.namedWindow("Trackbars")

#         cv2.createTrackbar("Confidence", "Trackbars", CONF_THRESHOLD, 100, self.on_conf_trackbar)
#         cv2.createTrackbar("IoU", "Trackbars", IOU_THRESHOLD, 100, self.on_iou_trackbar)
#         cv2.createTrackbar("Similarity Threshold", "Trackbars", SIMILARITY_THRESHOLD, 100, self.on_similarity_threshold_trackbar)
#         cv2.createTrackbar("Weight Global", "Trackbars", WEIGHT_GLOBAL, 100, self.on_weight_trackbar)
#         cv2.createTrackbar("Weight Upper Body", "Trackbars", WEIGHT_UPPER_BODY, 100, self.on_weight_trackbar)
#         cv2.createTrackbar("Weight Lower Body", "Trackbars", WEIGHT_LOWER_BODY, 100, self.on_weight_trackbar)
#         cv2.createTrackbar("Weight Upper Body Hist", "Trackbars", WEIGHT_UPPER_BODY_HIST, 100, self.on_weight_trackbar)
#         cv2.createTrackbar("Weight Lower Body Hist", "Trackbars", WEIGHT_LOWER_BODY_HIST, 100, self.on_weight_trackbar)
#         cv2.createTrackbar("Frame Buffer Size", "Trackbars", FRAME_BUFFER_SIZE, 20, self.on_frame_buffer_size_trackbar)

#     def on_conf_trackbar(self, value):
#         global CONF_THRESHOLD
#         CONF_THRESHOLD = value

#     def on_iou_trackbar(self, value):
#         global IOU_THRESHOLD
#         IOU_THRESHOLD = value

#     def on_similarity_threshold_trackbar(self, value):
#         global SIMILARITY_THRESHOLD
#         SIMILARITY_THRESHOLD = value

#     def on_frame_buffer_size_trackbar(self, value):
#         global FRAME_BUFFER_SIZE
#         FRAME_BUFFER_SIZE = max(1, value)  # Ensure at least 1 frame for averaging
#         self.similarity_scores_buffer = collections.defaultdict(lambda: collections.deque(maxlen=FRAME_BUFFER_SIZE))  # Update buffer size

#     def on_weight_trackbar(self, value):
#         """Callback function for trackbar to adjust weights and ensure they sum up to 100%."""
#         global WEIGHT_GLOBAL, WEIGHT_UPPER_BODY, WEIGHT_LOWER_BODY, WEIGHT_UPPER_BODY_HIST, WEIGHT_LOWER_BODY_HIST

#         # Get the current trackbar positions
#         WEIGHT_GLOBAL = cv2.getTrackbarPos("Weight Global", "Trackbars")
#         WEIGHT_UPPER_BODY = cv2.getTrackbarPos("Weight Upper Body", "Trackbars")
#         WEIGHT_LOWER_BODY = cv2.getTrackbarPos("Weight Lower Body", "Trackbars")
#         WEIGHT_UPPER_BODY_HIST = cv2.getTrackbarPos("Weight Upper Body Hist", "Trackbars")
#         WEIGHT_LOWER_BODY_HIST = cv2.getTrackbarPos("Weight Lower Body Hist", "Trackbars")

#         # Normalize weights to sum up to 100%
#         total_weight = WEIGHT_GLOBAL + WEIGHT_UPPER_BODY + WEIGHT_LOWER_BODY + WEIGHT_UPPER_BODY_HIST + WEIGHT_LOWER_BODY_HIST
#         if total_weight > 0:
#             WEIGHT_GLOBAL = int((WEIGHT_GLOBAL / total_weight) * 100)
#             WEIGHT_UPPER_BODY = int((WEIGHT_UPPER_BODY / total_weight) * 100)
#             WEIGHT_LOWER_BODY = int((WEIGHT_LOWER_BODY / total_weight) * 100)
#             WEIGHT_UPPER_BODY_HIST = int((WEIGHT_UPPER_BODY_HIST / total_weight) * 100)
#             WEIGHT_LOWER_BODY_HIST = int((WEIGHT_LOWER_BODY_HIST / total_weight) * 100)

#         # Update trackbar positions
#         cv2.setTrackbarPos("Weight Global", "Trackbars", WEIGHT_GLOBAL)
#         cv2.setTrackbarPos("Weight Upper Body", "Trackbars", WEIGHT_UPPER_BODY)
#         cv2.setTrackbarPos("Weight Lower Body", "Trackbars", WEIGHT_LOWER_BODY)
#         cv2.setTrackbarPos("Weight Upper Body Hist", "Trackbars", WEIGHT_UPPER_BODY_HIST)
#         cv2.setTrackbarPos("Weight Lower Body Hist", "Trackbars", WEIGHT_LOWER_BODY_HIST)

#     def get_unique_color(self, id):
#         """Generate a unique color based on ID"""
#         np.random.seed(id)  # Seed with the ID for consistency
#         return tuple(np.random.randint(0, 255, 3).tolist())

#     def depth_image_callback(self, msg):
#         # Decode the compressed depth image
#         np_arr = np.frombuffer(msg.data, np.uint8)
#         depth_frame = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

#         # Ensure the image size is 320x240
#         if depth_frame.shape != (320, 320):
#             self.get_logger().warning(f"Unexpected depth image size: {depth_frame.shape}. Expected (320, 320).")

#         # Calculate the closest obstacle distance
#         self.closest_obstacle_distance = np.min(depth_frame) / 10.0  # Convert minimum depth value to cm

#         active_ids = set()
#         current_time = self.get_clock().now()

#         for person_id, (bbox_center, bbox, _) in self.tracked_objects.items():
#             x1, y1, x2, y2 = bbox
#             x, y = bbox_center

#             if 0 <= x < depth_frame.shape[1] and 0 <= y < depth_frame.shape[0]:
#                 depth_value = round(depth_frame[y, x] / 10, 2)  # Convert mm to cm with two decimal places
#                 if depth_value == 0:
#                     if person_id in self.last_valid_depth:
#                         depth_value = self.last_valid_depth[person_id]
#                     else:
#                         continue
#                 else:
#                     self.last_valid_depth[person_id] = depth_value
#             else:
#                 if person_id in self.last_valid_depth:
#                     depth_value = self.last_valid_depth[person_id]
#                 else:
#                     depth_value = None

#             self.tracked_objects[person_id] = (bbox_center, bbox, depth_value)
#             self.id_last_seen[person_id] = current_time
#             active_ids.add(person_id)

#             if 1 in self.tracked_objects:
#                 self.force_drive(id_to_follow=1)

#     def color_image_callback(self, msg):
#         # Decode the compressed color image
#         np_arr = np.frombuffer(msg.data, np.uint8)
#         frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

#         # Ensure the image size is 320x240
#         if frame.shape != (320, 320, 3):
#             self.get_logger().warning(f"Unexpected color image size: {frame.shape}. Expected (320, 320, 3).")

#         self.current_color_image = frame.copy()

#         # Use current values of conf and iou thresholds from trackbars
#         results = model.track(classes=[0], max_det=3, show=False, source=frame, stream=True, tracker=tracker_config, conf=CONF_THRESHOLD / 100, iou=IOU_THRESHOLD / 100)
#         results = list(results)

#         active_ids = set()
#         current_time = self.get_clock().now().to_msg()

#         for result in results:
#             boxes = result.boxes.xyxy.cpu().numpy().astype(int) if result.boxes is not None else []
#             track_ids = result.boxes.id.cpu().numpy().astype(int) if result.boxes.id is not None else []

#             for box, track_id in zip(boxes, track_ids):
#                 x1, y1, x2, y2 = box
#                 person_img = frame[y1:y2, x1:x2]

#                 global_features = self.extract_global_features(person_img)
#                 upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram = self.extract_local_features(person_img)

#                 max_similarity = -1
#                 max_similarity_id = -1

#                 for known_id, (known_global_features, known_upper_body_features, known_lower_body_features, known_upper_body_histogram, known_lower_body_histogram) in known_persons.items():
#                     similarity = self.calculate_combined_similarity(
#                         global_features,
#                         known_global_features,
#                         upper_body_features,
#                         known_upper_body_features,
#                         lower_body_features,
#                         known_lower_body_features,
#                         upper_body_histogram,
#                         known_upper_body_histogram,
#                         lower_body_histogram,
#                         known_lower_body_histogram,
#                     )

#                     self.similarity_scores_buffer[known_id].append(similarity)
#                     avg_similarity = np.mean(list(self.similarity_scores_buffer[known_id])[-FRAME_BUFFER_SIZE:])

#                     if avg_similarity > max_similarity:
#                         max_similarity = avg_similarity
#                         max_similarity_id = known_id

#                 if max_similarity > SIMILARITY_THRESHOLD / 100:
#                     person_id = max_similarity_id
#                 else:
#                     person_id = int(track_id)
#                     if person_id not in known_persons:
#                         new_id = max(known_persons.keys(), default=0) + 1
#                         person_id = new_id
#                     known_persons[person_id] = (global_features, upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram)

#                 self.id_last_seen[person_id] = current_time
#                 active_ids.add(person_id)

#                 bbox_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
#                 bbox = (x1, y1, x2, y2)
                
#                 self.tracked_objects[person_id] = (bbox_center, bbox, self.tracked_objects.get(person_id, (None, None, None))[2])

#                 color = self.get_unique_color(person_id)
#                 cv2.rectangle(self.current_color_image, (x1, y1), (x2, y2), color, 2)
#                 label = f"ID: {person_id}"
#                 text_color = (255, 255, 255)

#                 center_x = (x1 + x2) // 2
#                 center_y = (y1 + y2) // 2
#                 centp = f"Center: ({center_x}, {center_y})"

#                 depth_value = self.tracked_objects[person_id][2]
#                 depth_text = f"Depth: {depth_value}cm" if depth_value is not None else "Depth: N/A"

#                 (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
#                 (center_text_width, center_text_height), _ = cv2.getTextSize(centp, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
#                 (depth_text_width, depth_text_height), _ = cv2.getTextSize(depth_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)

#                 label_y_position = y2 + text_height + 10
#                 center_text_y_position = label_y_position + center_text_height + 10
#                 depth_text_y_position = center_text_y_position + depth_text_height + 10

#                 cv2.rectangle(self.current_color_image, (x1, y2), (x1 + text_width, label_y_position), color, thickness=cv2.FILLED)
#                 cv2.putText(self.current_color_image, label, (x1, y2 + text_height), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
#                 cv2.rectangle(self.current_color_image, (x1, label_y_position + 5), (x1 + center_text_width, center_text_y_position), color, thickness=cv2.FILLED)
#                 cv2.putText(self.current_color_image, centp, (x1, label_y_position + center_text_height + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
#                 cv2.rectangle(self.current_color_image, (x1, center_text_y_position + 5), (x1 + depth_text_width, depth_text_y_position), color, thickness=cv2.FILLED)
#                 cv2.putText(self.current_color_image, depth_text, (x1, center_text_y_position + depth_text_height + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

#                 # Set the maximum length of the trajectory (number of points)
#                 max_trajectory_length = 20  # Adjust this value to control how long the trajectory is

#                 # Inside the loop where you handle each detected person
#                 if person_id == 1:
#                     self.trajectory.append(bbox_center)

#                     # If the trajectory exceeds the maximum length, remove the oldest point
#                     if len(self.trajectory) > max_trajectory_length:
#                         self.trajectory.popleft()  # Use popleft() to remove the first element

#                     # Draw lines connecting the points in the trajectory
#                     for i in range(1, len(self.trajectory)):
#                         cv2.line(self.current_color_image, self.trajectory[i - 1], self.trajectory[i], color, 2)



#         cv2.imshow("Color Image with Tracking", self.current_color_image)
#         cv2.waitKey(1)

#     def publish_id_distance_info(self, id, bbox_center, width, height, depth_value):
#         msg = IdDistanceInfo()
#         msg.id = id
#         msg.bbox_center = Point(x=float(bbox_center[0]), y=float(bbox_center[1]), z=0.0)
#         msg.width = float(width)
#         msg.height = float(height)
#         msg.distance = float(depth_value)
#         self.publisher_id_distance.publish(msg)

#     def extract_global_features(self, image):
#         pil_img = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#         input_img = transform(pil_img).unsqueeze(0).to(device)
#         features = predictor(input_img).cpu().numpy()
#         return features

#     def extract_local_features(self, image):
#         height, width, _ = image.shape
#         segments = [
#             image[:height//2, :],  
#             image[height//2:, :],  
#             image[:, :width//2],   
#             image[:, width//2:]    
#         ]
        
#         local_features = []
#         local_histograms = []
#         for segment in segments:
#             pil_img = PILImage.fromarray(cv2.cvtColor(segment, cv2.COLOR_BGR2RGB))
#             input_img = transform(pil_img).unsqueeze(0).to(device)
#             features = predictor(input_img).cpu().numpy()
#             hist = self.calculate_histogram(segment)
#             local_features.append(features)
#             local_histograms.append(hist)
        
#         upper_body_features = local_features[0]  
#         lower_body_features = local_features[1]  
        
#         upper_body_histogram = local_histograms[0]  
#         lower_body_histogram = local_histograms[1]  

#         return upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram

#     def calculate_histogram(self, image):
#         hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#         hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
#         cv2.normalize(hist, hist)
#         return hist.flatten()

#     def calculate_combined_similarity(self, global_features1, global_features2, upper_body_features1, upper_body_features2, lower_body_features1, lower_body_features2, upper_body_histogram1, upper_body_histogram2, lower_body_histogram1, lower_body_histogram2):
#         global_similarity = np.dot(global_features1, global_features2.T) / (np.linalg.norm(global_features1) * np.linalg.norm(global_features2))
#         upper_body_similarity = np.dot(upper_body_features1, upper_body_features2.T) / (np.linalg.norm(upper_body_features1) * np.linalg.norm(upper_body_features2))
#         lower_body_similarity = np.dot(lower_body_features1, lower_body_features2.T) / (np.linalg.norm(lower_body_features1) * np.linalg.norm(lower_body_features2))
#         upper_body_hist_similarity = cv2.compareHist(upper_body_histogram1, upper_body_histogram2, cv2.HISTCMP_CORREL)
#         lower_body_hist_similarity = cv2.compareHist(lower_body_histogram1, lower_body_histogram2, cv2.HISTCMP_CORREL)

#         # Normalize weights
#         total_weight = WEIGHT_GLOBAL + WEIGHT_UPPER_BODY + WEIGHT_LOWER_BODY + WEIGHT_UPPER_BODY_HIST + WEIGHT_LOWER_BODY_HIST
#         normalized_global_weight = WEIGHT_GLOBAL / total_weight
#         normalized_upper_body_weight = WEIGHT_UPPER_BODY / total_weight
#         normalized_lower_body_weight = WEIGHT_LOWER_BODY / total_weight
#         normalized_upper_body_hist_weight = WEIGHT_UPPER_BODY_HIST / total_weight
#         normalized_lower_body_hist_weight = WEIGHT_LOWER_BODY_HIST / total_weight
        
#         combined_similarity = (normalized_global_weight * global_similarity +
#                                normalized_upper_body_weight * upper_body_similarity +
#                                normalized_lower_body_weight * lower_body_similarity +
#                                normalized_upper_body_hist_weight * upper_body_hist_similarity +
#                                normalized_lower_body_hist_weight * lower_body_hist_similarity)
#         return combined_similarity

#     def force_drive(self, id_to_follow=1, duration=1.0, stop_distance=0.7):
#         if id_to_follow not in self.tracked_objects:
#             return

#         bbox_center, _, depth_value = self.tracked_objects[id_to_follow]

#         if not self.tracked_objects:
#             twist_msg = Twist()
#             twist_msg.linear.x = 0.0
#             twist_msg.angular.z = 0.0
#             self.publisher_cmd_vel.publish(twist_msg)
#             time.sleep(0.1)
#             return

#         for person_id, (_, _, depth) in self.tracked_objects.items():
#             if depth is not None and depth < stop_distance * 100:
#                 twist_msg = Twist()
#                 twist_msg.linear.x = 0.0
#                 twist_msg.angular.z = 0.0
#                 self.publisher_cmd_vel.publish(twist_msg)
#                 time.sleep(0.1)
#                 return

#         camera_center_x = self.current_color_image.shape[1] // 2
#         angular_z = 0.0

#         offset_x = bbox_center[0] - camera_center_x

#         max_angular_speed = 0.35
#         angular_z = -max_angular_speed * (offset_x / camera_center_x)
#         angular_z = max(-max_angular_speed, min(max_angular_speed, angular_z))

#         safty_distance_diff = depth_value - (stop_distance * 100)

#         twist_msg = Twist()
#         twist_msg.linear.x = 0.1
#         twist_msg.angular.z = angular_z
#         self.publisher_cmd_vel.publish(twist_msg)
#         time.sleep(0.1)

# def main(args=None):
#     rclpy.init(args=args)
#     node = AllInOneNode()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()











# # mean + range 잘안됨!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# import collections
# if not hasattr(collections, 'Mapping'):
#     import collections.abc
#     collections.Mapping = collections.abc.Mapping

# import cv2
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import CompressedImage
# from geometry_msgs.msg import Twist, Point
# from cv_bridge import CvBridge
# import numpy as np
# import torch
# from fastreid.config import get_cfg
# from fastreid.engine import DefaultPredictor
# from fastreid.data.transforms import build_transforms
# from PIL import Image as PILImage
# from ultralytics import YOLO
# from message.msg import IdDistanceInfo
# import time

# # CUDA device configuration
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # YOLOv5 model loading
# model = YOLO("yolov5nu.pt").to(device)

# # BotSORT configuration file path
# tracker_config = "/home/jinjuuk/.local/lib/python3.10/site-packages/ultralytics/cfg/trackers/botsort.yaml"

# # FastReID configuration
# cfg = get_cfg()
# cfg.merge_from_file("/home/jinjuuk/fast-reid/configs/Market1501/bagtricks_R50.yml")
# cfg.MODEL.WEIGHTS = "/home/jinjuuk/dev_ws/pt_files/market_mgn_R50-ibn.pth"
# cfg.MODEL.DEVICE = "cuda"  # GPU usage

# predictor = DefaultPredictor(cfg)
# transform = build_transforms(cfg, is_train=False)

# # Dictionary to store known persons' feature vectors and IDs
# known_persons = {}

# # Initial setting
# SIMILARITY_THRESHOLD = 0.9  # Adjusted for average similarity comparison

# class AllInOneNode(Node):
#     def __init__(self):
#         super().__init__('all_in_one_node')

#         # Subscribers
#         self.subscription_color = self.create_subscription(
#             CompressedImage,
#             '/camera/color/image_raw/compressed',
#             self.color_image_callback,
#             10
#         )
#         self.subscription_depth = self.create_subscription(
#             CompressedImage,
#             '/camera/depth/image_raw/compressed',
#             self.depth_image_callback,
#             10
#         )

#         # Publishers
#         self.publisher_cmd_vel = self.create_publisher(
#             Twist,
#             'cmd_vel',
#             10
#         )
#         self.publisher_id_distance = self.create_publisher(
#             IdDistanceInfo,
#             'id_distance_info',
#             10
#         )

#         # Initialize CvBridge for OpenCV
#         self.bridge = CvBridge()

#         # Tracking information storage
#         self.tracked_objects = {}  # id -> (bbox_center, bbox, depth)
#         self.id_last_seen = {}  # id -> last seen time
#         self.last_valid_depth = {}  # To store the last valid depth value for each person_id
#         self.closest_obstacle_distance = float('inf')  # To store the closest distance to an obstacle
#         self.feature_history = collections.defaultdict(lambda: collections.deque(maxlen=5))  # Store feature vectors over multiple frames
#         self.similarity_scores_buffer = collections.defaultdict(lambda: collections.deque(maxlen=5))  # Store similarity scores over multiple frames

#     def get_unique_color(self, id):
#         """Generate a unique color based on ID"""
#         np.random.seed(id)  # Seed with the ID for consistency
#         return tuple(np.random.randint(0, 255, 3).tolist())

#     def depth_image_callback(self, msg):
#         # Decode the compressed depth image
#         np_arr = np.frombuffer(msg.data, np.uint8)
#         depth_frame = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

#         # Ensure the image size is 320x240
#         if depth_frame.shape != (320, 320):
#             self.get_logger().warning(f"Unexpected depth image size: {depth_frame.shape}. Expected (240, 320).")

#         # Calculate the closest obstacle distance
#         self.closest_obstacle_distance = np.min(depth_frame) / 10.0  # Convert minimum depth value to cm

#         active_ids = set()
#         current_time = self.get_clock().now()

#         for person_id, (bbox_center, bbox, _) in self.tracked_objects.items():
#             x1, y1, x2, y2 = bbox
#             x, y = bbox_center

#             if 0 <= x < depth_frame.shape[1] and 0 <= y < depth_frame.shape[0]:
#                 depth_value = round(depth_frame[y, x] / 10, 2)  # Convert mm to cm with two decimal places
#                 if depth_value == 0:
#                     if person_id in self.last_valid_depth:
#                         depth_value = self.last_valid_depth[person_id]
#                     else:
#                         continue
#                 else:
#                     self.last_valid_depth[person_id] = depth_value
#             else:
#                 if person_id in self.last_valid_depth:
#                     depth_value = self.last_valid_depth[person_id]
#                 else:
#                     depth_value = None

#             self.tracked_objects[person_id] = (bbox_center, bbox, depth_value)
#             self.id_last_seen[person_id] = current_time
#             active_ids.add(person_id)

#             if 1 in self.tracked_objects:
#                 self.force_drive(id_to_follow=1)

        

#     def color_image_callback(self, msg):
#         # Decode the compressed color image
#         np_arr = np.frombuffer(msg.data, np.uint8)
#         frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

#         # Ensure the image size is 320x240
#         if frame.shape != (320, 320, 3):
#             self.get_logger().warning(f"Unexpected color image size: {frame.shape}. Expected (240, 320, 3).")

#         self.current_color_image = frame.copy()

#         results = model.track(classes=[0], max_det=4, show=False, source=frame, stream=True, tracker=tracker_config)
#         results = list(results)

#         active_ids = set()
#         current_time = self.get_clock().now()

#         for result in results:
#             boxes = result.boxes.xyxy.cpu().numpy().astype(int) if result.boxes is not None else []
#             track_ids = result.boxes.id.cpu().numpy().astype(int) if result.boxes.id is not None else []

#             for box, track_id in zip(boxes, track_ids):
#                 x1, y1, x2, y2 = box
#                 person_img = frame[y1:y2, x1:x2]

#                 # Extract features
#                 global_features = self.extract_global_features(person_img)
#                 upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram = self.extract_local_features(person_img)

#                 # Calculate combined similarity
#                 max_similarity = -1
#                 max_similarity_id = -1

#                 for known_id, feature_set in known_persons.items():
#                     known_global_features, known_upper_body_features, known_lower_body_features, known_upper_body_histogram, known_lower_body_histogram = feature_set

#                     similarity = self.calculate_combined_similarity(
#                         global_features,
#                         known_global_features,
#                         upper_body_features,
#                         known_upper_body_features,
#                         lower_body_features,
#                         known_lower_body_features,
#                         upper_body_histogram,
#                         known_upper_body_histogram,
#                         lower_body_histogram,
#                         known_lower_body_histogram,
#                     )

#                     # Update similarity scores buffer
#                     self.similarity_scores_buffer[known_id].append(similarity)
#                     avg_similarity = np.mean(self.similarity_scores_buffer[known_id])
#                     max_similarity_score = max(self.similarity_scores_buffer[known_id])
#                     min_similarity_score = min(self.similarity_scores_buffer[known_id])

#                     # Decision based on similarity threshold and range
#                     if (min_similarity_score < similarity < max_similarity_score) or (avg_similarity > SIMILARITY_THRESHOLD):
#                         if avg_similarity > max_similarity:
#                             max_similarity = avg_similarity
#                             max_similarity_id = known_id

#                 # Determine ID to assign
#                 if max_similarity_id != -1:
#                     person_id = max_similarity_id
#                 else:
#                     # Assign new ID
#                     new_id = max(known_persons.keys(), default=0) + 1
#                     person_id = new_id
#                     known_persons[person_id] = (global_features, upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram)

#                 # Update tracking info
#                 self.id_last_seen[person_id] = current_time
#                 active_ids.add(person_id)

#                 # Save feature vector history
#                 self.feature_history[person_id].append(global_features)

#                 bbox_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
#                 bbox = (x1, y1, x2, y2)
#                 self.tracked_objects[person_id] = (bbox_center, bbox, self.tracked_objects.get(person_id, (None, None, None))[2])

#                 # Draw bounding box and information on the frame
#                 color = self.get_unique_color(person_id)
#                 cv2.rectangle(self.current_color_image, (x1, y1), (x2, y2), color, 2)
#                 label = f"ID: {person_id}"
#                 text_color = (255, 255, 255)

#                 center_x = (x1 + x2) // 2
#                 center_y = (y1 + y2) // 2
#                 centp = f"Center: ({center_x}, {center_y})"

#                 depth_value = self.tracked_objects[person_id][2]
#                 depth_text = f"Depth: {depth_value}cm" if depth_value is not None else "Depth: N/A"

#                 (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
#                 (center_text_width, center_text_height), _ = cv2.getTextSize(centp, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
#                 (depth_text_width, depth_text_height), _ = cv2.getTextSize(depth_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)

#                 label_y_position = y2 + text_height + 10
#                 center_text_y_position = label_y_position + center_text_height + 10
#                 depth_text_y_position = center_text_y_position + depth_text_height + 10

#                 cv2.rectangle(self.current_color_image, (x1, y2), (x1 + text_width, label_y_position), color, thickness=cv2.FILLED)
#                 cv2.putText(self.current_color_image, label, (x1, y2 + text_height), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
#                 cv2.rectangle(self.current_color_image, (x1, label_y_position + 5), (x1 + center_text_width, center_text_y_position), color, thickness=cv2.FILLED)
#                 cv2.putText(self.current_color_image, centp, (x1, label_y_position + center_text_height + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
#                 cv2.rectangle(self.current_color_image, (x1, center_text_y_position + 5), (x1 + depth_text_width, depth_text_y_position), color, thickness=cv2.FILLED)
#                 cv2.putText(self.current_color_image, depth_text, (x1, center_text_y_position + depth_text_height + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

#         cv2.imshow("Color Image with Tracking", self.current_color_image)
#         cv2.waitKey(1)

#     def publish_id_distance_info(self, id, bbox_center, width, height, depth_value):
#         msg = IdDistanceInfo()
#         msg.id = id
#         msg.bbox_center = Point(x=float(bbox_center[0]), y=float(bbox_center[1]), z=0.0)
#         msg.width = float(width)
#         msg.height = float(height)
#         msg.distance = float(depth_value)
#         self.publisher_id_distance.publish(msg)

#     def extract_global_features(self, image):
#         pil_img = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#         input_img = transform(pil_img).unsqueeze(0).to(device)
#         features = predictor(input_img).cpu().numpy()
#         return features

#     def extract_local_features(self, image):
#         height, width, _ = image.shape
#         segments = [
#             image[:height//2, :],  
#             image[height//2:, :],  
#             image[:, :width//2],   
#             image[:, width//2:]    
#         ]
        
#         local_features = []
#         local_histograms = []
#         for segment in segments:
#             pil_img = PILImage.fromarray(cv2.cvtColor(segment, cv2.COLOR_BGR2RGB))
#             input_img = transform(pil_img).unsqueeze(0).to(device)
#             features = predictor(input_img).cpu().numpy()
#             hist = self.calculate_histogram(segment)
#             local_features.append(features)
#             local_histograms.append(hist)
        
#         upper_body_features = local_features[0]  
#         lower_body_features = local_features[1]  
        
#         upper_body_histogram = local_histograms[0]  
#         lower_body_histogram = local_histograms[1]  

#         return upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram

#     def calculate_histogram(self, image):
#         hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#         hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
#         cv2.normalize(hist, hist)
#         return hist.flatten()

#     def calculate_combined_similarity(self, global_features1, global_features2, upper_body_features1, upper_body_features2, lower_body_features1, lower_body_features2, upper_body_histogram1, upper_body_histogram2, lower_body_histogram1, lower_body_histogram2):
#         global_similarity = np.dot(global_features1, global_features2.T) / (np.linalg.norm(global_features1) * np.linalg.norm(global_features2))
#         upper_body_similarity = np.dot(upper_body_features1, upper_body_features2.T) / (np.linalg.norm(upper_body_features1) * np.linalg.norm(upper_body_features2))
#         lower_body_similarity = np.dot(lower_body_features1, lower_body_features2.T) / (np.linalg.norm(lower_body_features1) * np.linalg.norm(lower_body_features2))
#         upper_body_hist_similarity = cv2.compareHist(upper_body_histogram1, upper_body_histogram2, cv2.HISTCMP_CORREL)
#         lower_body_hist_similarity = cv2.compareHist(lower_body_histogram1, lower_body_histogram2, cv2.HISTCMP_CORREL)
        
#         combined_similarity = (0.3 * global_similarity +  
#                                 0.25 * upper_body_similarity +
#                                 0.25 * lower_body_similarity +
#                                 0.2 * lower_body_hist_similarity)
#         return combined_similarity

#     def force_drive(self, id_to_follow=1, duration=1.0, stop_distance=0.7):
#         if id_to_follow not in self.tracked_objects:
#             return

#         bbox_center, _, depth_value = self.tracked_objects[id_to_follow]

#         if not self.tracked_objects:
#             twist_msg = Twist()
#             twist_msg.linear.x = 0.0
#             twist_msg.angular.z = 0.0
#             self.publisher_cmd_vel.publish(twist_msg)
#             time.sleep(0.1)
#             return

#         for person_id, (_, _, depth) in self.tracked_objects.items():
#             if depth is not None and depth < stop_distance * 100:
#                 twist_msg = Twist()
#                 twist_msg.linear.x = 0.0
#                 twist_msg.angular.z = 0.0
#                 self.publisher_cmd_vel.publish(twist_msg)
#                 time.sleep(0.1)
#                 return

#         camera_center_x = self.current_color_image.shape[1] // 2
#         angular_z = 0.0

#         offset_x = bbox_center[0] - camera_center_x

#         max_angular_speed = 0.35
#         angular_z = -max_angular_speed * (offset_x / camera_center_x)
#         angular_z = max(-max_angular_speed, min(max_angular_speed, angular_z))

#         safty_distance_diff = depth_value - (stop_distance * 100)

#         twist_msg = Twist()
#         twist_msg.linear.x = 0.1
#         twist_msg.angular.z = angular_z
#         self.publisher_cmd_vel.publish(twist_msg)
#         time.sleep(0.1)

# def main(args=None):
#     rclpy.init(args=args)
#     node = AllInOneNode()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()









# import collections
# if not hasattr(collections, 'Mapping'):
#     import collections.abc
#     collections.Mapping = collections.abc.Mapping

# import cv2
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import CompressedImage
# from geometry_msgs.msg import Twist, Point
# from cv_bridge import CvBridge
# import numpy as np
# import torch
# from fastreid.config import get_cfg
# from fastreid.engine import DefaultPredictor
# from fastreid.data.transforms import build_transforms
# from PIL import Image as PILImage
# from ultralytics import YOLO
# from message.msg import IdDistanceInfo
# import time

# # CUDA device configuration
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # YOLOv5 model loading
# model = YOLO("yolov5nu.pt").to(device)

# # BotSORT configuration file path
# tracker_config = "/home/jinjuuk/.local/lib/python3.10/site-packages/ultralytics/cfg/trackers/botsort.yaml"

# # FastReID configuration
# cfg = get_cfg()
# cfg.merge_from_file("/home/jinjuuk/fast-reid/configs/Market1501/bagtricks_R50.yml")
# cfg.MODEL.WEIGHTS = "/home/jinjuuk/dev_ws/pt_files/market_mgn_R50-ibn.pth"
# cfg.MODEL.DEVICE = "cuda"  # GPU usage

# predictor = DefaultPredictor(cfg)
# transform = build_transforms(cfg, is_train=False)

# # Dictionary to store known persons' feature vectors and IDs
# known_persons = {}

# # Initial setting
# SIMILARITY_THRESHOLD = 0.80  # Adjusted for average similarity comparison

# class AllInOneNode(Node):
#     def __init__(self):
#         super().__init__('all_in_one_node')

#         # Subscribers
#         self.subscription_color = self.create_subscription(
#             CompressedImage,
#             '/camera/color/image_raw/compressed',
#             self.color_image_callback,
#             10
#         )
#         self.subscription_depth = self.create_subscription(
#             CompressedImage,
#             '/camera/depth/image_raw/compressed',
#             self.depth_image_callback,
#             10
#         )

#         # Publishers
#         self.publisher_cmd_vel = self.create_publisher(
#             Twist,
#             'cmd_vel',
#             10
#         )
#         self.publisher_id_distance = self.create_publisher(
#             IdDistanceInfo,
#             'id_distance_info',
#             10
#         )

#         # Initialize CvBridge for OpenCV
#         self.bridge = CvBridge()

#         # Tracking information storage
#         self.tracked_objects = {}  # id -> (bbox_center, bbox, depth)
#         self.id_last_seen = {}  # id -> last seen time
#         self.last_valid_depth = {}  # To store the last valid depth value for each person_id
#         self.closest_obstacle_distance = float('inf')  # To store the closest distance to an obstacle
#         self.feature_history = collections.defaultdict(lambda: collections.deque(maxlen=5))  # Store feature vectors over multiple frames
#         self.similarity_scores_buffer = collections.defaultdict(lambda: collections.deque(maxlen=5))  # Store similarity scores over multiple frames

#     def get_unique_color(self, id):
#         """Generate a unique color based on ID"""
#         np.random.seed(id)  # Seed with the ID for consistency
#         return tuple(np.random.randint(0, 255, 3).tolist())

#     def depth_image_callback(self, msg):
#         # Decode the compressed depth image
#         np_arr = np.frombuffer(msg.data, np.uint8)
#         depth_frame = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

#         # Ensure the image size is 320x240
#         # if depth_frame.shape != (320, 320):
#         #     self.get_logger().warning(f"Unexpected depth image size: {depth_frame.shape}. Expected (320, 320).")

#         # Calculate the closest obstacle distance
#         self.closest_obstacle_distance = np.min(depth_frame) / 10.0  # Convert minimum depth value to cm

#         active_ids = set()
#         current_time = self.get_clock().now()

#         for person_id, (bbox_center, bbox, _) in self.tracked_objects.items():
#             x1, y1, x2, y2 = bbox
#             x, y = bbox_center

#             if 0 <= x < depth_frame.shape[1] and 0 <= y < depth_frame.shape[0]:
#                 depth_value = round(depth_frame[y, x] / 10, 2)  # Convert mm to cm with two decimal places
#                 if depth_value == 0:
#                     if person_id in self.last_valid_depth:
#                         depth_value = self.last_valid_depth[person_id]
#                     else:
#                         continue
#                 else:
#                     self.last_valid_depth[person_id] = depth_value
#             else:
#                 if person_id in self.last_valid_depth:
#                     depth_value = self.last_valid_depth[person_id]
#                 else:
#                     depth_value = None

#             self.tracked_objects[person_id] = (bbox_center, bbox, depth_value)
#             self.id_last_seen[person_id] = current_time
#             active_ids.add(person_id)

#             if 1 in self.tracked_objects:
#                 self.force_drive(id_to_follow=1)

#         # cv2.imshow("Depth Image with Tracking", depth_frame)
#         # cv2.waitKey(1)

#     def color_image_callback(self, msg):
#         # Decode the compressed color image
#         np_arr = np.frombuffer(msg.data, np.uint8)
#         frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

#         # Ensure the image size is 320x240
#         # if frame.shape != (320, 320, 3):
#         #     self.get_logger().warning(f"Unexpected color image size: {frame.shape}. Expected (320, 320, 3).")

#         self.current_color_image = frame.copy()

#          # Confidence threshold (default: 0.25), IoU threshold for NMS (default: 0.45)
#         results = model.track(classes=[0], max_det=4, show=False, source=frame, stream=True, tracker=tracker_config, conf=0.80, iou=0.80)
#         results = list(results)

#         active_ids = set()
#         current_time = self.get_clock().now()

#         # First, find the ID 1 person and determine the filtering line
#         filter_line_y = None

#         for result in results:
#             boxes = result.boxes.xyxy.cpu().numpy().astype(int) if result.boxes is not None else []
#             track_ids = result.boxes.id.cpu().numpy().astype(int) if result.boxes.id is not None else []

#             # Determine filter line based on ID 1
#             for box, track_id in zip(boxes, track_ids):
#                 if track_id == 1:
#                     # Calculate the center y-coordinate of ID 1
#                     x1, y1, x2, y2 = box
#                     filter_line_y = (y1 + y2) // 2
#                     break

#             # If we have identified the filter line, break the loop
#             if filter_line_y is not None:
#                 break

#         # Process each detection and filter based on the y-coordinate
#         for result in results:
#             boxes = result.boxes.xyxy.cpu().numpy().astype(int) if result.boxes is not None else []
#             track_ids = result.boxes.id.cpu().numpy().astype(int) if result.boxes.id is not None else []

#             for box, track_id in zip(boxes, track_ids):
#                 # Only process if the person is in front of ID 1 or if ID 1 is not detected
#                 x1, y1, x2, y2 = box
#                 bbox_center_y = (y1 + y2) // 2

#                 if filter_line_y is None or bbox_center_y > filter_line_y:
#                     # Perform feature extraction and similarity comparison as before
#                     person_img = frame[y1:y2, x1:x2]

#                     # Extract features
#                     global_features = self.extract_global_features(person_img)
#                     upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram = self.extract_local_features(person_img)

#                     # Calculate combined similarity
#                     max_similarity = -1
#                     max_similarity_id = -1

#                     for known_id, feature_set in known_persons.items():
#                         known_global_features, known_upper_body_features, known_lower_body_features, known_upper_body_histogram, known_lower_body_histogram = feature_set

#                         combined_similarity = self.calculate_combined_similarity(
#                             global_features,
#                             known_global_features,
#                             upper_body_features,
#                             known_upper_body_features,
#                             lower_body_features,
#                             known_lower_body_features,
#                             upper_body_histogram,
#                             known_upper_body_histogram,
#                             lower_body_histogram,
#                             known_lower_body_histogram,
#                         )

#                         # Update similarity scores buffer
#                         self.similarity_scores_buffer[known_id].append(combined_similarity)
#                         avg_similarity = np.mean(self.similarity_scores_buffer[known_id])
#                         max_similarity_score = max(self.similarity_scores_buffer[known_id])
#                         min_similarity_score = min(self.similarity_scores_buffer[known_id])

#                         # Decision based on similarity threshold and range
#                         if (min_similarity_score <= combined_similarity <= max_similarity_score) and (avg_similarity >= SIMILARITY_THRESHOLD):
#                             if avg_similarity > max_similarity:
#                                 max_similarity = avg_similarity
#                                 max_similarity_id = known_id

#                     # Determine ID to assign
#                     if max_similarity_id != -1:
#                         person_id = max_similarity_id
#                     else:
#                         # Assign new ID
#                         new_id = max(known_persons.keys(), default=0) + 1
#                         person_id = new_id
#                         known_persons[person_id] = (global_features, upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram)

#                     # Update tracking info
#                     self.id_last_seen[person_id] = current_time
#                     active_ids.add(person_id)

#                     # Save feature vector history
#                     self.feature_history[person_id].append(global_features)

#                     bbox_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
#                     bbox = (x1, y1, x2, y2)
#                     self.tracked_objects[person_id] = (bbox_center, bbox, self.tracked_objects.get(person_id, (None, None, None))[2])

#                     # Draw bounding box and information on the frame
#                     color = self.get_unique_color(person_id)
#                     cv2.rectangle(self.current_color_image, (x1, y1), (x2, y2), color, 2)
#                     label = f"ID: {person_id}"
#                     text_color = (255, 255, 255)

#                     center_x = (x1 + x2) // 2
#                     center_y = (y1 + y2) // 2
#                     centp = f"Center: ({center_x}, {center_y})"

#                     depth_value = self.tracked_objects[person_id][2]
#                     depth_text = f"Depth: {depth_value}cm" if depth_value is not None else "Depth: N/A"

#                     (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
#                     (center_text_width, center_text_height), _ = cv2.getTextSize(centp, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
#                     (depth_text_width, depth_text_height), _ = cv2.getTextSize(depth_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)

#                     label_y_position = y2 + text_height + 10
#                     center_text_y_position = label_y_position + center_text_height + 10
#                     depth_text_y_position = center_text_y_position + depth_text_height + 10

#                     cv2.rectangle(self.current_color_image, (x1, y2), (x1 + text_width, label_y_position), color, thickness=cv2.FILLED)
#                     cv2.putText(self.current_color_image, label, (x1, y2 + text_height), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
#                     cv2.rectangle(self.current_color_image, (x1, label_y_position + 5), (x1 + center_text_width, center_text_y_position), color, thickness=cv2.FILLED)
#                     cv2.putText(self.current_color_image, centp, (x1, label_y_position + center_text_height + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
#                     cv2.rectangle(self.current_color_image, (x1, center_text_y_position + 5), (x1 + depth_text_width, depth_text_y_position), color, thickness=cv2.FILLED)
#                     cv2.putText(self.current_color_image, depth_text, (x1, center_text_y_position + depth_text_height + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

#         cv2.imshow("Color Image with Tracking", self.current_color_image)
#         cv2.waitKey(1)

#     def publish_id_distance_info(self, id, bbox_center, width, height, depth_value):
#         msg = IdDistanceInfo()
#         msg.id = id
#         msg.bbox_center = Point(x=float(bbox_center[0]), y=float(bbox_center[1]), z=0.0)
#         msg.width = float(width)
#         msg.height = float(height)
#         msg.distance = float(depth_value)
#         self.publisher_id_distance.publish(msg)

#     def extract_global_features(self, image):
#         pil_img = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#         input_img = transform(pil_img).unsqueeze(0).to(device)
#         features = predictor(input_img).cpu().numpy()
#         return features

#     def extract_local_features(self, image):
#         height, width, _ = image.shape
#         segments = [
#             image[:height//2, :],  
#             image[height//2:, :],  
#             image[:, :width//2],   
#             image[:, width//2:]    
#         ]
        
#         local_features = []
#         local_histograms = []
#         for segment in segments:
#             pil_img = PILImage.fromarray(cv2.cvtColor(segment, cv2.COLOR_BGR2RGB))
#             input_img = transform(pil_img).unsqueeze(0).to(device)
#             features = predictor(input_img).cpu().numpy()
#             hist = self.calculate_histogram(segment)
#             local_features.append(features)
#             local_histograms.append(hist)
        
#         upper_body_features = local_features[0]  
#         lower_body_features = local_features[1]  
        
#         upper_body_histogram = local_histograms[0]  
#         lower_body_histogram = local_histograms[1]  

#         return upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram

#     def calculate_histogram(self, image):
#         hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#         hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
#         cv2.normalize(hist, hist)
#         return hist.flatten()

#     def calculate_combined_similarity(self, global_features1, global_features2, upper_body_features1, upper_body_features2, lower_body_features1, lower_body_features2, upper_body_histogram1, upper_body_histogram2, lower_body_histogram1, lower_body_histogram2):
#         global_similarity = np.dot(global_features1, global_features2.T) / (np.linalg.norm(global_features1) * np.linalg.norm(global_features2))
#         upper_body_similarity = np.dot(upper_body_features1, upper_body_features2.T) / (np.linalg.norm(upper_body_features1) * np.linalg.norm(upper_body_features2))
#         lower_body_similarity = np.dot(lower_body_features1, lower_body_features2.T) / (np.linalg.norm(lower_body_features1) * np.linalg.norm(lower_body_features2))
#         upper_body_hist_similarity = cv2.compareHist(upper_body_histogram1, upper_body_histogram2, cv2.HISTCMP_CORREL)
#         lower_body_hist_similarity = cv2.compareHist(lower_body_histogram1, lower_body_histogram2, cv2.HISTCMP_CORREL)
        
#         combined_similarity = (0.3 * global_similarity +  
#                                 0.2 * upper_body_similarity +
#                                 0.25 * lower_body_similarity +
#                                 0.25 * lower_body_hist_similarity)
#         return combined_similarity

#     def force_drive(self, id_to_follow=1, duration=1.0, stop_distance=0.7):
#         if id_to_follow not in self.tracked_objects:
#             return

#         bbox_center, _, depth_value = self.tracked_objects[id_to_follow]

#         if not self.tracked_objects:
#             twist_msg = Twist()
#             twist_msg.linear.x = 0.0
#             twist_msg.angular.z = 0.0
#             self.publisher_cmd_vel.publish(twist_msg)
#             time.sleep(0.1)
#             return

#         for person_id, (_, _, depth) in self.tracked_objects.items():
#             if depth is not None and depth < stop_distance * 100:
#                 twist_msg = Twist()
#                 twist_msg.linear.x = 0.0
#                 twist_msg.angular.z = 0.0
#                 self.publisher_cmd_vel.publish(twist_msg)
#                 time.sleep(0.1)
#                 return

#         camera_center_x = self.current_color_image.shape[1] // 2
#         angular_z = 0.0

#         offset_x = bbox_center[0] - camera_center_x

#         max_angular_speed = 0.35
#         angular_z = -max_angular_speed * (offset_x / camera_center_x)
#         angular_z = max(-max_angular_speed, min(max_angular_speed, angular_z))

#         safty_distance_diff = depth_value - (stop_distance * 100)

#         twist_msg = Twist()
#         twist_msg.linear.x = 0.1
#         twist_msg.angular.z = angular_z
#         self.publisher_cmd_vel.publish(twist_msg)
#         time.sleep(0.1)

# def main(args=None):
#     rclpy.init(args=args)
#     node = AllInOneNode()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()












# import collections
# if not hasattr(collections, 'Mapping'):
#     import collections.abc
#     collections.Mapping = collections.abc.Mapping

# import cv2
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import CompressedImage
# from geometry_msgs.msg import Twist, Point
# from cv_bridge import CvBridge
# import numpy as np
# import torch
# from fastreid.config import get_cfg
# from fastreid.engine import DefaultPredictor
# from fastreid.data.transforms import build_transforms
# from PIL import Image as PILImage
# from ultralytics import YOLO
# from message.msg import IdDistanceInfo
# import time

# # CUDA device configuration
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # YOLOv5 model loading
# model = YOLO("yolov5su.pt").to(device)

# # BotSORT configuration file path
# tracker_config = "/home/jinjuuk/.local/lib/python3.10/site-packages/ultralytics/cfg/trackers/botsort.yaml"

# # FastReID configuration
# cfg = get_cfg()
# cfg.merge_from_file("/home/jinjuuk/fast-reid/configs/Market1501/bagtricks_R50.yml")
# cfg.MODEL.WEIGHTS = "/home/jinjuuk/dev_ws/pt_files/market_mgn_R50-ibn.pth"
# cfg.MODEL.DEVICE = "cuda"  # GPU usage

# predictor = DefaultPredictor(cfg)
# transform = build_transforms(cfg, is_train=False)

# # Dictionary to store known persons' feature vectors and IDs
# known_persons = {}

# # Initial setting
# SIMILARITY_THRESHOLD = 0.80  # Adjusted for average similarity comparison

# class AllInOneNode(Node):
#     def __init__(self):
#         super().__init__('all_in_one_node')

#         # Subscribers
#         self.subscription_color = self.create_subscription(
#             CompressedImage,
#             '/camera/color/image_raw/compressed',
#             self.color_image_callback,
#             10
#         )
#         self.subscription_depth = self.create_subscription(
#             CompressedImage,
#             '/camera/depth/image_raw/compressed',
#             self.depth_image_callback,
#             10
#         )

#         # Publishers
#         self.publisher_cmd_vel = self.create_publisher(
#             Twist,
#             'cmd_vel',
#             10
#         )
#         self.publisher_id_distance = self.create_publisher(
#             IdDistanceInfo,
#             'id_distance_info',
#             10
#         )

#         # Initialize CvBridge for OpenCV
#         self.bridge = CvBridge()

#         # Tracking information storage
#         self.tracked_objects = {}  # id -> (bbox_center, bbox, depth)
#         self.id_last_seen = {}  # id -> last seen time
#         self.last_valid_depth = {}  # To store the last valid depth value for each person_id
#         self.closest_obstacle_distance = float('inf')  # To store the closest distance to an obstacle
#         self.feature_history = collections.defaultdict(lambda: collections.deque(maxlen=5))  # Store feature vectors over multiple frames
#         self.similarity_scores_buffer = collections.defaultdict(lambda: collections.deque(maxlen=5))  # Store similarity scores over multiple frames
#         self.depth_frame = None # Add depth frame storage

#     def get_unique_color(self, id):
#         """Generate a unique color based on ID"""
#         np.random.seed(id)  # Seed with the ID for consistency
#         return tuple(np.random.randint(0, 255, 3).tolist())

#     def depth_image_callback(self, msg):
#         # Decode the compressed depth image
#         np_arr = np.frombuffer(msg.data, np.uint8)
#         depth_frame = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

#         # Store the depth frame for later use in color_image_callback
#         self.depth_frame = depth_frame

#         # Ensure the image size is 320x240
#         # if depth_frame.shape != (320, 320):
#         #     self.get_logger().warning(f"Unexpected depth image size: {depth_frame.shape}. Expected (320, 320).")

#         # Calculate the closest obstacle distance
#         self.closest_obstacle_distance = np.min(depth_frame) / 10.0  # Convert minimum depth value to cm

#         active_ids = set()
#         current_time = self.get_clock().now()

#         for person_id, (bbox_center, bbox, _) in self.tracked_objects.items():
#             x1, y1, x2, y2 = bbox
#             x, y = bbox_center

#             if 0 <= x < depth_frame.shape[1] and 0 <= y < depth_frame.shape[0]:
#                 depth_value = round(depth_frame[y, x] / 10, 2)  # Convert mm to cm with two decimal places
#                 if depth_value == 0:
#                     if person_id in self.last_valid_depth:
#                         depth_value = self.last_valid_depth[person_id]
#                     else:
#                         continue
#                 else:
#                     self.last_valid_depth[person_id] = depth_value
#             else:
#                 if person_id in self.last_valid_depth:
#                     depth_value = self.last_valid_depth[person_id]
#                 else:
#                     depth_value = None

#             self.tracked_objects[person_id] = (bbox_center, bbox, depth_value)
#             self.id_last_seen[person_id] = current_time
#             active_ids.add(person_id)

#             if 1 in self.tracked_objects:
#                 self.force_drive(id_to_follow=1)

#         # cv2.imshow("Depth Image with Tracking", depth_frame)
#         # cv2.waitKey(1)

#     def color_image_callback(self, msg):
#         # Decode the compressed color image
#         np_arr = np.frombuffer(msg.data, np.uint8)
#         frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

#         self.current_color_image = frame.copy()

#         # Confidence threshold and IoU threshold for NMS
#         results = model.track(classes=[0], max_det=4, show=False, source=frame, stream=True, tracker=tracker_config, conf=0.50, iou=0.80)
#         results = list(results)

#         active_ids = set()
#         current_time = self.get_clock().now()

#         # Initialize filter line and depth threshold
#         filter_line_y = None
#         filter_depth_threshold = None

#         # Check if the depth frame is available
#         if self.depth_frame is None:
#             self.get_logger().warning("No depth frame available. Skipping depth-based filtering.")
#             return

#         depth_frame = self.depth_frame  # Use the latest depth frame

#         # Step 1: Find the ID 1 person and determine the filtering line based on depth
#         for result in results:
#             boxes = result.boxes.xyxy.cpu().numpy().astype(int) if result.boxes is not None else []
#             track_ids = result.boxes.id.cpu().numpy().astype(int) if result.boxes.id is not None else []

#             for box, track_id in zip(boxes, track_ids):
#                 if track_id == 1:
#                     x1, y1, x2, y2 = box
#                     filter_line_y = (y1 + y2) // 2

#                     # Calculate depth if available
#                     if (x1 >= 0 and y1 >= 0 and x2 <= frame.shape[1] and y2 <= frame.shape[0]):
#                         depth_values = depth_frame[y1:y2, x1:x2]
#                         if depth_values.size > 0:
#                             filter_depth_threshold = np.mean(depth_values)
#                     break

#             if filter_line_y is not None:
#                 break

#         # Step 2: Process each detection and filter based on the y-coordinate and depth
#         for result in results:
#             boxes = result.boxes.xyxy.cpu().numpy().astype(int) if result.boxes is not None else []
#             track_ids = result.boxes.id.cpu().numpy().astype(int) if result.boxes.id is not None else []

#             for box, track_id in zip(boxes, track_ids):
#                 x1, y1, x2, y2 = box
#                 bbox_center_y = (y1 + y2) // 2

#                 # Depth-based filtering (if depth is available)
#                 if filter_depth_threshold is not None:
#                     depth_values = depth_frame[y1:y2, x1:x2]
#                     if depth_values.size > 0:
#                         person_depth = np.mean(depth_values)

#                     if person_depth >= filter_depth_threshold:
#                         continue  # Skip this person as they are behind ID 1 in depth

#                 # Y-coordinate-based filtering
#                 if filter_line_y is not None and bbox_center_y <= filter_line_y:
#                     continue  # Skip this person as they are behind ID 1

#                 # Perform feature extraction and similarity comparison as before
#                 person_img = frame[y1:y2, x1:x2]

#                 # Extract features
#                 global_features = self.extract_global_features(person_img)
#                 upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram = self.extract_local_features(person_img)

#                 # Calculate combined similarity
#                 max_similarity = -1
#                 max_similarity_id = -1

#                 for known_id, feature_set in known_persons.items():
#                     known_global_features, known_upper_body_features, known_lower_body_features, known_upper_body_histogram, known_lower_body_histogram = feature_set

#                     combined_similarity = self.calculate_combined_similarity(
#                         global_features,
#                         known_global_features,
#                         upper_body_features,
#                         known_upper_body_features,
#                         lower_body_features,
#                         known_lower_body_features,
#                         upper_body_histogram,
#                         known_upper_body_histogram,
#                         lower_body_histogram,
#                         known_lower_body_histogram,
#                     )

#                     # Update similarity scores buffer
#                     self.similarity_scores_buffer[known_id].append(combined_similarity)
#                     avg_similarity = np.mean(self.similarity_scores_buffer[known_id])
#                     max_similarity_score = max(self.similarity_scores_buffer[known_id])
#                     min_similarity_score = min(self.similarity_scores_buffer[known_id])

#                     # Decision based on similarity threshold and range
#                     if (min_similarity_score <= combined_similarity <= max_similarity_score) and (avg_similarity >= SIMILARITY_THRESHOLD):
#                         if avg_similarity > max_similarity:
#                             max_similarity = avg_similarity
#                             max_similarity_id = known_id

#                 # Determine ID to assign
#                 if max_similarity_id != -1:
#                     person_id = max_similarity_id
#                 else:
#                     # Assign new ID
#                     new_id = max(known_persons.keys(), default=0) + 1
#                     person_id = new_id
#                     known_persons[person_id] = (global_features, upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram)

#                 # Update tracking info
#                 self.id_last_seen[person_id] = current_time
#                 active_ids.add(person_id)

#                 # Save feature vector history
#                 self.feature_history[person_id].append(global_features)

#                 bbox_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
#                 bbox = (x1, y1, x2, y2)
#                 self.tracked_objects[person_id] = (bbox_center, bbox, self.tracked_objects.get(person_id, (None, None, None))[2])

#                 # Draw bounding box and information on the frame
#                 color = self.get_unique_color(person_id)
#                 cv2.rectangle(self.current_color_image, (x1, y1), (x2, y2), color, 2)
#                 label = f"ID: {person_id}"
#                 text_color = (255, 255, 255)

#                 center_x = (x1 + x2) // 2
#                 center_y = (y1 + y2) // 2
#                 centp = f"Center: ({center_x}, {center_y})"

#                 depth_value = self.tracked_objects[person_id][2]
#                 depth_text = f"Depth: {depth_value}cm" if depth_value is not None else "Depth: N/A"

#                 (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
#                 (center_text_width, center_text_height), _ = cv2.getTextSize(centp, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
#                 (depth_text_width, depth_text_height), _ = cv2.getTextSize(depth_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)

#                 label_y_position = y2 + text_height + 10
#                 center_text_y_position = label_y_position + center_text_height + 10
#                 depth_text_y_position = center_text_y_position + depth_text_height + 10

#                 cv2.rectangle(self.current_color_image, (x1, y2), (x1 + text_width, label_y_position), color, thickness=cv2.FILLED)
#                 cv2.putText(self.current_color_image, label, (x1, y2 + text_height), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
#                 cv2.rectangle(self.current_color_image, (x1, label_y_position + 5), (x1 + center_text_width, center_text_y_position), color, thickness=cv2.FILLED)
#                 cv2.putText(self.current_color_image, centp, (x1, label_y_position + center_text_height + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
#                 cv2.rectangle(self.current_color_image, (x1, center_text_y_position + 5), (x1 + depth_text_width, depth_text_y_position), color, thickness=cv2.FILLED)
#                 cv2.putText(self.current_color_image, depth_text, (x1, center_text_y_position + depth_text_height + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

#         cv2.imshow("Color Image with Tracking", self.current_color_image)
#         cv2.waitKey(1)


#     def publish_id_distance_info(self, id, bbox_center, width, height, depth_value):
#         msg = IdDistanceInfo()
#         msg.id = id
#         msg.bbox_center = Point(x=float(bbox_center[0]), y=float(bbox_center[1]), z=0.0)
#         msg.width = float(width)
#         msg.height = float(height)
#         msg.distance = float(depth_value)
#         self.publisher_id_distance.publish(msg)

#     def extract_global_features(self, image):
#         pil_img = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#         input_img = transform(pil_img).unsqueeze(0).to(device)
#         features = predictor(input_img).cpu().numpy()
#         return features

#     def extract_local_features(self, image):
#         height, width, _ = image.shape
#         segments = [
#             image[:height//2, :],  
#             image[height//2:, :],  
#             image[:, :width//2],   
#             image[:, width//2:]    
#         ]
        
#         local_features = []
#         local_histograms = []
#         for segment in segments:
#             pil_img = PILImage.fromarray(cv2.cvtColor(segment, cv2.COLOR_BGR2RGB))
#             input_img = transform(pil_img).unsqueeze(0).to(device)
#             features = predictor(input_img).cpu().numpy()
#             hist = self.calculate_histogram(segment)
#             local_features.append(features)
#             local_histograms.append(hist)
        
#         upper_body_features = local_features[0]  
#         lower_body_features = local_features[1]  
        
#         upper_body_histogram = local_histograms[0]  
#         lower_body_histogram = local_histograms[1]  

#         return upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram

#     def calculate_histogram(self, image):
#         hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#         hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
#         cv2.normalize(hist, hist)
#         return hist.flatten()

#     def calculate_combined_similarity(self, global_features1, global_features2, upper_body_features1, upper_body_features2, lower_body_features1, lower_body_features2, upper_body_histogram1, upper_body_histogram2, lower_body_histogram1, lower_body_histogram2):
#         global_similarity = np.dot(global_features1, global_features2.T) / (np.linalg.norm(global_features1) * np.linalg.norm(global_features2))
#         upper_body_similarity = np.dot(upper_body_features1, upper_body_features2.T) / (np.linalg.norm(upper_body_features1) * np.linalg.norm(upper_body_features2))
#         lower_body_similarity = np.dot(lower_body_features1, lower_body_features2.T) / (np.linalg.norm(lower_body_features1) * np.linalg.norm(lower_body_features2))
#         upper_body_hist_similarity = cv2.compareHist(upper_body_histogram1, upper_body_histogram2, cv2.HISTCMP_CORREL)
#         lower_body_hist_similarity = cv2.compareHist(lower_body_histogram1, lower_body_histogram2, cv2.HISTCMP_CORREL)
        
#         combined_similarity = (0.3 * global_similarity +  
#                                 0.2 * upper_body_similarity +
#                                 0.25 * lower_body_similarity +
#                                 0.25 * lower_body_hist_similarity)
#         return combined_similarity

#     def force_drive(self, id_to_follow=1, duration=1.0, stop_distance=0.7):
#         if id_to_follow not in self.tracked_objects:
#             return

#         bbox_center, _, depth_value = self.tracked_objects[id_to_follow]

#         if not self.tracked_objects:
#             twist_msg = Twist()
#             twist_msg.linear.x = 0.0
#             twist_msg.angular.z = 0.0
#             self.publisher_cmd_vel.publish(twist_msg)
#             time.sleep(0.1)
#             return

#         for person_id, (_, _, depth) in self.tracked_objects.items():
#             if depth is not None and depth < stop_distance * 100:
#                 twist_msg = Twist()
#                 twist_msg.linear.x = 0.0
#                 twist_msg.angular.z = 0.0
#                 self.publisher_cmd_vel.publish(twist_msg)
#                 time.sleep(0.1)
#                 return

#         camera_center_x = self.current_color_image.shape[1] // 2
#         angular_z = 0.0

#         offset_x = bbox_center[0] - camera_center_x

#         max_angular_speed = 0.35
#         angular_z = -max_angular_speed * (offset_x / camera_center_x)
#         angular_z = max(-max_angular_speed, min(max_angular_speed, angular_z))

#         safty_distance_diff = depth_value - (stop_distance * 100)

#         twist_msg = Twist()
#         twist_msg.linear.x = 0.1
#         twist_msg.angular.z = angular_z
#         self.publisher_cmd_vel.publish(twist_msg)
#         time.sleep(0.1)

# def main(args=None):
#     rclpy.init(args=args)
#     node = AllInOneNode()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()


