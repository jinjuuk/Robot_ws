import collections
if not hasattr(collections, 'Mapping'):
    import collections.abc
    collections.Mapping = collections.abc.Mapping
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
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


# CUDA 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# YOLOv5 모델 로드
model = YOLO("yolov5su.pt").to(device)

# BotSORT 설정 파일 경로
tracker_config = "/home/jinjuuk/.local/lib/python3.10/site-packages/ultralytics/cfg/trackers/botsort.yaml"

# FastReID 설정
cfg = get_cfg()
cfg.merge_from_file("/home/jinjuuk/fast-reid/configs/Market1501/bagtricks_R50.yml")
cfg.MODEL.WEIGHTS = "/home/jinjuuk/dev_ws/pt_files/market_mgn_R50-ibn.pth"
cfg.MODEL.DEVICE = "cuda"  # GPU 사용 설정

predictor = DefaultPredictor(cfg)
transform = build_transforms(cfg, is_train=False)

# 알려진 사람들의 특징 벡터와 ID를 저장할 딕셔너리
known_persons = {}

# 초기 설정값
SIMILARITY_THRESHOLD = 0.97

class AllInOneNodeTest(Node):
    def __init__(self):
        super().__init__('all_in_one_node_test')
        
        # 로그 출력: CUDA 사용 여부 확인
        if torch.cuda.is_available():
            cuda_device_name = torch.cuda.get_device_name(0)
            self.get_logger().info(f"CUDA is available. Using GPU: {cuda_device_name}")
        else:
            self.get_logger().info("CUDA is not available. Using CPU")

        # 구독자 설정
        self.subscription_color = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.color_image_callback,
            10
        )
        self.subscription_depth = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.depth_image_callback,
            10
        )
        
        # 발행자 설정
        self.publisher_cmd_vel = self.create_publisher(
            Twist,
            'Twist/cmd_vel',
            10
        )
        self.publisher_id_distance = self.create_publisher(
            IdDistanceInfo,  
            'id_distance_info',
            10
        )
        
        # OpenCV를 위한 CVBridge 초기화
        self.bridge = CvBridge()

        # 트래킹 정보 저장
        self.tracked_objects = {}  # id -> (bbox_center, bbox, depth)
        self.id_last_seen = {}  # id -> last seen time
         # 각 person_id에 대한 마지막 유효한 깊이 값을 저장하는 딕셔너리 초기화
        self.last_valid_depth = {}

        self.get_logger().info("AllInOneNode initialized")


    def color_image_callback(self, msg):
        self.get_logger().info("Received color image")
        
        # 이미지 변환
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.current_color_image = frame.copy()
        self.get_logger().info("Color image converted to OpenCV format")
        
        results = model.track(classes=[0], max_det=5, show=False, source=frame, stream=True, tracker=tracker_config)
        results = list(results)  # generator를 list로 변환
        self.get_logger().info(f"YOLO tracking complete. Detected {len(results)} results")

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
                    similarity = self.calculate_combined_similarity(global_features, known_global_features, upper_body_features, known_upper_body_features, lower_body_features, known_lower_body_features, upper_body_histogram, known_upper_body_histogram, lower_body_histogram, known_lower_body_histogram)
                    self.get_logger().info(f"Known_Id: {known_id}, {max_similarity_id},  Calculated similarity:  {similarity}, {max_similarity}")

                    if similarity > max_similarity:
                        max_similarity = similarity
                        max_similarity_id = known_id

                if max_similarity > SIMILARITY_THRESHOLD:
                    if max_similarity_id not in active_ids:
                        person_id = max_similarity_id
                        color = (0, 255, 0)
                else:
                    person_id = int(track_id)
                    if person_id in known_persons:
                        pass
                    else:
                        new_id = max(known_persons.keys(), default=0) + 1
                        person_id = new_id
                    known_persons[person_id] = (global_features, upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram)
                    color = (0, 0, 255)

                self.id_last_seen[person_id] = current_time
                active_ids.add(person_id)

                # 바운딩 박스의 중심 좌표 계산
                bbox_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                self.tracked_objects[person_id] = (bbox_center, (x1, y1, x2, y2), None)  # depth_value는 아직 없음

                # 실시간으로 바운딩 박스 표시
                cv2.rectangle(self.current_color_image, (x1, y1), (x2, y2), color, 2)
                label = f"ID: {person_id}"
                cv2.putText(self.current_color_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                # 추가적으로 센터 포인트 및 그 정보 표시
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                centp = f"Center: ({center_x}, {center_y})"
                cv2.putText(self.current_color_image, centp, (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


    def depth_image_callback(self, msg):
        self.get_logger().info("Received depth image")
        
        if not self.tracked_objects:
            self.get_logger().info("No tracked objects available, skipping depth processing")
            return

        # 깊이 이미지를 OpenCV 형식으로 변환
        depth_frame = self.bridge.imgmsg_to_cv2(msg, '16UC1')
        self.get_logger().info("Depth image converted to OpenCV format")
        
        active_ids = set()
        current_time = self.get_clock().now()

        for person_id, (bbox_center, bbox, _) in self.tracked_objects.items():
            if bbox is None:
                self.get_logger().info(f"No bbox available for Person ID: {person_id}, skipping.")
                continue
            
            x1, y1, x2, y2 = bbox
            x, y = bbox_center
            width = x2 - x1
            height = y2 - y1

            # 깊이 값이 0인 경우, 이전에 유효했던 깊이 값으로 대체하는 로직
            if 0 <= x < depth_frame.shape[1] and 0 <= y < depth_frame.shape[0]:
                depth_value = round(depth_frame[y, x] / 10, 2)  # mm를 cm로 변환 후 소수점 2자리까지 표시
                
                # 깊이 값이 0인 경우
                if depth_value == 0:
                    if person_id in self.last_valid_depth:
                        depth_value = self.last_valid_depth[person_id]
                        self.get_logger().info(f"Depth value was 0. Using last valid depth value: {depth_value}cm")
                    else:
                        self.get_logger().info(f"No valid depth value available for Person ID: {person_id}. Skipping.")
                        continue
                else:
                    # 깊이 값이 유효한 경우, 해당 값을 저장
                    self.last_valid_depth[person_id] = depth_value
            else:
                depth_value = None
                self.get_logger().info(f"Bbox center {bbox_center} is out of depth frame bounds.")

            self.get_logger().info(f"Person ID: {person_id}, Bbox center: {bbox_center}, Width: {width}, Height: {height}, Depth: {depth_value}cm")

            # Update tracking information with depth value
            self.tracked_objects[person_id] = (bbox_center, bbox, depth_value)

            # Update the last seen time for the ID
            self.id_last_seen[person_id] = current_time
            active_ids.add(person_id)

            # Define color for bounding box and text
            color = (0, 255, 0)

            depth_text = f"Depth: {depth_value}cm"
            cv2.putText(self.current_color_image, depth_text, (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # 커스텀 메시지 발행
            self.publish_id_distance_info(person_id, bbox_center, width, height, depth_value)

        # 실시간으로 처리된 이미지를 표시
        cv2.imshow("Color Image with Tracking", self.current_color_image)
        cv2.waitKey(1)
        



    def publish_id_distance_info(self, id, bbox_center, width, height, depth_value):
        self.get_logger().info(f"Preparing to publish IdDistanceInfo: ID={id}, Bbox Center={bbox_center}, Width={width}, Height={height}, Depth={depth_value}")
        
        msg = IdDistanceInfo()
        msg.id = id
        msg.bbox_center = Point(x=float(bbox_center[0]), y=float(bbox_center[1]), z=0.0)
        msg.width = float(width)
        msg.height = float(height)
        msg.distance = float(depth_value)
        
        # 발행 전 메시지 정보 로그
        self.get_logger().info(f"IdDistanceInfo message prepared: {msg}")
        
        self.publisher_id_distance.publish(msg)
        self.get_logger().info(f"Published IdDistanceInfo: ID={id}, Bbox Center={bbox_center}, Width={width}, Height={height}, Depth={depth_value}")


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
        
        combined_similarity = (0.3 * global_similarity +  
                                0.1 * upper_body_similarity +
                                0.3 * lower_body_similarity +
                                0.3 * lower_body_hist_similarity)
        return combined_similarity


def main(args=None):
    rclpy.init(args=args)
    node = AllInOneNodeTest()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
