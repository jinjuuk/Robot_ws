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
import time

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

# ID 1의 특징 벡터를 저장
id_1_features = None

# 유사도 임계값 설정
SIMILARITY_THRESHOLD = 0.97

class AllInOneMultiSimNode(Node):
    def __init__(self):
        super().__init__('all_in_one_multi_sim_node')
        
        if torch.cuda.is_available():
            cuda_device_name = torch.cuda.get_device_name(0)
            self.get_logger().info(f"CUDA is available. Using GPU: {cuda_device_name}")
        else:
            self.get_logger().info("CUDA is not available. Using CPU")

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

        self.bridge = CvBridge()
        self.tracked_objects = {}
        self.id_last_seen = {}
        self.last_valid_depth = {}
        self.closest_obstacle_distance = float('inf')
        self.current_color_image = None

        self.get_logger().info("AllInOneMultiSimNode initialized")

    def depth_image_callback(self, msg):
        if not self.tracked_objects:
            return

        depth_frame = self.bridge.imgmsg_to_cv2(msg, '16UC1')
        self.closest_obstacle_distance = np.min(depth_frame) / 10.0

        active_ids = set()
        current_time = self.get_clock().now()

        for person_id, (bbox_center, bbox, _) in self.tracked_objects.items():
            if bbox is None:
                continue

            x1, y1, x2, y2 = bbox
            x, y = bbox_center
            width = x2 - x1
            height = y2 - y1

            if 0 <= x < depth_frame.shape[1] and 0 <= y < depth_frame.shape[0]:
                depth_value = round(depth_frame[y, x] / 10, 2)
                
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

            if person_id == 1:
                self.force_drive(id_to_follow=1)

        cv2.imshow("Color Image with Tracking", self.current_color_image)
        cv2.waitKey(1)

    def color_image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.current_color_image = frame.copy()

        results = model.track(classes=[0], max_det=4, show=False, source=frame, stream=True, tracker=tracker_config)
        results = list(results)

        active_ids = set()
        current_time = self.get_clock().now()

        global id_1_features

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy().astype(int) if result.boxes is not None else []
            track_ids = result.boxes.id.cpu().numpy().astype(int) if result.boxes.id is not None else []

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
                person_img = frame[y1:y2, x1:x2]

                global_features = self.extract_global_features(person_img)

                if id_1_features is None:
                    # ID 1이 설정되지 않은 경우: 가장 가까운 사람을 ID 1로 설정
                    closest_id = self.get_closest_person_id(frame, boxes, track_ids)
                    if closest_id == track_id:
                        id_1_features = global_features
                        person_id = 1
                    else:
                        continue
                else:
                    # ID 1이 이미 설정된 경우: 유사도 검사
                    similarity = self.calculate_similarity(global_features, id_1_features)
                    if similarity > SIMILARITY_THRESHOLD:
                        person_id = 1
                    else:
                        continue

                bbox_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                bbox = (x1, y1, x2, y2)
                self.tracked_objects[person_id] = (bbox_center, bbox, self.tracked_objects.get(person_id, (None, None, None))[2])
                self.id_last_seen[person_id] = current_time
                active_ids.add(person_id)

                color = self.get_unique_color(person_id)
                cv2.rectangle(self.current_color_image, (x1, y1), (x2, y2), color, 2)
                label = f"ID: {person_id}"

                text_color = (255, 255, 255)
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                label_y_position = y2 + text_height + 10

                cv2.rectangle(self.current_color_image, (x1, y2), (x1 + text_width, label_y_position), color, thickness=cv2.FILLED)
                cv2.putText(self.current_color_image, label, (x1, y2 + text_height), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

    def get_closest_person_id(self, frame, boxes, track_ids):
        """가장 가까운 사람의 ID를 반환하는 함수"""
        min_distance = float('inf')
        closest_id = None
        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # 중심점이 프레임 내에 있는지 확인
            if 0 <= center_x < frame.shape[1] and 0 <= center_y < frame.shape[0]:
                depth_value = frame[center_y, center_x]
                if depth_value < min_distance:
                    min_distance = depth_value
                    closest_id = track_id
        return closest_id

    def extract_global_features(self, image):
        pil_img = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        input_img = transform(pil_img).unsqueeze(0).to(device)
        features = predictor(input_img).cpu().numpy()
        return features

    def calculate_similarity(self, features1, features2):
        return np.dot(features1, features2.T) / (np.linalg.norm(features1) * np.linalg.norm(features2))

    def force_drive(self, id_to_follow=1, duration=1.0, stop_distance=0.7):
        if id_to_follow not in self.tracked_objects:
            return

        bbox_center, _, depth_value = self.tracked_objects[id_to_follow]

        if depth_value is None or depth_value == 0:
            return

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
    node = AllInOneMultiSimNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
