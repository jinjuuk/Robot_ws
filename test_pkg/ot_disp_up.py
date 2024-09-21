
'''
노드이름 바꾸고, setup.py 추가해야함
'''




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

# 알려진 사람들의 특징 벡터와 ID를 저장할 딕셔너리
known_persons = {}

# 초기 설정값
SIMILARITY_THRESHOLD = 0.97

class AllInOneNodeUp(Node):
    def __init__(self):
        super().__init__('all_in_one_up_node')
        
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
            'cmd_vel',
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
        # 장애물까지의 가장 가까운 거리를 저장할 변수 초기화
        self.closest_obstacle_distance = float('inf')

        self.get_logger().info("AllInOneNode initialized")


    # Function to generate a unique color based on ID
    def get_unique_color(self, id):
        np.random.seed(id)  # Seed with the ID for consistency
        return tuple(np.random.randint(0, 255, 3).tolist())

    def color_image_callback(self, msg):
        self.get_logger().info("Received color image")

        # 이미지 변환
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.current_color_image = frame.copy()
        self.get_logger().info("Color image converted to OpenCV format")

        results = model.track(classes=[0], max_det=4, show=False, source=frame, stream=True, tracker=tracker_config)
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
                    self.get_logger().info(f"Known_Id: {known_id}, {max_similarity_id}, Calculated similarity: {similarity}, {max_similarity}")

                    if similarity > max_similarity:
                        max_similarity = similarity
                        max_similarity_id = known_id

                if max_similarity > SIMILARITY_THRESHOLD:
                    if max_similarity_id not in active_ids:
                        person_id = max_similarity_id
                else:
                    person_id = int(track_id)
                    if person_id not in known_persons:
                        new_id = max(known_persons.keys(), default=0) + 1
                        person_id = new_id
                    known_persons[person_id] = (global_features, upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram)

                self.id_last_seen[person_id] = current_time
                active_ids.add(person_id)

                # 바운딩 박스의 중심 좌표 계산
                bbox_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                self.tracked_objects[person_id] = (bbox_center, (x1, y1, x2, y2), None)  # depth_value는 아직 없음

                # Get unique color for the person ID
                color = self.get_unique_color(person_id)

                # 실시간으로 바운딩 박스 표시
                cv2.rectangle(self.current_color_image, (x1, y1), (x2, y2), color, 2)
                label = f"ID: {person_id}"

                # Set text color to white
                text_color = (255, 255, 255)

                # Calculate text size for background rectangle for ID
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)

                # Draw a filled rectangle as background for the ID text, matching the bounding box color
                cv2.rectangle(self.current_color_image, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, thickness=cv2.FILLED)

                # Draw the ID text on top of the rectangle
                cv2.putText(self.current_color_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

                # Additional center point and information display
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                centp = f"Center: ({center_x}, {center_y})"

                # Calculate text size for center point background
                (center_text_width, center_text_height), baseline = cv2.getTextSize(centp, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)

                # Draw a filled rectangle as background for the center point text, below the ID text
                cv2.rectangle(self.current_color_image, (x1, y1 + 5), (x1 + center_text_width, y1 + center_text_height + 15), color, thickness=cv2.FILLED)

                # Draw the center point text
                cv2.putText(self.current_color_image, centp, (x1, y1 + center_text_height + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

    def depth_image_callback(self, msg):
        self.get_logger().info("Received depth image")
        
        if not self.tracked_objects:
            self.get_logger().info("No tracked objects available, skipping depth processing")
            return

        # Convert depth image to OpenCV format
        depth_frame = self.bridge.imgmsg_to_cv2(msg, '16UC1')
        self.get_logger().info("Depth image converted to OpenCV format")

        # Calculate distance to the closest obstacle
        self.closest_obstacle_distance = np.min(depth_frame) / 10.0  # Convert minimum depth value to cm

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

            # Handling depth values, using last valid if current is zero
            if 0 <= x < depth_frame.shape[1] and 0 <= y < depth_frame.shape[0]:
                depth_value = round(depth_frame[y, x] / 10, 2)  # Convert mm to cm with two decimal places
                
                if depth_value == 0:
                    if person_id in self.last_valid_depth:
                        depth_value = self.last_valid_depth[person_id]
                        self.get_logger().info(f"Depth value was 0. Using last valid depth value: {depth_value}cm")
                    else:
                        self.get_logger().info(f"No valid depth value available for Person ID: {person_id}. Skipping.")
                        continue
                else:
                    self.last_valid_depth[person_id] = depth_value
            else:
                if person_id in self.last_valid_depth:
                    depth_value = self.last_valid_depth[person_id]
                    self.get_logger().info(f"Depth value was None. Using last valid depth value: {depth_value}cm")
                else:
                    depth_value = None
                    self.get_logger().info(f"Bbox center {bbox_center} is out of depth frame bounds and no previous depth value available.")


            self.get_logger().info(f"Person ID: {person_id}, Bbox center: {bbox_center}, Width: {width}, Height: {height}, Depth: {depth_value}cm")

            # Handle object detection and following
            if 1 in self.tracked_objects:
                self.force_drive(id_to_follow=1)
            else:
                self.get_logger().info("ID 1 is not being tracked. Not initiating force drive.")

            # Update tracking information with depth value
            self.tracked_objects[person_id] = (bbox_center, bbox, depth_value)

            # Update the last seen time for the ID
            self.id_last_seen[person_id] = current_time
            active_ids.add(person_id)

            # Get unique color for the person ID
            color = self.get_unique_color(person_id)

            # Set text color to white
            text_color = (255, 255, 255)

            # Calculate text for center point display, assuming you need this in depth_image_callback
            centp = f"Center: ({x}, {y})"
            (center_text_width, center_text_height), _ = cv2.getTextSize(centp, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)

            depth_text = f"Depth: {depth_value}cm"
            (depth_text_width, depth_text_height), _ = cv2.getTextSize(depth_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)

            # Adjust the drawing position if needed and draw the rectangle and text
            cv2.rectangle(self.current_color_image, (x1, y1 + center_text_height + 25), (x1 + depth_text_width, y1 + center_text_height + depth_text_height + 35), color, thickness=cv2.FILLED)

            # Draw the depth text
            cv2.putText(self.current_color_image, depth_text, (x1, y1 + center_text_height + depth_text_height + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

            # Publish custom message
            self.publish_id_distance_info(person_id, bbox_center, width, height, depth_value)

        # Display the processed image in real-time
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
        self.get_logger().info(f"Published IdDistanceInfo: ID={id}, Bbox Center={bbox_center}, bbox_Width={width}, bbox_Height={height}, Depth={depth_value}")


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
        
        combined_similarity = (0.2 * global_similarity +  
                                0.1 * upper_body_similarity +
                                0.4 * lower_body_similarity +
                                0.3 * lower_body_hist_similarity)
        return combined_similarity


    def force_drive(self, id_to_follow=1, duration=1.0, stop_distance=1): # stop_distance = 0.5는 50cm
        """
        로봇을 특정 방향으로 주행시키는 함수.

        :param id_to_follow: 추종할 객체의 ID (기본값은 1)
        :param duration: 주행 지속 시간 (초 단위)
        :param stop_distance: 주행을 멈출 장애물까지의 거리 (미터 단위)
        """

        if id_to_follow not in self.tracked_objects:
            self.get_logger().info(f'ID {id_to_follow} is not being tracked.')
            return

        bbox_center, _, depth_value = self.tracked_objects[id_to_follow]

        if depth_value is None or depth_value == 0:
            self.get_logger().info(f'No valid depth value for ID {id_to_follow}. Cannot proceed.')
            return

        camera_center_x = self.current_color_image.shape[1] // 2
        angular_z = 0.0

        # 중심 좌표가 카메라의 중심에서 얼마나 떨어져 있는지 계산
        offset_x = bbox_center[0] - camera_center_x

        # 회전 속도 비율을 조정할 최대값 설정 (예: 0.5로 설정)
        max_angular_speed = 0.35

        # 카메라 중심에서의 오프셋을 기준으로 회전 속도를 비례적으로 조정
        angular_z = -max_angular_speed * (offset_x / camera_center_x)

        # angular_z가 max_angular_speed보다 크거나 작지 않도록 클램핑
        angular_z = max(-max_angular_speed, min(max_angular_speed, angular_z))

        self.get_logger().info(f'Calculated angular_z: {angular_z:.2f} for offset: {offset_x}')


        twist_msg = Twist()


        # 장애물이 일정 거리 안에 있을 경우 멈춤
        if depth_value < stop_distance * 100:  # depth_value는 cm 단위이므로 100을 곱함
            self.get_logger().info(f'Safty distance detected within {depth_value / 100:.2f} meters. Stopping force drive.')
            twist_msg.linear.x = 0.0
            twist_msg.angular.z = 0.0
            self.publisher_cmd_vel.publish(twist_msg)
            time.sleep(0.1)

        else:
            safty_distance_diff =  depth_value - (stop_distance * 100)
            self.get_logger().info(f'Object distance: {depth_value}cm,  Distance between Object and Safty_Zone: {safty_distance_diff}cm')
            twist_msg.linear.x = 0.1  # 직진
            twist_msg.angular.z = angular_z  # 회전 방향
            self.publisher_cmd_vel.publish(twist_msg)
            time.sleep(0.1)
            return 



def main(args=None):
    rclpy.init(args=args)
    node = AllInOneNodeUp()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
