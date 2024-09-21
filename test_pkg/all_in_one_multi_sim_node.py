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
SIMILARITY_THRESHOLD = 0.90

class AllInOneMultiSimNode(Node):
    def __init__(self):
        super().__init__('all_in_one_multi_sim_node')
        
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

        # Initialize a dictionary to store similarities for each person across frames
        self.similarity_history = {}  # person_id -> list of similarities
        self.frame_count = 5  # Number of frames to average over

        self.get_logger().info("AllInOneMultiSimNode initialized")

    # Function to generate a unique color based on ID
    def get_unique_color(self, id):
        np.random.seed(id)  # Seed with the ID for consistency
        return tuple(np.random.randint(0, 255, 3).tolist())

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

            # Update tracking information with depth value
            self.tracked_objects[person_id] = (bbox_center, bbox, depth_value)

            # Update the last seen time for the ID
            self.id_last_seen[person_id] = current_time
            active_ids.add(person_id)

            # Handle object detection and following
            if 1 in self.tracked_objects:
                self.force_drive(id_to_follow=1)
            else:
                self.get_logger().info("ID 1 is not being tracked. Not initiating force drive.")

        # Display the processed image in real-time
        cv2.imshow("Color Image with Tracking", self.current_color_image)
        cv2.waitKey(1)

    def color_image_callback(self, msg):
        self.get_logger().info("Received color image")


        # Define weights for each similarity
        weight_global = 0.22
        weight_upper_body = 0.13
        weight_lower_body = 0.15
        weight_shoe = 0.13
        weight_upper_body_hist = 0.11
        weight_lower_body_hist = 0.13
        weight_shoe_hist = 0.13

        # Convert image
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.current_color_image = frame.copy()
        self.get_logger().info("Color image converted to OpenCV format")

        results = model.track(classes=[0], max_det=4, show=False, source=frame, stream=True, tracker=tracker_config)
        results = list(results)  # Convert generator to list
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
                upper_body_features, lower_body_features, shoe_features, upper_body_histogram, lower_body_histogram, shoe_histogram = self.extract_local_features(person_img)

                max_similarity_id = -1
                identified = False

                for known_id, (known_global_features, known_upper_body_features, known_lower_body_features, known_shoe_features, 
                known_upper_body_histogram, known_lower_body_histogram, known_shoe_histogram) in known_persons.items():

                    # Calculate similarities separately
                    global_similarity = self.calculate_similarity(global_features, known_global_features)
                    upper_body_similarity = self.calculate_similarity(upper_body_features, known_upper_body_features)
                    lower_body_similarity = self.calculate_similarity(lower_body_features, known_lower_body_features)
                    shoe_similarity = self.calculate_similarity(shoe_features, known_shoe_features)

                    upper_body_hist_similarity = self.calculate_histogram_similarity(upper_body_histogram, known_upper_body_histogram)
                    lower_body_hist_similarity = self.calculate_histogram_similarity(lower_body_histogram, known_lower_body_histogram)
                    shoe_hist_similarity = self.calculate_histogram_similarity(shoe_histogram, known_shoe_histogram)

                    # Apply weights to similarities
                    weighted_similarity = (
                        weight_global * global_similarity +
                        weight_upper_body * upper_body_similarity +
                        weight_lower_body * lower_body_similarity +
                        weight_shoe * shoe_similarity +
                        weight_upper_body_hist * upper_body_hist_similarity +
                        weight_lower_body_hist * lower_body_hist_similarity +
                        weight_shoe_hist * shoe_hist_similarity
                    )

                    # Update similarity history
                    if known_id not in self.similarity_history:
                        self.similarity_history[known_id] = []

                    self.similarity_history[known_id].append(weighted_similarity)

                    # Maintain only the last 'frame_count' similarities
                    if len(self.similarity_history[known_id]) > self.frame_count:
                        self.similarity_history[known_id].pop(0)

                    # Calculate the average similarity over the last 'frame_count' frames
                    average_similarity = np.mean(self.similarity_history[known_id])

                    
                    # Determine if person is identified by comparing the average similarity over the last 'frame_count' frames
                    if average_similarity > SIMILARITY_THRESHOLD:  # Adjust threshold if necessary
                        max_similarity_id = known_id
                        identified = True
                        break  # Exit the loop as person is identified
                
                if identified:
                    person_id = max_similarity_id
                else:
                    person_id = int(track_id)
                    if person_id not in known_persons:
                        new_id = max(known_persons.keys(), default=0) + 1
                        person_id = new_id
                    known_persons[person_id] = (global_features, upper_body_features, lower_body_features, shoe_features,
                                                upper_body_histogram, lower_body_histogram, shoe_histogram)

                self.id_last_seen[person_id] = current_time
                active_ids.add(person_id)

                # Calculate the center coordinates of the bounding box
                bbox_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                bbox = (x1, y1, x2, y2)
                self.tracked_objects[person_id] = (bbox_center, bbox, self.tracked_objects.get(person_id, (None, None, None))[2])  # Use stored depth value

                # Get unique color for the person ID
                color = self.get_unique_color(person_id)

                # Draw bounding box in real-time
                cv2.rectangle(self.current_color_image, (x1, y1), (x2, y2), color, 2)
                label = f"ID: {person_id}"

                # Set text color to white
                text_color = (255, 255, 255)

                # Define center point text
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                centp = f"Center: ({center_x}, {center_y})"

                # Retrieve depth value
                depth_value = self.tracked_objects[person_id][2]
                depth_text = f"Depth: {depth_value}cm" if depth_value is not None else "Depth: N/A"

                # Calculate text size for background rectangles
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                (center_text_width, center_text_height), _ = cv2.getTextSize(centp, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                (depth_text_width, depth_text_height), _ = cv2.getTextSize(depth_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)

                # Calculate the position for the background rectangles and text to display them at the bottom left of the bounding box
                label_y_position = y2 + text_height + 10
                center_text_y_position = label_y_position + center_text_height + 10
                depth_text_y_position = center_text_y_position + depth_text_height + 10

                # Draw a filled rectangle as background for the ID text, matching the bounding box color
                cv2.rectangle(self.current_color_image, (x1, y2), (x1 + text_width, label_y_position), color, thickness=cv2.FILLED)

                # Draw the ID text on top of the rectangle
                cv2.putText(self.current_color_image, label, (x1, y2 + text_height), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

                # Draw a filled rectangle as background for the center point text, below the ID text
                cv2.rectangle(self.current_color_image, (x1, label_y_position + 5), (x1 + center_text_width, center_text_y_position), color, thickness=cv2.FILLED)

                # Draw the center point text
                cv2.putText(self.current_color_image, centp, (x1, label_y_position + center_text_height + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

                # Draw a filled rectangle as background for the depth text, below the center point text
                cv2.rectangle(self.current_color_image, (x1, center_text_y_position + 5), (x1 + depth_text_width, depth_text_y_position), color, thickness=cv2.FILLED)

                # Draw the depth text
                cv2.putText(self.current_color_image, depth_text, (x1, center_text_y_position + depth_text_height + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)


                # 현재 시간을 가져옵니다.
                current_time = self.get_clock().now()

                # 15초 동안 ID를 유지하다가 안 보이면 유사도 기록을 삭제
                for known_id in list(self.similarity_history.keys()):
                    if known_id in self.id_last_seen:
                        # ID가 마지막으로 감지된 이후 경과한 시간을 계산합니다.
                        time_since_last_seen = (current_time - self.id_last_seen[known_id]).nanoseconds / 1e9  # 초 단위로 변환
                        if time_since_last_seen > 15:
                            self.get_logger().info(f"Removing similarity history for ID {known_id} due to inactivity.")
                            del self.similarity_history[known_id]
                            del self.id_last_seen[known_id]  # `id_last_seen`에서도 제거
                    else:
                        # `id_last_seen`에 기록이 없는 경우도 유사도 기록을 삭제합니다.
                        del self.similarity_history[known_id]



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

        # Define regions for upper body, lower body, and shoes
        upper_body = image[:height//3, :]
        lower_body = image[height//3: 2*height//3, :]
        shoes = image[2*height//3:, :]

        segments = [upper_body, lower_body, shoes]
        local_features = []
        local_histograms = []

        for segment in segments:
            pil_img = PILImage.fromarray(cv2.cvtColor(segment, cv2.COLOR_BGR2RGB))
            input_img = transform(pil_img).unsqueeze(0).to(device)
            features = predictor(input_img).cpu().numpy()
            hist = self.calculate_histogram(segment)
            local_features.append(features)
            local_histograms.append(hist)

        # Unpack features and histograms
        upper_body_features = local_features[0]
        lower_body_features = local_features[1]
        shoe_features = local_features[2]

        upper_body_histogram = local_histograms[0]
        lower_body_histogram = local_histograms[1]
        shoe_histogram = local_histograms[2]

        return upper_body_features, lower_body_features, shoe_features, upper_body_histogram, lower_body_histogram, shoe_histogram

    def calculate_histogram(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()

    # Removed the combined similarity function since it is no longer needed

    # New methods for separate similarity calculations
    def calculate_similarity(self, features1, features2):
        """Calculates cosine similarity between two feature vectors."""
        return np.dot(features1, features2.T) / (np.linalg.norm(features1) * np.linalg.norm(features2))

    def calculate_histogram_similarity(self, hist1, hist2):
        """Calculates histogram similarity using correlation."""
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    def force_drive(self, id_to_follow=1, duration=1.0, stop_distance=0.7):  # stop_distance is in meters
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

        # Check if any objects are detected in the frame
        if not self.tracked_objects:
            self.get_logger().info('No objects detected in the frame. Stopping force drive.')
            twist_msg = Twist()
            twist_msg.linear.x = 0.0
            twist_msg.angular.z = 0.0
            self.publisher_cmd_vel.publish(twist_msg)
            time.sleep(0.1)
            return

        # Check if any tracked object is too close
        for person_id, (_, _, depth) in self.tracked_objects.items():
            if depth is not None and depth < stop_distance * 100:  # depth is in cm, convert stop_distance to cm
                self.get_logger().info(f'Object ID {person_id} is too close: {depth / 100:.2f} meters. Stopping force drive.')
                twist_msg = Twist()
                twist_msg.linear.x = 0.0
                twist_msg.angular.z = 0.0
                self.publisher_cmd_vel.publish(twist_msg)
                time.sleep(0.1)
                return

        camera_center_x = self.current_color_image.shape[1] // 2
        angular_z = 0.0

        # Calculate how far the center of the bbox is from the camera's center
        offset_x = bbox_center[0] - camera_center_x

        # Maximum angular speed setting (e.g., set to 0.35)
        max_angular_speed = 0.35

        # Adjust rotational speed proportionally based on the offset from the camera center
        angular_z = -max_angular_speed * (offset_x / camera_center_x)

        # Clamp angular_z to be no larger than max_angular_speed and no smaller than -max_angular_speed
        angular_z = max(-max_angular_speed, min(max_angular_speed, angular_z))

        self.get_logger().info(f'Calculated angular_z: {angular_z:.2f} for offset: {offset_x}')

        # If no object is too close, proceed with driving
        safty_distance_diff = depth_value - (stop_distance * 100)
        self.get_logger().info(f'Object ID {id_to_follow} distance: {depth_value}cm, Distance between Object and Safety Zone: {safty_distance_diff}cm')

        twist_msg = Twist()
        twist_msg.linear.x = 0.1  # Move forward
        twist_msg.angular.z = angular_z  # Adjust orientation
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
