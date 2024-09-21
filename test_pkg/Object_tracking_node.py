import rclpy as rp
from rclpy.node import Node
from rclpy.time import Time  # Time 객체 가져오기
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from std_msgs.msg import String
import cv2 
from cv_bridge import CvBridge
import numpy as np
import torch
from fastreid.config import get_cfg
from fastreid.engine import DefaultPredictor
from fastreid.data.transforms import build_transforms
from PIL import Image as PILImage
from ultralytics import YOLO
import time

class ObjectTrackingNode(Node):
    def __init__(self):
        super().__init__('object_tracking_node')

        # Image subscriber
        self.subscriber_ = self.create_subscription(
            Image, 
            '/camera/color/image_raw', 
            self.img_sub_callback, 
            10)

        # Object center point publisher
        self.publisher_cp = self.create_publisher(Point, 'object_tracking/centre_point', 10)
        
        # Object ID publisher
        self.publisher_id = self.create_publisher(String, 'object_tracking/id', 10)

        # For converting ROS Image messages to OpenCV images
        self.bridge = CvBridge()

        # Initialize YOLOv5, FastReID, and other necessary components
        self.init_model()

    def init_model(self):
        # YOLOv5 모델 및 FastReID 설정
        self.model = YOLO("yolov5su.pt")
        
        # FastReID 설정
        self.cfg = get_cfg()
        self.cfg.merge_from_file("/home/jinjuuk/fast-reid/configs/Market1501/AGW_R50.yml")
        self.cfg.MODEL.WEIGHTS = "/home/jinjuuk/dev_ws/pt_files/market_agw_R50.pth"
        self.cfg.MODEL.DEVICE = "cuda"
        
        self.predictor = DefaultPredictor(self.cfg)
        self.transform = build_transforms(self.cfg, is_train=False)

        self.known_persons = {}
        self.id_last_seen = {}

        # 설정 값들 초기화
        self.SIMILARITY_THRESHOLD = 0.97
        self.ID_HOLD_TIME = 10
        self.YOLO_CONF_THRESHOLD = 0.60
        self.YOLO_IOU_THRESHOLD = 0.60

    def img_sub_callback(self, msg):
        self.get_logger().info('Received image message')

        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.get_logger().info('Converted ROS Image to OpenCV format')

        current_time = self.get_clock().now()
        self.get_logger().info('Current time recorded')

        results = self.model.track(classes=[0],
                                   max_det=10, 
                                   show=False, 
                                   source=frame, 
                                   conf=self.YOLO_CONF_THRESHOLD,
                                   iou=self.YOLO_IOU_THRESHOLD,
                                   tracker="/home/jinjuuk/.local/lib/python3.10/site-packages/ultralytics/cfg/trackers/botsort.yaml")
        self.get_logger().info('YOLOv5 tracking executed')

        active_ids = set()

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

                for known_id, (known_global_features, known_upper_body_features, known_lower_body_features, known_upper_body_histogram, known_lower_body_histogram) in self.known_persons.items():
                    similarity = self.calculate_combined_similarity(global_features, known_global_features, upper_body_features, known_upper_body_features, lower_body_features, known_lower_body_features, upper_body_histogram, known_upper_body_histogram, lower_body_histogram, known_lower_body_histogram)

                    if similarity > max_similarity:
                        max_similarity = similarity
                        max_similarity_id = known_id

                if max_similarity > self.SIMILARITY_THRESHOLD:
                    if max_similarity_id not in active_ids:
                        person_id = max_similarity_id
                else:
                    person_id = int(track_id)
                    if person_id not in self.known_persons:
                        new_id = max(self.known_persons.keys(), default=0) + 1
                        person_id = new_id
                    self.known_persons[person_id] = (global_features, upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram)

                self.id_last_seen[person_id] = current_time

                active_ids.add(person_id)

                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                cp_msg = Point()
                cp_msg.x = float(center_x)
                cp_msg.y = float(center_y)
                cp_msg.z = 0.0
                self.publisher_cp.publish(cp_msg)

                id_msg = String()
                id_msg.data = str(person_id)
                self.publisher_id.publish(id_msg)

        for person_id in list(self.id_last_seen.keys()):
            now_time = self.get_clock().now().to_msg()
            last_seen_time = self.id_last_seen[person_id].to_msg()

            time_diff_sec = (now_time.sec - last_seen_time.sec) + (now_time.nanosec - last_seen_time.nanosec) / 1e9

            if time_diff_sec > self.ID_HOLD_TIME:
                del self.known_persons[person_id]
                del self.id_last_seen[person_id]

    def extract_global_features(self, image):
        pil_img = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        input_img = self.transform(pil_img).unsqueeze(0)
        features = self.predictor(input_img).cpu().numpy()
        return features

    def calculate_histogram(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()

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
            input_img = self.transform(pil_img).unsqueeze(0)
            features = self.predictor(input_img).cpu().numpy()
            hist = self.calculate_histogram(segment)
            local_features.append(features)
            local_histograms.append(hist)

        upper_body_features = local_features[0]  
        lower_body_features = local_features[1]  

        upper_body_histogram = local_histograms[0]  
        lower_body_histogram = local_histograms[1]  

        return upper_body_features, lower_body_features, upper_body_histogram, lower_body_histogram

    def calculate_combined_similarity(self, global_features1, global_features2, upper_body_features1, upper_body_features2, lower_body_features1, lower_body_features2, upper_body_histogram1, upper_body_histogram2, lower_body_histogram1, lower_body_histogram2):
        global_similarity = np.dot(global_features1, global_features2.T) / (np.linalg.norm(global_features1) * np.linalg.norm(global_features2))
        upper_body_similarity = np.dot(upper_body_features1, upper_body_features2.T) / (np.linalg.norm(upper_body_features1) * np.linalg.norm(upper_body_features2))
        lower_body_similarity = np.dot(lower_body_features1, lower_body_features2.T) / (np.linalg.norm(lower_body_features1) * np.linalg.norm(lower_body_features2))
        upper_body_hist_similarity = cv2.compareHist(upper_body_histogram1, upper_body_histogram2, cv2.HISTCMP_CORREL)
        lower_body_hist_similarity = cv2.compareHist(lower_body_histogram1, lower_body_histogram2, cv2.HISTCMP_CORREL)

        combined_similarity = (0.3 * global_similarity +  
                               0.3 * upper_body_similarity +
                               0.1 * lower_body_similarity +
                               0.3 * lower_body_hist_similarity)

        return combined_similarity


def main(args=None):
    rp.init(args=args)
    node = ObjectTrackingNode()
    rp.spin(node)
    node.destroy_node()
    rp.shutdown()

if __name__ == '__main__':
    main()
