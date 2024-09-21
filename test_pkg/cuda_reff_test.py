import cv2
import numpy as np
from ultralytics import YOLO
import torch
from queue import Queue
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles

from sensor_msgs.msg import Image
from std_srvs.srv import SetBool
from hi5_message.srv import OrderCall

from cv_bridge import CvBridge, CvBridgeError

class Yolov8Node(Node):
    def __init__(self):
        super().__init__('yolo_node')
        self.init()

        qos_profile_sensor = QoSPresetProfiles.SENSOR_DATA.value
        self.processing_thread = None
        self.processing_lock = threading.Lock()

        self.img_subscriber = self.create_subscription(Image, "image_raw_cam2", self.change_image, qos_profile_sensor)
        self.robot_status_service = self.create_service(SetBool, "robot_status_yolo", self.robot_status_callback)

        self.trash_detect_service = self.create_client(SetBool, "trash_detect")
        self.trash_position_detect_service = self.create_client(OrderCall, "trash_position_detect")

        self.wait_for_service(self.trash_detect_service, "trash detect service")

    def init(self):
        self.bridge = CvBridge()
        self.frame = None
        self.detected_objects = []
        self.trash_detected = False
        self.avg_angle = None

        self.model = YOLO('/home/bo/Downloads/newjeans.pt')  # YOLOv8 모델을 로드
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if torch.cuda.is_available():
            self.model.to(device)
        self.get_logger().info(f'Using device: {device}')

        self.initial_gray = None
        self.detection_enabled = False
        self.robot_status = False

        self.camera_matrix = np.array([[474.51901407, 0, 302.47811758],
                                        [0, 474.18970657, 250.66191453],
                                        [0, 0, 1]])
        self.dist_coeffs = np.array([[-0.06544764, -0.07654065, -0.00761827, -0.00279316, 0.08062307]])
        self.object_coords = {}
        self.robot_origin_x = 295
        self.robot_origin_y = 184
        self.pixel_to_mm_ratio = 1.55145

        # 큐를 생성하고 최대 크기를 설정합니다.
        self.frame_queue = Queue(maxsize=30)
        self.angle_list = []

        self.roi_x_large = 52
        self.roi_y_large = 0
        self.roi_width_large = 500
        self.roi_height_large = 310

        self.roi_x_medium = 270
        self.roi_y_medium = 0
        self.roi_width_medium = 270
        self.roi_height_medium = 60

        self.roi_x_small = 464
        self.roi_y_small = 118
        self.roi_width_small = 35
        self.roi_height_small = 35

        self.threshold1 = 50
        self.threshold2 = 150
        self.diff_thresh = 45
        self.confidence_threshold = 0.5
        self.h_min = 0
        self.h_max = 179
        self.s_min = 0
        self.s_max = 255
        self.v_min = 0
        self.v_max = 255

    def wait_for_service(self, client, service_name):
        while not client.wait_for_service(timeout_sec=5.0):
            self.get_logger().info(f'Waiting for {service_name}...')
        self.get_logger().info(f"{service_name} is ready")

    def undistored_frame(self, frame):
        undistored_frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)
        return undistored_frame

    def change_image(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="8UC3")
            self.frame = self.undistored_frame(frame)
        except CvBridgeError as e:
            self.get_logger().error(f"Conversion failed: {e}")
            return

        frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.GaussianBlur(frame_gray, (21, 21), 0)

        if not self.detection_enabled:
            self.initial_gray = frame_gray
            self.detection_enabled = True
            self.get_logger().info("Initial background set, detection enabled.")
            return

        if not self.robot_status:
            if self.detection_enabled and self.initial_gray is not None:
                if self.detect_object(frame_gray):
                    self.get_logger().info("Trash detected!")
                    self.add_frame_to_queue(self.frame)
                else:
                    pass
            else:
                pass
        else:
            pass

    def add_frame_to_queue(self, frame):
        while True:
            if self.frame_queue.full():
                self.get_logger().info("Queue is full, processing detection.")
                break
            else:
                self.frame_queue.put(frame)
        self.process_detection_queue()

    def robot_status_callback(self, req, res):
        self.get_logger().info(f"Robot Status: {req.data}")
        self.robot_status = req.data
        res.success = True
        res.message = "success"
        return res

    def trash_position_detect_call(self, x, y, angle):
        request = OrderCall.Request()
        request.data = f"{x},{y},{angle}"
        self.trash_position_detect_service.call_async(request)
        self.robot_status = True

    def trash_detect_call(self):
        request = SetBool.Request()
        request.data = self.trash_detected
        self.trash_detect_service.call_async(request)
        self.robot_status = True

    def calculate_angle(self, pt1, pt2):
        angle = np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0]) * 180.0 / np.pi
        return angle

    def find_similar_object(self, center_x, center_y):
        threshold = 50  # example threshold value
        for obj_id, coords in self.object_coords.items():
            if coords:
                avg_x = sum([coord[0] for coord in coords]) / len(coords)
                avg_y = sum([coord[1] for coord in coords]) / len(coords)
                distance = np.sqrt((avg_x - center_x) ** 2 + (avg_y - center_y) ** 2)
                if distance < threshold:
                    return obj_id
        return None
    
    def detect_object(self, frame_gray):
            frame_delta = cv2.absdiff(self.initial_gray, frame_gray)
            _, diff_mask = cv2.threshold(frame_delta, self.diff_thresh, 255, cv2.THRESH_BINARY)
            diff_mask = cv2.dilate(diff_mask, None, iterations=2)

            edges = cv2.Canny(frame_gray, self.threshold1, self.threshold2)
            edges = cv2.dilate(edges, None, iterations=1)
            edges = cv2.bitwise_and(edges, edges, mask=diff_mask)

            hsv_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
            lower_bound = np.array([self.h_min, self.s_min, self.v_min])
            upper_bound = np.array([self.h_max, self.s_max, self.v_max])
            hsv_mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)

            detection_mask = np.zeros(diff_mask.shape, dtype=np.uint8)
            detection_mask[self.roi_y_large:self.roi_y_large + self.roi_height_large, self.roi_x_large:self.roi_x_large + self.roi_width_large] = 255
            detection_mask[self.roi_y_medium:self.roi_y_medium + self.roi_height_medium, self.roi_x_medium:self.roi_x_medium + self.roi_width_medium] = 0
            detection_mask[self.roi_y_small:self.roi_y_small + self.roi_height_small, self.roi_x_small:self.roi_x_small + self.roi_width_small] = 0

            combined_mask = cv2.bitwise_or(diff_mask, edges)
            combined_mask = cv2.bitwise_and(combined_mask, combined_mask, mask=detection_mask)
            combined_mask = cv2.bitwise_and(combined_mask, hsv_mask)

            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            self.detected_objects = []

            for cnt in contours:
                if cv2.contourArea(cnt) > 100:
                    x, y, w, h = cv2.boundingRect(cnt)
                    if self.is_within_roi(x, y, w, h):
                        self.detected_objects.append((x, y, w, h))
                        cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        self.trash_detected = True

            return len(self.detected_objects) > 0

    def is_within_roi(self, x, y, w, h):
            return (self.roi_x_large <= x <= self.roi_x_large + self.roi_width_large and
                    self.roi_y_large <= y <= self.roi_y_large + self.roi_height_large)

    def detect_position(self, frame_queue):
        x_list = []
        y_list = []
        angle_list = []
        try:
            while not frame_queue.empty():
                frame = frame_queue.get()
                results = self.model(frame, conf=0.5)
                for result in results:
                    if result.masks is None:
                        self.get_logger().info("No masks detected")
                        continue

                    masks = result.masks
                    for mask in masks.data.cpu().numpy():
                        mask = mask.astype(np.uint8)
                        if mask.shape[:2] != (frame.shape[0], frame.shape[1]):
                            mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                        else:
                            mask_resized = mask

                        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            cnt = max(contours, key=cv2.contourArea)
                            rect = cv2.minAreaRect(cnt)
                            box = cv2.boxPoints(rect)
                            box = np.intp(box)
                            cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

                            M = cv2.moments(cnt)
                            if M["m00"] != 0:
                                center_x = int(M["m10"] / M["m00"])
                                center_y = int(M["m01"] / M["m00"])
                            else:
                                center_x, center_y = 0, 0

                            bottom_points = sorted(box, key=lambda pt: pt[1], reverse=True)[:2]
                            bottom_center_x = int((bottom_points[0][0] + bottom_points[1][0]) / 2)
                            bottom_center_y = int((bottom_points[0][1] + bottom_points[1][1]) / 2)

                            angle = self.calculate_angle((center_x, center_y), (bottom_center_x, bottom_center_y))

                            obj_id = self.find_similar_object(center_x, center_y)
                            if not obj_id:
                                obj_id = f"object_{len(self.object_coords)}"
                                self.object_coords[obj_id] = [(center_x, center_y)]
                            else:
                                self.object_coords[obj_id].append((center_x, center_y))

                            distance_x = (center_x - self.robot_origin_x) * self.pixel_to_mm_ratio
                            distance_y = (center_y - self.robot_origin_y) * self.pixel_to_mm_ratio

                            x_list.append(distance_x)
                            y_list.append(distance_y)
                            angle_list.append(angle)
                            print

            if x_list and y_list and angle_list:
                avg_x = sum(x_list) / len(x_list)
                avg_y = sum(y_list) / len(y_list)
                avg_angle = sum(angle_list) / len(angle_list)
                return avg_x, avg_y, avg_angle
            else:
                return None, None, None
        except Exception as e:
            self.get_logger().error(f"Error in detect_position: {e}")
            return None, None, None

    def process_detection_queue(self):
        try:
            if not self.frame_queue.empty():
                x, y, angle = self.detect_position(self.frame_queue)
                if x is not None and y is not None and angle is not None:
                    self.get_logger().info(f"x: {x}, y: {y}, angle: {angle}")
                    self.trash_position_detect_call(x, y, angle)
                else:
                    self.trash_detect_call()
        except Exception as e:
            self.get_logger().error(f"Error in process_detection_queue: {e}")

def main(args=None):
    rclpy.init(args=args)
    yolov8_node = Yolov8Node()
    rclpy.spin(yolov8_node)
    yolov8_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()