import torch
import rclpy
from rclpy.node import Node
from message.msg import IdDistanceInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import socket
from geometry_msgs.msg import Point

class UDPVISUALTESTNODE(Node):
    def __init__(self):
        super().__init__('udp_visual_test_node')
        
        # CUDA 장치 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            cuda_device_name = torch.cuda.get_device_name(0)
            self.get_logger().info(f"CUDA is available. Using GPU: {cuda_device_name}")
        else:
            self.get_logger().info("CUDA is not available. Using CPU")

        # UDP 소켓 설정
        self.udp_ip = "0.0.0.0"  # 모든 IP에서 수신
        self.udp_port = 5006     # 수신 포트 번호
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.udp_ip, self.udp_port))

        # 발행자 설정
        self.publisher_id_distance = self.create_publisher(
            IdDistanceInfo,  
            'id_distance_info',
            10
        )

        # OpenCV를 위한 CVBridge 초기화
        self.bridge = CvBridge()

        # 이미지 저장을 위한 변수
        self.current_color_image = None
        self.current_depth_image = None
        self.closest_point = None
        self.distance = None
        self.id = None
        self.bbox = None

        self.get_logger().info("CombinedVisualizationNode initialized")

    def udp_receive(self):
        # UDP로부터 데이터 수신
        data, _ = self.sock.recvfrom(65535)

        # 수신한 데이터를 numpy 배열로 변환하고 이미지를 디코딩
        np_data = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

        if img is not None:
            self.color_image_callback(img)
        else:
            self.get_logger().warn("Failed to decode image from UDP data")

    def color_image_callback(self, img):
        self.get_logger().info("Received color image via UDP")
        
        # 이미지를 처리하는 로직을 추가할 수 있습니다
        self.current_color_image = img
        self.display_combined_view()

    def depth_image_callback(self, msg):
        self.get_logger().info("Received depth image via UDP")
        
        # 수신한 데이터를 numpy 배열로 변환
        np_data = np.frombuffer(msg, dtype=np.uint16)
        depth_image = np_data.reshape((480, 640))  # 예시로 480x640의 해상도를 가정

        # 깊이 이미지를 적절한 데이터 유형(float32)으로 변환 후 GPU로 이동
        depth_image_float = depth_image.astype(np.float32)
        depth_tensor = torch.tensor(depth_image_float, device=self.device)

        # 가장 가까운 지점을 찾기 위한 GPU 계산
        min_val = depth_tensor.min().item()
        min_idx = depth_tensor.argmin().item()

        # 2D 좌표로 변환
        height, width = depth_tensor.shape
        min_idx_y, min_idx_x = divmod(min_idx, width)
        self.closest_point = (min_idx_x, min_idx_y)
        self.distance = min_val

        # OpenCV로 다시 변환을 위해 CPU로 이동
        self.current_depth_image = depth_tensor.cpu().numpy()
        self.display_combined_view()

    def display_combined_view(self):
        if self.current_color_image is not None:
            color_image = self.current_color_image.copy()

            if self.bbox is not None:
                # 바운딩 박스 그리기
                x1, y1, x2, y2 = self.bbox
                cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if self.closest_point is not None:
                # 가장 가까운 지점 표시
                cv2.circle(color_image, self.closest_point, 5, (0, 0, 255), -1)
                # 텍스트로 ID와 거리 표시
                cv2.putText(color_image, f'ID: {self.id}, Distance: {self.distance / 1000:.2f}m', 
                            (self.closest_point[0] + 10, self.closest_point[1] + 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 컬러 이미지를 창에 표시
            cv2.imshow("Color Image with Tracking", color_image)
            cv2.waitKey(1)

        if self.current_depth_image is not None:
            # 깊이 이미지를 표시하기 위해 시각적 변환
            depth_display = cv2.convertScaleAbs(self.current_depth_image, alpha=0.03)
            depth_colored = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)

            if self.bbox is not None:
                # 바운딩 박스 그리기
                x1, y1, x2, y2 = self.bbox
                cv2.rectangle(depth_colored, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if self.closest_point is not None:
                # 가장 가까운 점을 원으로 표시
                cv2.circle(depth_colored, self.closest_point, 5, (0, 0, 255), -1)
                # 텍스트로 거리 표시
                cv2.putText(depth_colored, f'Distance: {self.distance / 1000:.2f}m', 
                            (self.closest_point[0] + 10, self.closest_point[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 깊이 이미지를 창에 표시
            cv2.imshow("Depth Image with Tracking", depth_colored)
            cv2.waitKey(1)

    def publish_id_distance_info(self, id, bbox_center, depth_value):
        self.get_logger().info(f"Preparing to publish IdDistanceInfo: ID={id}, Bbox Center={bbox_center}, Depth={depth_value}")
        
        msg = IdDistanceInfo()
        msg.id = id
        msg.closest_point = Point(x=float(bbox_center[0]), y=float(bbox_center[1]), z=0.0)
        msg.distance = float(depth_value)
        
        # 발행 전 메시지 정보 로그
        self.get_logger().info(f"IdDistanceInfo message prepared: {msg}")
        
        self.publisher_id_distance.publish(msg)
        self.get_logger().info(f"Published IdDistanceInfo: ID={id}, Bbox Center={bbox_center}, Depth={depth_value}")

def main(args=None):
    rclpy.init(args=args)
    node = UDPVISUALTESTNODE()

    while rclpy.ok():
        node.udp_receive()
        rclpy.spin_once(node)

    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
