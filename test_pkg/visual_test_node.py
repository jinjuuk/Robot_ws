# import rclpy
# from rclpy.node import Node
# from message.msg import IdDistanceInfo
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
# import cv2
# import numpy as np

# class CombinedVisualizationNode(Node):
#     def __init__(self):
#         super().__init__('combined_visualization_node')
        
#         # 구독자 설정
#         self.subscription_id_distance = self.create_subscription(
#             IdDistanceInfo,
#             'id_distance_info',
#             self.id_distance_callback,
#             10
#         )
#         self.subscription_color_image = self.create_subscription(
#             Image,
#             '/camera/color/image_raw',
#             self.color_image_callback,
#             10
#         )
#         self.subscription_depth_image = self.create_subscription(
#             Image,
#             '/camera/depth/image_raw',
#             self.depth_image_callback,
#             10
#         )

#         # OpenCV를 위한 CVBridge 초기화
#         self.bridge = CvBridge()

#         # 이미지 저장을 위한 변수
#         self.current_color_image = None
#         self.current_depth_image = None
#         self.closest_point = None
#         self.distance = None
#         self.id = None

#     def id_distance_callback(self, msg):
#         self.id = msg.id
#         self.closest_point = (int(msg.closest_point.x), int(msg.closest_point.y))
#         self.distance = msg.distance
#         self.get_logger().info(f'Received IdDistanceInfo: ID={self.id}, Closest Point={self.closest_point}, Distance={self.distance}')

#     def color_image_callback(self, msg):
#         # 컬러 이미지를 OpenCV 형식으로 변환
#         self.current_color_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
#         self.display_combined_view()

#     def depth_image_callback(self, msg):
#         # 깊이 이미지를 OpenCV 형식으로 변환
#         self.current_depth_image = self.bridge.imgmsg_to_cv2(msg, '16UC1')
#         self.display_combined_view()

#     def display_combined_view(self):
#         if self.current_color_image is not None:
#             color_image = self.current_color_image.copy()

#             if self.closest_point is not None:
#                 # 가장 가까운 지점 표시
#                 cv2.circle(color_image, self.closest_point, 5, (0, 0, 255), -1)
#                 # 텍스트로 ID와 거리 표시
#                 cv2.putText(color_image, f'ID: {self.id}, Distance: {self.distance:.2f}m', 
#                             (self.closest_point[0] + 10, self.closest_point[1] + 10), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
#             # 컬러 이미지를 창에 표시
#             cv2.imshow("Color Image with Tracking", color_image)
#             cv2.waitKey(1)

#         if self.current_depth_image is not None:
#             # 깊이 이미지를 표시하기 위해 시각적 변환
#             depth_display = cv2.convertScaleAbs(self.current_depth_image, alpha=0.03)
#             depth_colored = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)

#             if self.closest_point is not None:
#                 # 가장 가까운 점을 원으로 표시
#                 cv2.circle(depth_colored, self.closest_point, 5, (0, 0, 255), -1)
#                 # 텍스트로 거리 표시
#                 cv2.putText(depth_colored, f'Distance: {self.distance:.2f}m', 
#                             (self.closest_point[0] + 10, self.closest_point[1] - 10), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

#             # 깊이 이미지를 창에 표시
#             cv2.imshow("Depth Image with Tracking", depth_colored)
#             cv2.waitKey(1)

# def main(args=None):
#     rclpy.init(args=args)
#     node = CombinedVisualizationNode()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()
#     cv2.destroyAllWindows()

# if __name__ == '__main__':
#     main()





# import rclpy
# from rclpy.node import Node
# from message.msg import IdDistanceInfo
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
# import cv2
# import numpy as np

# class CombinedVisualizationNode(Node):
#     def __init__(self):
#         super().__init__('combined_visualization_node')
        
#         # 구독자 설정
#         self.subscription_id_distance = self.create_subscription(
#             IdDistanceInfo,
#             'id_distance_info',
#             self.id_distance_callback,
#             10
#         )
#         self.subscription_color_image = self.create_subscription(
#             Image,
#             '/camera/color/image_raw',
#             self.color_image_callback,
#             10
#         )
#         self.subscription_depth_image = self.create_subscription(
#             Image,
#             '/camera/depth/image_raw',
#             self.depth_image_callback,
#             10
#         )

#         # OpenCV를 위한 CVBridge 초기화
#         self.bridge = CvBridge()

#         # 이미지 저장을 위한 변수
#         self.current_color_image = None
#         self.current_depth_image = None
#         self.closest_point = None
#         self.distance = None
#         self.id = None
#         self.bbox = None

#     def id_distance_callback(self, msg):
#         self.id = msg.id
#         self.closest_point = (int(msg.closest_point.x), int(msg.closest_point.y))
#         self.distance = msg.distance
#         self.get_logger().info(f'Received IdDistanceInfo: ID={self.id}, Closest Point={self.closest_point}, Distance={self.distance}')

#     def color_image_callback(self, msg):
#         # 컬러 이미지를 OpenCV 형식으로 변환
#         self.current_color_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
#         self.display_combined_view()

#     def depth_image_callback(self, msg):
#         # 깊이 이미지를 OpenCV 형식으로 변환
#         self.current_depth_image = self.bridge.imgmsg_to_cv2(msg, '16UC1')
#         self.display_combined_view()

#     def display_combined_view(self):
#         if self.current_color_image is not None:
#             color_image = self.current_color_image.copy()

#             if self.bbox is not None:
#                 # 바운딩 박스 그리기
#                 x1, y1, x2, y2 = self.bbox
#                 cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

#             if self.closest_point is not None:
#                 # 가장 가까운 지점 표시
#                 cv2.circle(color_image, self.closest_point, 5, (0, 0, 255), -1)
#                 # 텍스트로 ID와 거리 표시
#                 cv2.putText(color_image, f'ID: {self.id}, Distance: {self.distance:.2f}m', 
#                             (self.closest_point[0] + 10, self.closest_point[1] + 10), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
#             # 컬러 이미지를 창에 표시
#             cv2.imshow("Color Image with Tracking", color_image)
#             cv2.waitKey(1)

#         if self.current_depth_image is not None:
#             # 깊이 이미지를 표시하기 위해 시각적 변환
#             depth_display = cv2.convertScaleAbs(self.current_depth_image, alpha=0.03)
#             depth_colored = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)

#             if self.bbox is not None:
#                 # 바운딩 박스 그리기
#                 x1, y1, x2, y2 = self.bbox
#                 cv2.rectangle(depth_colored, (x1, y1), (x2, y2), (0, 255, 0), 2)

#             if self.closest_point is not None:
#                 # 가장 가까운 점을 원으로 표시
#                 cv2.circle(depth_colored, self.closest_point, 5, (0, 0, 255), -1)
#                 # 텍스트로 거리 표시
#                 cv2.putText(depth_colored, f'Distance: {self.distance:.2f}m', 
#                             (self.closest_point[0] + 10, self.closest_point[1] - 10), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

#             # 깊이 이미지를 창에 표시
#             cv2.imshow("Depth Image with Tracking", depth_colored)
#             cv2.waitKey(1)

# def main(args=None):
#     rclpy.init(args=args)
#     node = CombinedVisualizationNode()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()
#     cv2.destroyAllWindows()

# if __name__ == '__main__':
#     main()







# import torch
# import rclpy
# from rclpy.node import Node
# from message.msg import IdDistanceInfo
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
# import cv2
# import numpy as np

# class CombinedVisualizationNode(Node):
#     def __init__(self):
#         super().__init__('combined_visualization_node')
        
#         # CUDA 장치 설정
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         if torch.cuda.is_available():
#             cuda_device_name = torch.cuda.get_device_name(0)
#             self.get_logger().info(f"CUDA is available. Using GPU: {cuda_device_name}")
#         else:
#             self.get_logger().info("CUDA is not available. Using CPU")

#         # 구독자 설정
#         self.subscription_id_distance = self.create_subscription(
#             IdDistanceInfo,
#             'id_distance_info',
#             self.id_distance_callback,
#             10
#         )
#         self.subscription_color_image = self.create_subscription(
#             Image,
#             '/camera/color/image_raw',
#             self.color_image_callback,
#             10
#         )
#         self.subscription_depth_image = self.create_subscription(
#             Image,
#             '/camera/depth/image_raw',
#             self.depth_image_callback,
#             10
#         )

#         # OpenCV를 위한 CVBridge 초기화
#         self.bridge = CvBridge()

#         # 이미지 저장을 위한 변수
#         self.current_color_image = None
#         self.current_depth_image = None
#         self.closest_point = None
#         self.distance = None
#         self.id = None
#         self.bbox = None

#     def id_distance_callback(self, msg):
#         self.id = msg.id
#         self.closest_point = (int(msg.closest_point.x), int(msg.closest_point.y))
#         self.distance = msg.distance
#         self.get_logger().info(f'Received IdDistanceInfo: ID={self.id}, Closest Point={self.closest_point}, Distance={self.distance}')

#     def color_image_callback(self, msg):
#         # 컬러 이미지를 OpenCV 형식으로 변환
#         self.current_color_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
#         self.display_combined_view()

#     def depth_image_callback(self, msg):
#         # 깊이 이미지를 OpenCV 형식으로 변환
#         depth_image = self.bridge.imgmsg_to_cv2(msg, '16UC1')

#         # 깊이 이미지를 적절한 데이터 유형(float32)으로 변환
#         depth_image_float = depth_image.astype(np.float32)

#         # 깊이 이미지를 GPU로 이동
#         depth_tensor = torch.tensor(depth_image_float, device=self.device)

#         # 가장 가까운 지점을 찾기 위한 GPU 계산
#         min_val = torch.min(depth_tensor).item()
#         min_idx = torch.argmin(depth_tensor).item()

#         # 2D 좌표로 변환
#         min_idx_y, min_idx_x = divmod(min_idx, depth_tensor.size(1))
#         self.closest_point = (min_idx_x, min_idx_y)
#         self.distance = min_val

#         # OpenCV로 다시 변환을 위해 CPU로 이동
#         self.current_depth_image = depth_tensor.cpu().numpy()
#         self.display_combined_view()

#     def display_combined_view(self):
#         if self.current_color_image is not None:
#             color_image = self.current_color_image.copy()

#             if self.bbox is not None:
#                 # 바운딩 박스 그리기
#                 x1, y1, x2, y2 = self.bbox
#                 cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

#             if self.closest_point is not None:
#                 # 가장 가까운 지점 표시
#                 cv2.circle(color_image, self.closest_point, 5, (0, 0, 255), -1)
#                 # 텍스트로 ID와 거리 표시
#                 cv2.putText(color_image, f'ID: {self.id}, Distance: {self.distance / 1000:.2f}m', 
#                             (self.closest_point[0] + 10, self.closest_point[1] + 10), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
#             # 컬러 이미지를 창에 표시
#             cv2.imshow("Color Image with Tracking", color_image)
#             cv2.waitKey(1)

#         if self.current_depth_image is not None:
#             # 깊이 이미지를 표시하기 위해 시각적 변환
#             depth_display = cv2.convertScaleAbs(self.current_depth_image, alpha=0.03)
#             depth_colored = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)

#             if self.bbox is not None:
#                 # 바운딩 박스 그리기
#                 x1, y1, x2, y2 = self.bbox
#                 cv2.rectangle(depth_colored, (x1, y1), (x2, y2), (0, 255, 0), 2)

#             if self.closest_point is not None:
#                 # 가장 가까운 점을 원으로 표시
#                 cv2.circle(depth_colored, self.closest_point, 5, (0, 0, 255), -1)
#                 # 텍스트로 거리 표시
#                 cv2.putText(depth_colored, f'Distance: {self.distance / 1000:.2f}m', 
#                             (self.closest_point[0] + 10, self.closest_point[1] - 10), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

#             # 깊이 이미지를 창에 표시
#             cv2.imshow("Depth Image with Tracking", depth_colored)
#             cv2.waitKey(1)

# def main(args=None):
#     rclpy.init(args=args)
#     node = CombinedVisualizationNode()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()
#     cv2.destroyAllWindows()

# if __name__ == '__main__':
#     main()






# import torch
# import rclpy
# from rclpy.node import Node
# from message.msg import IdDistanceInfo
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
# import cv2
# import numpy as np

# class CombinedVisualizationNode(Node):
#     def __init__(self):
#         super().__init__('combined_visualization_node')
        
#         # CUDA 장치 설정
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         if torch.cuda.is_available():
#             cuda_device_name = torch.cuda.get_device_name(0)
#             self.get_logger().info(f"CUDA is available. Using GPU: {cuda_device_name}")
#         else:
#             self.get_logger().info("CUDA is not available. Using CPU")

#         # 구독자 설정
#         self.subscription_id_distance = self.create_subscription(
#             IdDistanceInfo,
#             'id_distance_info',
#             self.id_distance_callback,
#             10
#         )
#         self.subscription_color_image = self.create_subscription(
#             Image,
#             '/camera/color/image_raw',
#             self.color_image_callback,
#             10
#         )
#         self.subscription_depth_image = self.create_subscription(
#             Image,
#             '/camera/depth/image_raw',
#             self.depth_image_callback,
#             10
#         )

#         # OpenCV를 위한 CVBridge 초기화
#         self.bridge = CvBridge()

#         # 이미지 저장을 위한 변수
#         self.current_color_image = None
#         self.current_depth_image = None
#         self.closest_point = None
#         self.distance = None
#         self.id = None
#         self.bbox = None

#     def id_distance_callback(self, msg):
#         self.id = msg.id
#         self.closest_point = (int(msg.closest_point.x), int(msg.closest_point.y))
#         self.distance = msg.distance
#         self.get_logger().info(f'Received IdDistanceInfo: ID={self.id}, Closest Point={self.closest_point}, Distance={self.distance}')

#     def color_image_callback(self, msg):
#         # 컬러 이미지를 OpenCV 형식으로 변환
#         self.current_color_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
#         self.display_combined_view()

#     def depth_image_callback(self, msg):
#         # 깊이 이미지를 OpenCV 형식으로 변환
#         depth_image = self.bridge.imgmsg_to_cv2(msg, '16UC1')

#         # 깊이 이미지를 적절한 데이터 유형(float32)으로 변환 후 GPU로 이동
#         depth_image_float = depth_image.astype(np.float32)
#         depth_tensor = torch.tensor(depth_image_float, device=self.device)

#         # 가장 가까운 지점을 찾기 위한 GPU 계산
#         min_val = depth_tensor.min().item()
#         min_idx = depth_tensor.argmin().item()

#         # 2D 좌표로 변환
#         height, width = depth_tensor.shape
#         min_idx_y, min_idx_x = divmod(min_idx, width)
#         self.closest_point = (min_idx_x, min_idx_y)
#         self.distance = min_val

#         # OpenCV로 다시 변환을 위해 CPU로 이동
#         self.current_depth_image = depth_tensor.cpu().numpy()
#         self.display_combined_view()

#     def display_combined_view(self):
#         if self.current_color_image is not None:
#             color_image = self.current_color_image.copy()

#             if self.bbox is not None:
#                 # 바운딩 박스 그리기
#                 x1, y1, x2, y2 = self.bbox
#                 cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

#             if self.closest_point is not None:
#                 # 가장 가까운 지점 표시
#                 cv2.circle(color_image, self.closest_point, 5, (0, 0, 255), -1)
#                 # 텍스트로 ID와 거리 표시
#                 cv2.putText(color_image, f'ID: {self.id}, Distance: {self.distance / 1000:.2f}m', 
#                             (self.closest_point[0] + 10, self.closest_point[1] + 10), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
#             # 컬러 이미지를 창에 표시
#             cv2.imshow("Color Image with Tracking", color_image)
#             cv2.waitKey(1)

#         if self.current_depth_image is not None:
#             # 깊이 이미지를 표시하기 위해 시각적 변환
#             depth_display = cv2.convertScaleAbs(self.current_depth_image, alpha=0.03)
#             depth_colored = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)

#             if self.bbox is not None:
#                 # 바운딩 박스 그리기
#                 x1, y1, x2, y2 = self.bbox
#                 cv2.rectangle(depth_colored, (x1, y1), (x2, y2), (0, 255, 0), 2)

#             if self.closest_point is not None:
#                 # 가장 가까운 점을 원으로 표시
#                 cv2.circle(depth_colored, self.closest_point, 5, (0, 0, 255), -1)
#                 # 텍스트로 거리 표시
#                 cv2.putText(depth_colored, f'Distance: {self.distance / 1000:.2f}m', 
#                             (self.closest_point[0] + 10, self.closest_point[1] - 10), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

#             # 깊이 이미지를 창에 표시
#             cv2.imshow("Depth Image with Tracking", depth_colored)
#             cv2.waitKey(1)

# def main(args=None):
#     rclpy.init(args=args)
#     node = CombinedVisualizationNode()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()
#     cv2.destroyAllWindows()

# if __name__ == '__main__':
#     main()






import torch
import rclpy
from rclpy.node import Node
from message.msg import IdDistanceInfo
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class CombinedVisualizationNode(Node):
    def __init__(self):
        super().__init__('combined_visualization_node')
        
        # CUDA 장치 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            cuda_device_name = torch.cuda.get_device_name(0)
            self.get_logger().info(f"CUDA is available. Using GPU: {cuda_device_name}")
        else:
            self.get_logger().info("CUDA is not available. Using CPU")

        # 구독자 설정
        self.subscription_id_distance = self.create_subscription(
            IdDistanceInfo,
            'id_distance_info',
            self.id_distance_callback,
            10
        )
        self.subscription_color_image = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.color_image_callback,
            10
        )
        self.subscription_depth_image = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.depth_image_callback,
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

    def id_distance_callback(self, msg):
        self.id = msg.id
        self.closest_point = (int(msg.bbox_center.x), int(msg.bbox_center.y))
        self.distance = msg.distance
        self.bbox = self.calculate_bbox_from_center(self.closest_point, msg.width, msg.height)
        self.get_logger().info(f'Received IdDistanceInfo: ID={self.id}, Closest Point={self.closest_point}, Width={msg.width}, Height={msg.height}, Distance={self.distance}')

    def calculate_bbox_from_center(self, center, width, height):
        """중심 좌표와 크기로부터 바운딩 박스 좌표 계산."""
        x_center, y_center = center
        half_width, half_height = width / 2, height / 2
        x1 = int(x_center - half_width)
        y1 = int(y_center - half_height)
        x2 = int(x_center + half_width)
        y2 = int(y_center + half_height)
        return (x1, y1, x2, y2)

    def color_image_callback(self, msg):
        # 컬러 이미지를 OpenCV 형식으로 변환
        self.current_color_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.display_combined_view()

    def depth_image_callback(self, msg):
        # 깊이 이미지를 OpenCV 형식으로 변환
        depth_image = self.bridge.imgmsg_to_cv2(msg, '16UC1')

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

def main(args=None):
    rclpy.init(args=args)
    node = CombinedVisualizationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
