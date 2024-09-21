import time
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import CompressedImage

# class ImageNode(Node):
#     def __init__(self):
#         super().__init__('image_node')

#         self.image_sub = self.create_subscription(
#             CompressedImage, 
#             'img_processed', 
#             self.image_callback, 
#             10)

#         self.frame_count = 0
#         self.last_time = time.time()

#     def image_callback(self, msg):
#         # 여기에 이미지 처리 코드를 추가할 수 있습니다.
        
#         # 프레임 처리 및 FPS 계산
#         self.frame_count += 1
#         current_time = time.time()
#         elapsed_time = current_time - self.last_time

#         if elapsed_time >= 1.0:  # 매 1초마다 FPS를 계산합니다.
#             fps = self.frame_count / elapsed_time
#             self.get_logger().info(f"FPS: {fps:.2f}")

#             # 카운터 초기화
#             self.last_time = current_time
#             self.frame_count = 0

# def main(args=None):
#     rclpy.init(args=args)
#     node = ImageNode()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()



import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ImageNode(Node):
    def __init__(self):
        super().__init__('image_node')

        self.image_sub = self.create_subscription(
            Image, 
            '/camera/color/image_raw', 
            self.image_callback, 
            10)
            #img_processed
        self.bridge = CvBridge()
        self.frame_count = 0
        self.last_time = time.time()

    def image_callback(self, msg):
        # 이미지 메시지를 OpenCV 이미지로 변환
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='8UC3')
        
        # OpenCV 창에 실시간 영상 표시
        cv2.imshow("Camera", cv_image)
        cv2.waitKey(1)  # 1ms 대기, 영상 업데이트

        # 프레임 처리 및 FPS 계산
        self.frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - self.last_time

        if (elapsed_time >= 1.0):  # 매 1초마다 FPS를 계산합니다.
            fps = self.frame_count / elapsed_time
            self.get_logger().info(f"FPS: {fps:.2f}")

            # 카운터 초기화
            self.last_time = current_time
            self.frame_count = 0

def main(args=None):
    rclpy.init(args=args)
    node = ImageNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()  # OpenCV 창 닫기

if __name__ == '__main__':
    main()
