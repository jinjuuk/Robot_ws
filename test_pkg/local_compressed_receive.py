import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, CameraInfo , CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np
from sensor_msgs_py import point_cloud2
import torch
import time



class ImageNode(Node):
    def __init__(self):
        super().__init__('image_node')

        self.image_sub = self.create_subscription(
            CompressedImage, 
            '/camera/color/image_raw/compressed',
            self.image_callback,
            10)

        

        self.bridge = CvBridge()
        self.latest_image = None
        self.last_time = time.time()
        self.frame_count = 0


        
    def image_callback(self, msg):
        # Convert the compressed image data to an OpenCV image
        np_arr = np.frombuffer(msg.data, np.uint8)
        self.latest_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        # self.get_logger().info(f'토픽 {msg}')
        

        if self.latest_image is not None:
            # Display the image with overlay
            cv2.imshow("Image", self.latest_image)
            cv2.waitKey(1)



            # Calculate FPS
            self.frame_count += 1
            current_time = time.time()
            elapsed_time = current_time - self.last_time



            if elapsed_time >= 1.0:  # Every second
                fps = self.frame_count / elapsed_time
                self.get_logger().info(f"FPS: {fps:.2f}")

                

                # Reset counters
                self.last_time = current_time
                self.frame_count = 0



def main(args=None):
    rclpy.init(args=args)
    node = ImageNode()
    rclpy.spin(node)
    rclpy.shutdown()



if __name__ == '__main__':
    main()
