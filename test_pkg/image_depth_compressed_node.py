import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import cv2



class ImagePublisher(Node):
    def __init__(self):
        super().__init__('image_publisher')

        self.color_subscriber = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.color_image_callback,
            10
        )

    
        self.color_publisher = self.create_publisher(
            CompressedImage,
            '/camera/color/image_raw/compressed',
            10
        )

        
        self.depth_subscriber = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.depth_image_callback,
            10
        )

        
        self.depth_publisher = self.create_publisher(
            CompressedImage,
            '/camera/depth/image_raw/compressed',
            10
        )

        self.bridge = CvBridge()

    def color_image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'CvBridge Error: {e}')
            return

        # 이미지 리사이즈
        resized_image = cv2.resize(cv_image, (320, 320)) #(320, 240)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
        _, encoded_img = cv2.imencode('.jpg', resized_image, encode_param)
        compressed_img_msg = CompressedImage()
        compressed_img_msg.header = msg.header
        compressed_img_msg.format = 'jpeg'
        compressed_img_msg.data = encoded_img.tobytes()

        self.color_publisher.publish(compressed_img_msg)
        self.get_logger().info('Compressed and resized color image published.')


    def depth_image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
        except Exception as e:
            self.get_logger().error(f'CvBridge Error: {e}')
            return

        # 이미지 리사이즈
        resized_image = cv2.resize(cv_image, (320, 320)) #(320, 240)
        encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
        _, encoded_img = cv2.imencode('.png', resized_image, encode_param)
        compressed_img_msg = CompressedImage()
        compressed_img_msg.header = msg.header
        compressed_img_msg.format = 'png'
        compressed_img_msg.data = encoded_img.tobytes()

        self.depth_publisher.publish(compressed_img_msg)
        self.get_logger().info('Compressed and resized depth image published.')


def main(args=None):
    rclpy.init(args=args)
    node = ImagePublisher()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
