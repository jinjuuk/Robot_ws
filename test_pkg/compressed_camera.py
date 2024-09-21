import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import cv2



class ImagePublisher(Node):

    def __init__(self):

        super().__init__('image_publisher')

        

        # Subscriber for the raw image

        self.subscriber = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            10
        )

        

        # Publisher for the compressed image
        self.publisher = self.create_publisher(
            CompressedImage,
            '/camera/color/image_raw/compressed',
            10
        )

        

        # Create a CvBridge instance
        self.bridge = CvBridge()



    def image_callback(self, msg):
        # Convert the ROS Image message to OpenCV format
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'CvBridge Error: {e}')
            return

        

        # Convert the image to grayscale

        # gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        

        # Encode image as JPEG

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]  # Reduce quality for better compression
        self.get_logger().info(f'encode_param: {encode_param}')

        _, encoded_img = cv2.imencode('.jpg', cv_image, encode_param)
        self.get_logger().info(f'_: {_}, encoded_img: {encoded_img}')
        

        # Convert encoded image to CompressedImage message

        compressed_img_msg = CompressedImage()
        

        compressed_img_msg.header = msg.header
        self.get_logger().info(f'compressed_img_msg.header: {compressed_img_msg.header}')

        compressed_img_msg.format = 'jpeg'
        self.get_logger().info(f'compressed_img_msg.format: {compressed_img_msg.format}')

        compressed_img_msg.data = encoded_img.tobytes()
        self.get_logger().info(f'compressed_img_msg.data: {compressed_img_msg.data}')
        

        # Publish the compressed image

        self.publisher.publish(compressed_img_msg)

        # self.get_logger().info('Published compressed image')



def main(args=None):

    rclpy.init(args=args)

    node = ImagePublisher()

    rclpy.spin(node)

    rclpy.shutdown()



if __name__ == '__main__':

    main()