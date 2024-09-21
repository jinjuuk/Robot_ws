import rclpy as rp
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point, Twist
from std_msgs.msg import String
import cv2
from cv_bridge import CvBridge
import numpy as np

class DepthConversionNode(Node):
    
    def __init__(self):
        super().__init__('depth_conversion_node')
        
        # 구독자 설정
        self.subscriber_depth_camera = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.depth_callback,
            10
        )
        
        self.subscriber_cp = self.create_subscription(
            Point, 
            'object_tracking/centre_point', 
            self.centre_point_callback,
            10
        )
        
        self.subscriber_id = self.create_subscription(
            String, 
            'object_tracking/id', 
            self.id_callback,
            10
        )
        
        # 발행자 설정
        self.publisher_cmd_vel = self.create_publisher(
            Twist, 
            'Twist/cmd_vel', 
            10
        )
        
        self.publisher_cp_id_depth = self.create_publisher(
            String, 
            'cp+id+depth', 
            10
        )
        
        self.bridge = CvBridge()
        
        # 토픽에서 받은 최신 값을 저장할 변수들
        self.latest_centre_point = None
        self.latest_id = None
        self.latest_depth_image = None

    def depth_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, '16UC1')
            self.latest_depth_image = cv_image
            self.process_and_publish()

        except Exception as e:
            self.get_logger().error(f'Error in depth_callback: {e}')

    def centre_point_callback(self, msg):
        self.latest_centre_point = (int(msg.x), int(msg.y))
        self.process_and_publish()

    def id_callback(self, msg):
        self.latest_id = msg.data
        self.process_and_publish()

    def process_and_publish(self):
        if self.latest_centre_point is not None and self.latest_id is not None and self.latest_depth_image is not None:
            x, y = self.latest_centre_point
            if 0 <= x < self.latest_depth_image.shape[1] and 0 <= y < self.latest_depth_image.shape[0]:
                depth_value = self.latest_depth_image[y, x]
                message = f"cp:({x},{y}) id:{self.latest_id} depth:{depth_value}"
                self.publisher_cp_id_depth.publish(String(data=message))
                self.get_logger().info(f'Published: {message}')
                
                # Twist 메시지를 생성하고 발행하는 예시
                twist = Twist()
                twist.linear.x = 0.1  # 예시 값, 필요에 따라 조정
                twist.angular.z = 0.0  # 예시 값, 필요에 따라 조정
                self.publisher_cmd_vel.publish(twist)

def main(args=None):
    rp.init(args=args)
    node = DepthConversionNode()
    rp.spin(node)
    node.destroy_node()
    rp.shutdown()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()
