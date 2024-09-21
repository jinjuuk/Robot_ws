import rclpy as rp
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import numpy as np

class DepthSubscriber(Node):
    
    def __init__(self):
        super().__init__('depth_subscriber')
        
        self.subscriber = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.depth_callback,
            10
        )
        
        self.bridge = CvBridge()
        
    def depth_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, '16UC1')
            
            self.get_logger().info(f"mid : {cv_image[200][320]}")
            
            # # 이미지 반전(가까운 물체가 어둡게)
            # max_value = np.max(cv_image)
            # inverted_image = max_value - cv_image
            
            # 정규화
            normalized_cv_image = cv2.normalize(
                cv_image, 
                None, 
                alpha=0, 
                beta=65535, 
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_16UC1
            )
            
            # 노이즈 제거
            denoised_image = cv2.medianBlur(normalized_cv_image, 5)
            
            cv2.imshow("Received Video", denoised_image)
            cv2.waitKey(1)
                
        except Exception as e:
            self.get_logger().error(f'Error: {e}')
            return
        

def main(args=None):
    rp.init(args=args)
    depth_subscriber = DepthSubscriber()
    rp.spin(depth_subscriber)
    depth_subscriber.destroy_node()
    rp.shutdown()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()