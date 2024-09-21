import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from message.msg import IdDistanceInfo

class TestSubscriberNode(Node):

    def __init__(self):
        super().__init__('test_subscriber_node')
        
        # Twist/cmd_vel 토픽을 구독
        self.subscription_cmd_vel = self.create_subscription(
            Twist,
            'cmd_vel',
            self.cmd_vel_callback,
            10
        )

        # id_distance_info 토픽을 구독
        self.subscription_id_distance = self.create_subscription(
            IdDistanceInfo,
            'id_distance_info',
            self.id_distance_callback,
            10
        )

        self.get_logger().info("TestSubscriberNode initialized and ready to receive messages.")

    def cmd_vel_callback(self, msg):
        # Twist 메시지 수신 시 로그 출력
        linear = msg.linear
        angular = msg.angular
        self.get_logger().info(
            f'Received Twist message: Linear={linear.x}, {linear.y}, {linear.z}; Angular={angular.x}, {angular.y}, {angular.z}'
        )

    def id_distance_callback(self, msg):
        # IdDistanceInfo 메시지 수신 시 로그 출력
        id = msg.id
        bbox_center = msg.bbox_center
        distance = msg.distance
        width = msg.width
        height = msg.height
        self.get_logger().info(
            f'Received IdDistanceInfo message: ID={id}, Bbox Center=({bbox_center.x}, {bbox_center.y}, {bbox_center.z}), '
            f'Width={width}, Height={height}, Distance={distance}cm'
        )


def main(args=None):
    rclpy.init(args=args)
    node = TestSubscriberNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
