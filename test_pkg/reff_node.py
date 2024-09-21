def force_drive(self, linear_x=0.0, linear_y=0.0, linear_z=0.0, angular_x=0.0, angular_y=0.0, angular_z=0.0, duration=1.0, stop_distance=0.5):

        """ 

        강제로 로봇을 특정 방향으로 주행시키는 함수 

        직진 linear_x > 0 , 후진 linear_x < 0 , 오른쪽 회전 angular.z < 0, 왼쪽 회전 angular.z > 0 나머지 값들은 필요없음

        duration은 몇초동안 할지, stop_distance depth camera의 어느정도 까지의 거리까지 갈것인가에대한 것

        """

        twist_msg = Twist()

        twist_msg.linear.x = linear_x

        twist_msg.linear.y = linear_y

        twist_msg.linear.z = linear_z



        twist_msg.angular.x = angular_x

        twist_msg.angular.y = angular_y

        twist_msg.angular.z = angular_z



        end_time = self.get_clock().now().seconds_nanoseconds()[0] + duration

        while self.get_clock().now().seconds_nanoseconds()[0] < end_time:

            if self.closest_obstacle_distance < stop_distance:

                self.get_logger().info(f'Obstacle detected within {self.closest_obstacle_distance:.2f} meters. Stopping force drive.')

                break  # 장애물이 감지되면 주행 중지



            self.cmd_vel_publisher.publish(twist_msg)

            time.sleep(0.1)



        # 주행 종료를 위해 속도를 0으로 설정

        twist_msg.linear.x = 0.0

        twist_msg.linear.y = 0.0

        twist_msg.linear.z = 0.0



        twist_msg.angular.x = 0.0

        twist_msg.angular.y = 0.0

        twist_msg.angular.z = 0.0



        self.cmd_vel_publisher.publish(twist_msg)