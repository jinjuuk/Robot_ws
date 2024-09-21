import rclpy

from rclpy.node import Node

from sensor_msgs.msg import Image

from cv_bridge import CvBridge

import cv2

import torch

from deep_sort_realtime.deepsort_tracker import DeepSort





class ImageSubscriber(Node):

    def __init__(self):

        super().__init__('image_subscriber')

        self.subscription = self.create_subscription(

            Image,

            '/camera/color/image_raw',

            self.listener_callback,

            10)

        

        self.bridge = CvBridge()

        self.should_stop = False



        # YOLOv5 모델 로드

        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/jinjuuk/dev_ws/pt_files/yolov5n.pt')

        self.model.eval()

         

        # 객체 추적 트래킹 선언 

        self.tracker = DeepSort(max_age=50, n_init=3, nn_budget=70) 

        

    def listener_callback(self, msg):

        if self.should_stop:

            return



        # ROS 2 이미지 메시지를 OpenCV 이미지로 변환

        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        # 이미지를 화면에 표시   



        # YOLOv5 모델로 객체 인식

        results = self.model(cv_image)

        trackers = []



        for det in results.xyxy[0]:  # 각 객체에 대해 반복

            x_min, y_min, x_max, y_max, conf, cls = det

            if int(cls) == 0:  # 클래스 ID가 0인 경우 (사람인 경우)

                

                # 텐서를 CPU로 이동시키고, NumPy 배열로 변환

                x_min = x_min.cpu().item()

                y_min = y_min.cpu().item()

                x_max = x_max.cpu().item()

                y_max = y_max.cpu().item()

                conf = conf.cpu().item()



                trackers.append([[x_min, y_min, x_max-x_min, y_max-y_min], conf, cls])

        

        tracks = self.tracker.update_tracks(trackers, frame=cv_image)

        

        for track in tracks:



            if not track.is_confirmed() or track.time_since_update > 1:

                continue

                

            track_id = track.track_id

            ltrb = track.to_ltrb()  # 바운딩 박스 좌표 [left, top, right, bottom]

            x_min, y_min, x_max, y_max = map(int, ltrb)

            

            # 추적된 사람 바운딩 박스 그리기

            cv2.rectangle(cv_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

            cv2.putText(cv_image, f"ID: {track_id}", (x_max, y_max - 10),

                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            

            # 중심점 계산 및 표시

            center_x = int((x_min + x_max) / 2)

            center_y = int((y_min + y_max) / 2)

            cv2.circle(cv_image, (center_x, center_y), 5, (0, 255, 0), -1)

            cv2.putText(cv_image, f"Center: ({center_x}, {center_y})", (center_x - 50, center_y - 10),

                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)



        cv2.imshow("cv_image", cv_image) 

        

        # 'q' 키를 누르면 루프 탈출

        if cv2.waitKey(1) & 0xFF == ord('q'):

            self.should_stop = True

            self.get_logger().info("카메라 종료")

            cv2.destroyAllWindows()

            rclpy.shutdown()



def main(args=None):

    rclpy.init(args=args)

    image_subscriber = ImageSubscriber()

    rclpy.spin(image_subscriber)

    rclpy.shutdown()