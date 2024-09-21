# import rclpy as rp
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# import cv2 
# from cv_bridge import CvBridge
# import torch
# import numpy as np
# import time

# class Test_node(Node):
#     def __init__(self):
#         super().__init__('test_node')
#         self.subscriber_ = self.create_subscription(Image, '/camera/color/image_raw', self.image_callback, 10)
#         self.br = CvBridge()
#         self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/jinjuuk/dev_ws/pt_files/yolov5n.pt')
#         self.person_class_id = self.get_person_class_id()

#     def get_person_class_id(self):
#         for key, value in self.model.names.items():
#             if value == 'person':
#                 return key
#         return None

#     def image_callback(self, msg):
#         # Convert ROS Image message to OpenCV image
#         frame = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')

#         # Perform prediction
#         results = self.model(frame)
        
#         # Filter for 'person' class only
#         if self.person_class_id is not None:
#             filtered_results = [box for box in results.xyxy[0] if int(box[5]) == self.person_class_id]

#             # Annotate frame with segmentation results
#             for box in filtered_results:
#                 x1, y1, x2, y2, conf, class_id = box
#                 cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
#                 cv2.putText(frame, f'Person {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
#                 center_x = (x1 + x2) / 2
#                 center_y = (y1 + y2) / 2
#                 print(f"Person detected at center: ({center_x:.2f}, {center_y:.2f})")

#         # Display the frame with detections
#         cv2.imshow('YOLOv5 Real-Time Object Detection', frame)
#         cv2.waitKey(1)

# def main(args=None):
#     rp.init(args=args)
#     test_node = Test_node()
#     rp.spin(test_node)
#     test_node.destroy_node()
#     rp.shutdown()

# if __name__ == '__main__':
#     main()




import rclpy as rp
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2 
from cv_bridge import CvBridge
import torch
import numpy as np
import time

class Test_node(Node):
    def __init__(self):
        super().__init__('test_node')
        self.subscriber_ = self.create_subscription(Image, '/camera/color/image_raw', self.image_callback, 10)
        self.br = CvBridge()
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/jinjuuk/dev_ws/pt_files/yolov5n.pt')
        self.person_class_id = self.get_person_class_id()

    def get_person_class_id(self):
        for key, value in self.model.names.items():
            if value == 'person':
                return key
        return None

    def image_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        frame = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        image_center_x = frame.shape[1] / 2
        image_center_y = frame.shape[0] / 2

        # Perform prediction
        results = self.model(frame)
        
        # Convert results to CPU (if using GPU) and to NumPy
        results = results.xyxy[0].cpu().numpy()

        # Filter for 'person' class only
        if self.person_class_id is not None:
            filtered_results = [box for box in results if int(box[5]) == int(self.person_class_id)]
            
            if filtered_results:
                # Find the closest bounding box to the image center
                closest_box = min(filtered_results, key=lambda box: np.sqrt(((box[0] + box[2])/2 - image_center_x)**2 + ((box[1] + box[3])/2 - image_center_y)**2))
                
                # Annotate frame with the closest bounding box
                x1, y1, x2, y2, conf, class_id = closest_box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f'Person {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                print(f"Person detected at center: ({center_x:.2f}, {center_y:.2f})")

        # Display the frame with detections
        cv2.imshow('YOLOv5 Real-Time Object Detection', frame)
        cv2.waitKey(1)

def main(args=None):
    rp.init(args=args)
    test_node = Test_node()
    rp.spin(test_node)
    test_node.destroy_node()
    rp.shutdown()

if __name__ == '__main__':
    main()




