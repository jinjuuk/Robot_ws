import cv2
import torch
import time
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/jinjuuk/dev_ws/pt_files/yolov5s.pt')

DeepSORT_path = '/home/jinjuuk/Yolov5_DeepSort_Pytorch'
# Initialize DeepSORT
cfg = get_config()
cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
deepsort = DeepSort(
    cfg.DEEPSORT.REID_CKPT,
    max_dist=cfg.DEEPSORT.MAX_DIST,
    min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
    max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
    max_age=cfg.DEEPSORT.MAX_AGE,
    n_init=cfg.DEEPSORT.N_INIT,
    nn_budget=cfg.DEEPSORT.NN_BUDGET,
    use_cuda=True
)

# Open the camera
cap = cv2.VideoCapture(0)  # 0 for the default camera

# Set camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Could not open video device.")
    exit()

while cap.isOpened():
    success, frame = cap.read()

    if success:
        # Perform YOLOv5 prediction
        results = model(frame)

        # Filter for the 'person' class
        person_class_id = None
        for key, value in model.names.items():
            if value == 'person':
                person_class_id = key
                break

        if person_class_id is not None:
            # Extract bounding boxes, confidences, and class IDs
            bboxes = []
            for box in results.xyxy[0]:
                if int(box[5]) == person_class_id:
                    x1, y1, x2, y2, conf, class_id = box
                    bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]  # xywh format
                    bboxes.append((bbox, conf))

            # Prepare the data for DeepSORT
            bbox_xywh = []
            confs = []
            for bbox, conf in bboxes:
                bbox_xywh.append(bbox)
                confs.append(conf)

            # Convert to numpy array
            bbox_xywh = torch.tensor(bbox_xywh)
            confs = torch.tensor(confs)

            # Update DeepSORT tracker
            outputs = deepsort.update(bbox_xywh, confs, frame)

            # Draw bounding boxes and tracker IDs
            for output in outputs:
                x1, y1, x2, y2, track_id = output
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f'ID: {int(track_id)}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('YOLOv5 DeepSORT Real-Time Object Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
