from ultralytics import YOLO
import cv2
import torch

# Load a model
model = YOLO("yolov5n.pt")  # Load the YOLOv8 Nano model


tracker_config = "/home/jinjuuk/.local/lib/python3.10/site-packages/ultralytics/cfg/trackers/botsort.yaml"


try:
    # Start tracking
    results = model.track(classes=[0],            # Track only the "person" class
                          max_det=1,              # Detect up to 5 objects
                          show=True,              # Show the results on the screen
                          source=0,               # Use webcam as source
                          tracker=tracker_config, # Use BotSORT tracker for Re-ID
                          stream=True             # Stream results in real-time
                          ) 

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            person_img = result.orig_img[int(y1):int(y2), int(x1):int(x2)]
     

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

except Exception as e:
    print(f"Error occurred: {e}")

cv2.destroyAllWindows()


