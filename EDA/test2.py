import cv2
import os
import time

def check_camera_indices(max_indices=30):
    available_indices = []
    for index in range(max_indices):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            print(f"Camera found at index {index}")
            available_indices.append(index)
            cap.release()
        else:
            print(f"No camera found at index {index}")
    return available_indices

# Ensure the required environment variable is set
os.environ['QT_QPA_PLATFORM'] = 'xcb'

while True:
    print("Checking for available camera indices...")
    available_cameras = check_camera_indices()
    print(f"Available camera indices: {available_cameras}")

    if available_cameras:
        camera_index = available_cameras[0]
        print(f"Using camera at index {camera_index}")

        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            print("Error: Could not open video device.")
            exit()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            cv2.imshow('Camera Feed', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    # Wait for a few seconds before checking again
    time.sleep(5)

