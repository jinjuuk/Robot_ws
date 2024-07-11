import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("path/.pt file")

cam = cv2.VideoCapture(0)

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)