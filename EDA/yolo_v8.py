import os
from ultralytics import YOLO


# 모델 로드
model = YOLO("yolov8s.pt")


# 모델 학습 설정
model.train(data="/home/jinjuuk/dev_ws/data_prepare/test_combine/custom_data.yaml",
            epochs=300,
            batch=16,
            imgsz=640,
            lr0=0.001,
            name= 'custom_combine_test',
            optimizer='Adam',
            pretrained=True,
            patience=20,
            )  # 모델 학습