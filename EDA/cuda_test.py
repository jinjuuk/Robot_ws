import torch
import cv2
import cv2.aruco as aruco
import mediapipe as mp
from ultralytics import YOLO
import numpy as np

# YOLOv8 모델 로드
model = YOLO('/home/jinjuuk/dev_ws/pt_files/segmentation_s_batch16_freeze8.pt')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if torch.cuda.is_available():
    model.to(device)

print(f'Using device: {device}')

# CUDA 장치 정보 출력
if torch.cuda.is_available():
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Device Count: {torch.cuda.device_count()}")

# 모델의 매개변수 이동 확인
for name, param in model.named_parameters():
    if param.is_cuda:
        print(f"{name} is on CUDA")
    else:
        print(f"{name} is not on CUDA")

# PyTorch GPU 사용 예제
x = torch.rand(5, 3).to(device)
y = torch.rand(5, 3).to(device)
z = x + y
print(f"Result tensor z is on device: {z.device}")

# Mediapipe hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

# ArUco marker detection
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 여기에 다른 코드 추가

cap.release()
cv2.destroyAllWindows()
