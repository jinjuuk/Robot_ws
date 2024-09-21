import torch
from ultralytics import YOLO
from torch.quantization import quantize_dynamic, prepare, convert


# 예제 모델 로드 (YOLOv8 모델을 사용하는 경우 해당 모델로 대체)
model = YOLO("/home/jinjuuk/dev_ws/pt_files/newjeans_m.pt")
model.eval()


# Quantization 설정
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
prepare(model.model, inplace=True)


# Custom dataset for loading images from a directory
class ImageFolderDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith(('png', 'jpg', 'jpeg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, 0  # dummy label since we don't need labels for calibration

# Data transforms
transform = transforms.Compose([
    transforms.Resize((640, 640)),  # Resizing to YOLOv8's expected input size
    transforms.ToTensor(),
])

# Data loader
calibration_dataset = ImageFolderDataset(image_dir, transform)
calibration_loader = DataLoader(calibration_dataset, batch_size=10, shuffle=True)

# Calibration 수행 함수
def calibrate(model, data_loader):
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            inputs, _ = batch  # 입력과 레이블(여기서는 레이블을 사용하지 않음)
            model(inputs)

# Calibration 수행
calibrate(model.model, calibration_loader)

# 모델 Quantization
convert(model.model, inplace=True)

# Quantized 모델 저장
torch.save(model.model.state_dict(), '/home/jinjuuk/dev_ws/pt_files/quantized_model.pth')

# Quantized 모델 로드 및 추론
quantized_model = YOLO("yolov8n.pt")  # YOLO 모델 로드
quantized_model.model.load_state_dict(torch.load('/home/jinjuuk/dev_ws/pt_files/quantized_model.pth'))
quantized_model.model.eval()

# 임의의 입력 데이터로 추론
example_inputs = torch.randn(1, 3, 640, 640)  # YOLOv8의 예상 입력 크기
output = quantized_model.model(example_inputs)

# 추론 결과 출력
print(output)
