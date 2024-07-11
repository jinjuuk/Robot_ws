import os
import xml.etree.ElementTree as ET

# Pascal VOC 형식의 XML 파일을 YOLO 형식으로 변환하는 함수
def convert_voc_to_yolo(xml_file, yolo_file, class_names):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    with open(yolo_file, 'w') as f:
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            if int(difficult) == 1:
                continue
            cls = obj.find('name').text
            if cls not in class_names:
                continue
            cls_id = class_names.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
                 float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            bb = ((b[0] + b[1]) / 2.0 / width, (b[2] + b[3]) / 2.0 / height,
                  (b[1] - b[0]) / width, (b[3] - b[2]) / height)
            f.write(f"{cls_id} {' '.join([str(a) for a in bb])}\n")

# 데이터셋 경로 및 클래스 목록 설정
xml_folder = '/home/jinjuuk/Downloads/new_datasets'  # XML 파일들이 있는 폴더 경로
yolo_folder = '/home/jinjuuk/Downloads/TEST'  # YOLO 형식의 .txt 파일을 저장할 폴더 경로
classes = ['star', 'cup']  # 클래스 이름 목록

# 폴더 생성
os.makedirs(yolo_folder, exist_ok=True)

# XML 파일을 YOLO 형식으로 변환
for xml_file in os.listdir(xml_folder):
    if xml_file.endswith('.xml'):
        xml_path = os.path.join(xml_folder, xml_file)
        yolo_path = os.path.join(yolo_folder, xml_file.replace('.xml', '.txt'))
        convert_voc_to_yolo(xml_path, yolo_path, classes)






