import json
import os

def convert_txt_to_coco(txt_folder, image_folder, output_json, categories):
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    category_mapping = {cat_id: cat_name for cat_id, cat_name in enumerate(categories, 1)}

    for cat_id, cat_name in category_mapping.items():
        coco_format["categories"].append({
            "id": cat_id,
            "name": cat_name
        })

    annotation_id = 1

    for txt_file in os.listdir(txt_folder):
        if txt_file.endswith('.txt'):
            image_id = int(os.path.splitext(txt_file)[0])
            image_path = os.path.join(image_folder, f"{image_id}.jpg")
            
            # Verify that the image exists
            if not os.path.exists(image_path):
                print(f"Image {image_path} does not exist!")
                continue

            # Assuming all images are 640x480, adjust if necessary
            height, width = 480, 640  # Replace with actual dimensions if different

            coco_format["images"].append({
                "id": image_id,
                "file_name": f"{image_id}.jpg",
                "height": height,
                "width": width
            })

            with open(os.path.join(txt_folder, txt_file), 'r') as file:
                lines = file.readlines()

                for line in lines:
                    parts = line.strip().split()
                    class_id = int(parts[1]) + 1  # Assuming class_id starts from 0 in txt, adjust as needed
                    x_center = float(parts[2]) * width
                    y_center = float(parts[3]) * height
                    bbox_width = float(parts[4]) * width
                    bbox_height = float(parts[5]) * height

                    x_min = x_center - bbox_width / 2
                    y_min = y_center - bbox_height / 2

                    coco_format["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": class_id,
                        "bbox": [x_min, y_min, bbox_width, bbox_height],
                        "area": bbox_width * bbox_height,
                        "segmentation": [],
                        "iscrowd": 0
                    })
                    annotation_id += 1

    with open(output_json, 'w') as json_file:
        json.dump(coco_format, json_file, indent=4)

# Example usage
categories = ["bolong", "normal", "remuk"]  # List your categories here
convert_txt_to_coco('/path/to/txt/annotations', '/path/to/images', '/path/to/output/annotations.json', categories)