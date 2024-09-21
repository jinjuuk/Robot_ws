import cv2
import albumentations as A
import numpy as np
import os

# Define the augmentations
transform = A.Compose([
    A.HorizontalFlip(p=0.9),
    A.VerticalFlip(p=0.9),
    A.Rotate(limit=45, p=0.9),
    A.RandomBrightnessContrast(p=0.9, brightness_limit=0.2, contrast_limit=0.2),
    A.Blur(blur_limit=3, p=0.9),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.9),
])

# Path to the folder containing the images
input_folder = '/home/jinjuuk/dev_ws/captured_images'
output_folder = '/home/jinjuuk/dev_ws/captured_images_arg_test'

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Function to perform augmentation
def augment_image(image_path, output_path, start_index, num_augmented=3):
    image = cv2.imread(image_path)
    for i in range(num_augmented):
        augmented = transform(image=image)['image']
        augmented_filename = os.path.join(output_path, f"augmented_image_{start_index + i}.jpg")
        cv2.imwrite(augmented_filename, augmented)
    return start_index + num_augmented

# Iterate through all images and apply augmentation
current_index = 0
for image_file in os.listdir(input_folder):
    if image_file.endswith(('.jpg', '.jpeg', '.png')):
        current_index = augment_image(os.path.join(input_folder, image_file), output_folder, current_index)

print("Data augmentation completed.")
