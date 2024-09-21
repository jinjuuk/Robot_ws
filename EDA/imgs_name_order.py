import os
import shutil

# Path to the folder containing the images
input_folder = '/home/jinjuuk/dev_ws/captured_images'
output_folder = '/home/jinjuuk/dev_ws/captured_images_arg_test'

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Function to rename and copy images
def rename_and_copy_images(input_folder, output_folder):
    files = sorted(os.listdir(input_folder))
    for i, file_name in enumerate(files):
        if file_name.endswith(('.jpg', '.jpeg', '.png')):
            src_path = os.path.join(input_folder, file_name)
            dst_path = os.path.join(output_folder, f'image_{i}.jpg')
            shutil.copy(src_path, dst_path)

# Rename and copy images from input folder to output folder
rename_and_copy_images(input_folder, output_folder)

print("Renaming and copying completed.")

