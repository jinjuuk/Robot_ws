import os
import shutil

# Define the paths to the input folders
input_folder_1 = '/home/jinjuuk/dev_ws/additional_waste_img'
input_folder_2 = '/home/jinjuuk/dev_ws/star_side_imgs'

# Define the path to the output folder
output_folder = '/home/jinjuuk/dev_ws/images_combine'

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Function to copy and rename images from a folder
def copy_and_rename_images(input_folder, output_folder, start_index):
    files = sorted(os.listdir(input_folder))
    for i, file_name in enumerate(files):
        if file_name.endswith(('.jpg', '.jpeg', '.png')):
            src_path = os.path.join(input_folder, file_name)
            dst_path = os.path.join(output_folder, f'combined_image_{start_index + i}.jpg')
            shutil.copy(src_path, dst_path)

# Copy and rename images from the first folder
start_index = 0
copy_and_rename_images(input_folder_1, output_folder, start_index)

# Copy and rename images from the second folder
start_index += len(os.listdir(input_folder_1))
copy_and_rename_images(input_folder_2, output_folder, start_index)


print("Images have been combined and renamed successfully.")
