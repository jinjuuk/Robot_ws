import os
import shutil

def rename_and_copy_folder(image_folder, label_folder, output_path, start_index=661):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    image_output_path = os.path.join(output_path, "images")
    label_output_path = os.path.join(output_path, "labels")

    if not os.path.exists(image_output_path):
        os.makedirs(image_output_path)

    if not os.path.exists(label_output_path):
        os.makedirs(label_output_path)

    # Mapping for class reordering
    class_mapping = {0: 1, 1: 0, 2: 2}  # 'cup'->1, 'star'->0, 'waste'->2

    # Copy images and labels from image_folder and label_folder to output_path with new names
    current_index = start_index
    image_files = [f for f in os.listdir(image_folder) if f.endswith(".jpg")]
    label_files = [f for f in os.listdir(label_folder) if f.endswith(".txt")]

    for image_file in image_files:
        new_image_name = f"star_cup{current_index}.jpg"
        shutil.copy(os.path.join(image_folder, image_file), os.path.join(image_output_path, new_image_name))

        corresponding_label_file = image_file.replace(".jpg", ".txt")
        if corresponding_label_file in label_files:
            new_label_name = f"star_cup{current_index}.txt"
            
            # Read the label file
            with open(os.path.join(label_folder, corresponding_label_file), 'r') as label_in:
                lines = label_in.readlines()

            # Write the modified label file
            with open(os.path.join(label_output_path, new_label_name), 'w') as label_out:
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) > 0:
                        class_id = int(parts[0])
                        new_class_id = class_mapping[class_id]
                        label_out.write(f"{new_class_id} {' '.join(parts[1:])}\n")

        current_index += 1

# Usage
image_folder = "/home/jinjuuk/dev_ws/data_prepare/test_roboflow/images"
label_folder = "/home/jinjuuk/dev_ws/data_prepare/test_roboflow/labels"
output_path = "/home/jinjuuk/dev_ws/data_prepare/output_data/"

rename_and_copy_folder(image_folder, label_folder, output_path, start_index=661)
















# import os
# import shutil

# def rename_and_merge_folders(folder1, folder2, output_path, start_index=528):
#     if not os.path.exists(output_path):
#         os.makedirs(output_path)

#     image_output_path = os.path.join(output_path, "images")
#     label_output_path = os.path.join(output_path, "labels")

#     if not os.path.exists(image_output_path):
#         os.makedirs(image_output_path)

#     if not os.path.exists(label_output_path):
#         os.makedirs(label_output_path)

#     # First, copy files from folder1
#     for filename in os.listdir(folder1):
#         if filename.endswith(".jpg"):
#             shutil.copy(os.path.join(folder1, filename), os.path.join(image_output_path, filename))
#         elif filename.endswith(".txt"):
#             shutil.copy(os.path.join(folder1, filename), os.path.join(label_output_path, filename))

#     # Then, rename and copy files from folder2
#     current_index = start_index
#     for filename in os.listdir(folder2):
#         if filename.endswith(".jpg"):
#             new_image_name = f"star_cup{current_index}.jpg"
#             shutil.copy(os.path.join(folder2, filename), os.path.join(image_output_path, new_image_name))

#             label_filename = filename.replace(".jpg", ".txt")
#             new_label_name = f"star_cup{current_index}.txt"
#             shutil.copy(os.path.join(folder2, label_filename), os.path.join(label_output_path, new_label_name))

#             current_index += 1

# # Usage
# folder1 = "path/to/first/folder"
# folder2 = "path/to/second/folder"
# output_path = "/home/jinjuuk/dev_ws/data_prepare/output_data"

# rename_and_merge_folders(folder1, folder2, output_path, start_index=528)
