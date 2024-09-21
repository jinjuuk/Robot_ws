import os

# Define the directory containing the images
image_directory = '/home/jinjuuk/dev_ws/data_prepare/Robot_Arm_org'

# List all files in the directory
files = os.listdir(image_directory)

# Define the pattern to keep
pattern_to_keep = '_aug_0.jpg'

# Iterate through the files and delete those that do not match the pattern
for file_name in files:
    # Check if the file matches the pattern to keep
    if not file_name.endswith(pattern_to_keep):
        # Construct full file path
        file_path = os.path.join(image_directory, file_name)
        # Remove the file
        os.remove(file_path)
        print(f"Deleted: {file_path}")

print("Cleanup completed.")

