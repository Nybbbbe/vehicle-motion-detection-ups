import os
import sys

data_folders = [
    'C:/Users/janny/Aalto_project_2/data/elsaesserstr2',
    'C:/Users/janny/Aalto_project_2/data/herrmanmitschstr4',
    'C:/Users/janny/Aalto_project_2/data/herrmanmitschstr4_handheld',
    'C:/Users/janny/Aalto_project_2/data/hirtenweg1',
    'C:/Users/janny/Aalto_project_2/data/madisonallee1',
    'C:/Users/janny/Aalto_project_2/data/zinkmatttenstr1_handheld'
]

def count_images_in_directory(directory_path):
    # List of common image file extensions (add more if needed)
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    image_count = 0
    
    # Iterate over all files in the directory
    for filename in os.listdir(directory_path):
        # Check if the file is an image by looking at its extension
        if os.path.splitext(filename)[1].lower() in image_extensions:
            image_count += 1
    
    return image_count

def is_image_file(filename):
    # List of common image file extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    # Get the file extension and check if it's one of the image types
    return any(filename.lower().endswith(ext) for ext in image_extensions)

def rename_file(original_file_path):
    # Split the original file path to directory and filename
    dir_path, filename = os.path.split(original_file_path)

    # Check if the file is an image
    if not is_image_file(filename):
        print(f'Skipping non-image file: {filename}')
        return
    
    # Split the filename by underscore
    parts = filename.split('_')
    
    # Extract the number as a string and convert to integer for filename
    if len(parts) > 1:
        new_filename = '2_' + str(int(parts[1])) + '.jpg'
    else:
        # If the filename doesn't match expected pattern, don't change it
        return
    
    # Construct new file path
    new_file_path = os.path.join(dir_path, new_filename)
    
    # Rename the file
    os.rename(original_file_path, new_file_path)
    print(f'Renamed "{filename}" to "{new_filename}"')

def generate_fixed_labels(labels_file_path, output_file_path, max_frame):
    # Initialize a dictionary to hold frame labels, default to 0 (no movement)
    frame_labels = {frame: 0 for frame in range(max_frame + 1)}

    # Read the labels file and update frame labels for movement detected
    with open(labels_file_path, 'r') as file:
        for line in file:
            start, end = map(int, line.strip().split(','))
            for frame in range(start, end + 1):
                frame_labels[frame] = 1  # Update label to 1 for movement detected

    # Write the fixed labels to a new file
    with open(output_file_path, 'w') as output_file:
        for frame, label in frame_labels.items():
            output_file.write(f'2_{frame}.jpg, {label}\n')

    print(f"Fixed labels written to {output_file_path}")

def generate_dataset(fixed_labels_path, dataset_output_path, image_folder_path):
    # Read fixed labels into a list
    with open(fixed_labels_path, 'r') as file:
        labels = [line.strip().split(', ') for line in file]

    # Open the dataset output file for writing
    with open(dataset_output_path, 'w') as dataset_file:
        for i in range(len(labels) - 1):
            # Extract frame names and labels
            frame_1, label_1 = labels[i]
            frame_2, label_2 = labels[i + 1]

            # Determine the label for the pair (1 if any of the frames indicates movement)
            pair_label = max(int(label_1), int(label_2))

            # Construct the full paths for the image files
            full_path_1 = os.path.join(image_folder_path, frame_1)
            full_path_2 = os.path.join(image_folder_path, frame_2)

            # Write to the dataset file
            dataset_file.write(f'{full_path_1}, {full_path_2}, {pair_label}\n')

    print(f"Dataset written to {dataset_output_path}")

for data_folder in data_folders:
    # Verify the directory exists
    # if not os.path.isdir(data_folder):
    #     print("The specified directory does not exist.")
    #     sys.exit(1)
    
    # # Go through all files in the directory
    # for filename in os.listdir(data_folder):
    #     file_path = os.path.join(data_folder, filename)
        
    #     # Check if it's a file and not a directory
    #     if os.path.isfile(file_path):
    #         # Use the utility function to rename the file
    #         rename_file(file_path)

    labels_file_path = f'{data_folder}/labels.txt'
    output_file_path = f'{data_folder}/fixed_labels.txt'
    max_frame = count_images_in_directory(data_folder) - 1

    # Generate the fixed labels file
    generate_fixed_labels(labels_file_path, output_file_path, max_frame)

    directory, folder_name = os.path.split(data_folder)

    fixed_labels_path = output_file_path
    dataset_output_path = f'{directory}/{folder_name}_dataset.txt'
    image_folder_path = data_folder  # Update this path to the actual image directory

    # Generate the dataset file
    generate_dataset(fixed_labels_path, dataset_output_path, image_folder_path)