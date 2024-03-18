import os

def generate_dataset(fixed_labels_path, dataset_output_path, image_folder_path):
    # Read fixed labels into a list
    with open(fixed_labels_path, 'r') as file:
        labels = [line.strip().split(', ') for line in file]

    # Open the dataset output file for writing
    with open(dataset_output_path, 'w') as dataset_file:
        for i in range(len(labels) - 4):
            # Extract frame names and labels
            frame_1, label_1 = labels[i]
            frame_2, label_2 = labels[i + 1]
            frame_3, label_3 = labels[i + 2]
            frame_4, label_4 = labels[i + 3]
            frame_5, label_5 = labels[i + 4]


            # Determine the label for the pair (1 if any of the frames indicates movement)
            pair_label = max(int(label_1), int(label_2), int(label_3), int(label_4), int(label_5))

            # Construct the full paths for the image files
            full_path_1 = os.path.join(image_folder_path, frame_1)
            full_path_2 = os.path.join(image_folder_path, frame_2)
            full_path_3 = os.path.join(image_folder_path, frame_3)
            full_path_4 = os.path.join(image_folder_path, frame_4)
            full_path_5 = os.path.join(image_folder_path, frame_5)

            # Write to the dataset file
            dataset_file.write(f'{full_path_1}, {full_path_2}, {full_path_3}, {full_path_4}, {full_path_5}, {pair_label}\n')

    print(f"Dataset written to {dataset_output_path}")

# Paths configuration
fixed_labels_path = 'C:/Users/janny/Aalto_project_2/data/elsaesserstr1/fixed_labels.txt'
dataset_output_path = 'C:/Users/janny/Aalto_project_2/data/elsaesserstr1_dataset5.txt'
image_folder_path = 'C:/Users/janny/Aalto_project_2/data/elsaesserstr1'  # Update this path to the actual image directory

# Generate the dataset file
generate_dataset(fixed_labels_path, dataset_output_path, image_folder_path)