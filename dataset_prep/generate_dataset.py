import os

def generate_dataset(fixed_labels_path, dataset_output_path, data_folder_path):
    # Read fixed labels into a list
    with open(fixed_labels_path, 'r') as file:
        labels = [line.strip().split(', ') for line in file]

    # Open the dataset output file for writing
    with open(dataset_output_path, 'w') as dataset_file:
        for i in range(2, len(labels) - 2):
            # Extract frame names and labels
            frame_1, _ = labels[i - 2]
            frame_2, _ = labels[i - 1]
            frame_3, label_3 = labels[i]
            frame_4, _ = labels[i + 1]
            frame_5, _ = labels[i + 2]

            # Construct the full paths for the image files
            img_full_path_1 = os.path.join(data_folder_path, 'images', frame_1)
            img_full_path_2 = os.path.join(data_folder_path, 'images', frame_2)
            img_full_path_3 = os.path.join(data_folder_path, 'images', frame_3)
            img_full_path_4 = os.path.join(data_folder_path, 'images', frame_4)
            img_full_path_5 = os.path.join(data_folder_path, 'images', frame_5)

            audio_full_path_0 = os.path.join(data_folder_path, 'audio_0', frame_3)
            audio_full_path_1 = os.path.join(data_folder_path, 'audio_1', frame_3)
            audio_full_path_2 = os.path.join(data_folder_path, 'audio_2', frame_3)
            audio_full_path_3 = os.path.join(data_folder_path, 'audio_3', frame_3)
            audio_full_path_4 = os.path.join(data_folder_path, 'audio_4', frame_3)
            audio_full_path_5 = os.path.join(data_folder_path, 'audio_5', frame_3)
            audio_full_path_6 = os.path.join(data_folder_path, 'audio_6', frame_3)

            # Write to the dataset file
            dataset_file.write(
                f'{img_full_path_1}, {img_full_path_2}, '
                f'{img_full_path_3}, {img_full_path_4}, '
                f'{img_full_path_5}, {audio_full_path_0}, '
                f'{audio_full_path_1}, {audio_full_path_2}, '
                f'{audio_full_path_3}, {audio_full_path_5}, '
                f'{audio_full_path_4}, {audio_full_path_6}, '
                f'{label_3}\n' 
            )

    print(f"Dataset written to {dataset_output_path}")

# Paths configuration
fixed_labels_path = 'C:/Users/janny/Aalto_project_2/data/granadaallee1/images/fixed_labels.txt'
dataset_output_path = 'C:/Users/janny/Aalto_project_2/data/granadaallee1_dataset_IaA.txt'
data_folder_path = 'C:/Users/janny/Aalto_project_2/data/granadaallee1'  # Update this path to the actual image directory

# Generate the dataset file
generate_dataset(fixed_labels_path, dataset_output_path, data_folder_path)