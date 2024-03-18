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

if __name__ == "__main__":
    labels_file_path = 'C:/Users/janny/Aalto_project_2/data/zinkmatttenstr1/labels.txt'
    output_file_path = 'C:/Users/janny/Aalto_project_2/data/zinkmatttenstr1/fixed_labels.txt'
    max_frame = 2877

    # Generate the fixed labels file
    generate_fixed_labels(labels_file_path, output_file_path, max_frame)