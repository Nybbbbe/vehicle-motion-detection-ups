import os
import sys

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
    
    # # Split the filename by underscore
    parts = filename.split('_')
    
    # # Extract the number as a string and convert to integer for filename
    # if len(parts) > 1:
    #     new_filename = str(int(parts[1].split('.')[0])) + '.jpg'
    # else:
    #     # If the filename doesn't match expected pattern, don't change it
    #     return

    # # Split the filename by underscore
    parts = filename.split('.')
    
    # Extract the number as a string and convert to integer for filename
    if len(parts) > 1:
        new_filename = str(int(parts[0])) + '.jpg'
    else:
        # If the filename doesn't match expected pattern, don't change it
        return
    
    # Construct new file path
    new_file_path = os.path.join(dir_path, new_filename)
    
    # Rename the file
    os.rename(original_file_path, new_file_path)
    print(f'Renamed "{filename}" to "{new_filename}"')

def main():
    # Check for the right number of command line arguments
    if len(sys.argv) != 2:
        print("Usage: python script.py <directory>")
        sys.exit(1)
    
    # Get the directory path from command line arguments
    directory_path = sys.argv[1]
    
    # Verify the directory exists
    if not os.path.isdir(directory_path):
        print("The specified directory does not exist.")
        sys.exit(1)
    
    # Go through all files in the directory
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        
        # Check if it's a file and not a directory
        if os.path.isfile(file_path):
            # Use the utility function to rename the file
            rename_file(file_path)

if __name__ == "__main__":
    main()