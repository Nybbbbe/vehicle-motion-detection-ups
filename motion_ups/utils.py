from sklearn.model_selection import train_test_split

def read_dataset(file_path, n=None):
    """
    Read n lines from the dataset file and parse them into pairs and labels.

    Args:
        file_path (str): The path to the dataset file.
        n (int): The number of rows to read from the file.

    Returns:
        pairs (list of tuples): List of tuples containing image paths for the pairs.
        labels (list of int): List of labels.
    """
    pairs = []
    labels = []
    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            if n != None and i >= n:
                break
            parts = line.strip().split(', ')
            if len(parts) == 3:
                img_path1, img_path2, label = parts
                pairs.append((img_path1, img_path2))
                labels.append(int(label))
    return pairs, labels

def split_dataset(pairs, labels, test_size=0.2):
    # Split the dataset into training and testing sets
    pairs_train, pairs_test, labels_train, labels_test = train_test_split(pairs, labels, test_size=test_size, random_state=42)
    return pairs_train, pairs_test, labels_train, labels_test