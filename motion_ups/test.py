import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from custom_dataloader import CarMovementDataset, resize, resize_and_pad
from model import MotionDetectionModel, MotionDetectionModel_Resnet18_RNN, MotionDetectionModel_Resnet50_RNN
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
from utils import read_dataset, split_dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

file_path = 'C:/Users/janny/Aalto_project_2/data/full_dataset.txt'

pairs, labels = read_dataset(file_path)
pairs_train, pairs_test, labels_train, labels_test = split_dataset(pairs, labels, test_size=0.2)

print(f'Total pairs: {len(pairs)}')
print(f'Training pairs: {len(pairs_train)}')
print(f'Testing pairs: {len(pairs_test)}')

# Assuming you have already defined pairs_test and labels_test during training
transform = transforms.Compose([
    transforms.Lambda(lambda img: resize(img, target_width=512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_dataset = CarMovementDataset(pairs=pairs_test, labels=labels_test, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

model_folder = 'MDM_Resnet50_RNN_25_epochs_1_location_trained_20240318-111308'

# Load the trained model (assuming it's saved as 'motion_detection_model.pth')
model = MotionDetectionModel_Resnet50_RNN(num_classes=2)
model.load_state_dict(torch.load(f'{model_folder}/{model_folder}.pth'))
model.to(device)
model.eval()

# Initialize lists to store targets and predictions
all_targets = []
all_predictions = []

with torch.no_grad():
    for i, data in tqdm(enumerate(test_dataloader, 0), total=len(test_dataloader)):
        img_pair, labels = data
        img1, img2 = img_pair[0].to(device), img_pair[1].to(device)
        labels = labels.to(device)
        outputs = model(img1, img2)
        _, predicted = torch.max(outputs.data, 1)

        all_targets.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

# Calculate metrics
accuracy = accuracy_score(all_targets, all_predictions)
precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_predictions, average='binary')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1-score": f1,
}

# _unseen_location
with open(f'{model_folder}/{model_folder}_results_test.json', 'w') as f:
    json.dump(metrics, f, indent=4)