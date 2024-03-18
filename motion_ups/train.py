import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import time
import json
from tqdm import tqdm
from custom_dataloader import CarMovementDataset, resize, resize_and_pad
from model import MotionDetectionModel, MotionDetectionModel_Resnet18_RNN, MotionDetectionModel_Resnet50_RNN
from utils import read_dataset, split_dataset

BACTH_SIZE = 128

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_name = 'MDM_Resnet50_RNN_25_epochs_1_location'
training_date = time.strftime("%Y%m%d-%H%M%S")

MODEL_FOLDER_NAME = f'{model_name}_trained_{training_date}'

if not os.path.exists(MODEL_FOLDER_NAME):
    os.makedirs(MODEL_FOLDER_NAME, exist_ok=True)

# Loading the data
dateset_file_path = 'C:/Users/janny/Aalto_project_2/data/elsaesserstr1_dataset.txt'

pairs, labels = read_dataset(dateset_file_path)
pairs_train, pairs_test, labels_train, labels_test = split_dataset(pairs, labels)

print(f'Total pairs: {len(pairs)}')
print(f'Training pairs: {len(pairs_train)}')
print(f'Testing pairs: {len(pairs_test)}')

transform = transforms.Compose([
    transforms.Lambda(lambda img: resize(img, target_width=512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = CarMovementDataset(pairs=pairs_train, labels=labels_train, transform=transform)
# train_dataloader = DataLoader(train_dataset, batch_size=BACTH_SIZE, shuffle=True)

# test_dataset = CarMovementDataset(pairs=pairs_test, labels=labels_test, transform=transform)
# test_dataloader = DataLoader(test_dataset, batch_size=BACTH_SIZE, shuffle=False)

# Training

# Instantiate the model
model = MotionDetectionModel_Resnet50_RNN(num_classes=2)
# Define the loss function
criterion = nn.CrossEntropyLoss()
# Define the optimizer (using Adam here, but you can choose others like SGD)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model.to(device)

# Number of epochs to train for
num_epochs = 25

# Lists to store metrics for visualization
epoch_losses = []
batch_losses = []
val_epoch_losses = []
val_batch_losses = []
train_accuracies = []
test_accuracies = []

dataset_size = len(train_dataset)
train_size = int(dataset_size * 0.8)
val_size = dataset_size - train_size

for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    epoch_batch_losses = []

    # Splitting the dataset
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
    
    # Creating data loaders
    train_loader = DataLoader(train_subset, batch_size=BACTH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BACTH_SIZE)

    # Wrap train_dataloader with tqdm for a progress bar
    progress_bar = tqdm(enumerate(train_loader, 0), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")

    start_time = time.time()
    
    for i, data in progress_bar:
        # Each batch consists of image pairs and their labels
        img_pair, labels = data
        img1, img2 = img_pair[0].to(device), img_pair[1].to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(img1, img2)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        running_loss += batch_loss
        epoch_batch_losses.append(batch_loss)

        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

        # Update progress bar with loss information
        progress_bar.set_postfix(loss=running_loss/(i+1))

    # Calculate average loss for the epoch
    avg_loss = running_loss / len(train_loader)
    epoch_losses.append(avg_loss)
    batch_losses.extend(epoch_batch_losses)
    train_accuracy = 100 * correct_train / total_train
    train_accuracies.append(train_accuracy)

    # Validation phase
    model.eval()
    correct_test = 0
    total_test = 0
    val_running_loss = 0.0
    with torch.no_grad():
        val_progress_bar = tqdm(enumerate(val_loader, 0), total=len(val_loader), desc=f"Epoch {epoch+1}/{num_epochs}")
        for i, data in val_progress_bar:
            img_pair, labels = data
            img1, img2 = img_pair[0].to(device), img_pair[1].to(device)
            labels = labels.to(device)

            outputs = model(img1, img2)
            loss = criterion(outputs, labels)
            val_batch_losses.append(loss)

            val_running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

            val_progress_bar.set_postfix(loss=val_running_loss/(i+1))

    val_loss = val_running_loss / len(val_loader)
    val_epoch_losses.append(val_loss)

    test_accuracy = 100 * correct_test / total_test
    test_accuracies.append(test_accuracy)

    end_time = time.time()
    epoch_duration = end_time - start_time

    print(f'Finished Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Duration: {epoch_duration:.2f} sec')
    print(f'Validation Loss: {val_loss:.4f}, Val Acc: {test_accuracy:.2f}%')

    
    model_file_name = f"{MODEL_FOLDER_NAME}/{MODEL_FOLDER_NAME}_epoch_{epoch+1}.pth"

    # Save the model state dictionary
    torch.save(model.state_dict(), model_file_name)

    print(f"Model saved as {model_file_name}")


print('Finished Training')

model_file_name = f"{MODEL_FOLDER_NAME}/{MODEL_FOLDER_NAME}.pth"


# Save the model state dictionary
torch.save(model.state_dict(), model_file_name)

print(f"Model saved as {model_file_name}")

epoch_losses_filename = f"{MODEL_FOLDER_NAME}/epoch_losses.json"
batch_losses_filename = f"{MODEL_FOLDER_NAME}/batch_losses.json"
val_epoch_losses_filename = f"{MODEL_FOLDER_NAME}/val_epoch_losses.json"
val_batch_losses_filename = f"{MODEL_FOLDER_NAME}/val_batch_losses.json"
train_accuracies_filename = f"{MODEL_FOLDER_NAME}/train_accuracies.json"
test_accuracies_filename = f"{MODEL_FOLDER_NAME}/test_accuracies.json"

with open(epoch_losses_filename, 'w') as f:
    json.dump(epoch_losses, f)
print(f"Saved epoch losses to {epoch_losses_filename}")

with open(batch_losses_filename, 'w') as f:
    json.dump(batch_losses, f)
print(f"Saved batch losses to {batch_losses_filename}")

with open(val_epoch_losses_filename, 'w') as f:
    json.dump(val_epoch_losses, f)
print(f"Saved val epoch losses to {val_epoch_losses_filename}")

with open(val_batch_losses_filename, 'w') as f:
    json.dump(val_batch_losses, f)
print(f"Saved val batch losses to {val_batch_losses_filename}")

with open(train_accuracies_filename, 'w') as f:
    json.dump(train_accuracies, f)
print(f"Saved training accuracies to {train_accuracies_filename}")

with open(test_accuracies_filename, 'w') as f:
    json.dump(test_accuracies, f)
print(f"Saved test accuracies to {test_accuracies_filename}")