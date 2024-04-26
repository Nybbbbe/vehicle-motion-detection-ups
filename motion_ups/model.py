import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights, ResNet50_Weights

class MotionDetectionModel(nn.Module):
    def __init__(self, num_classes=2):
        super(MotionDetectionModel, self).__init__()
        # Use ResNet18 as the base model for feature extraction
        base_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.base_cnn = nn.Sequential(*list(base_model.children())[:-1])  # Remove the last layer
        
        # Assuming the output features of ResNet18 is 512, and we're concatenating features from 2 images
        self.fc1 = nn.Linear(512 * 2, 512)  # First fully connected layer
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)  # Final classification layer

    def forward(self, image1, image2):
        # Feature extraction for each image
        features1 = self.base_cnn(image1)
        features2 = self.base_cnn(image2)
        
        # Flatten the features
        features1 = features1.view(features1.size(0), -1)
        features2 = features2.view(features2.size(0), -1)
        
        # Concatenate the features from the two images
        combined_features = torch.cat((features1, features2), dim=1)
        
        # Fully connected layers for classification
        x = self.fc1(combined_features)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x
    

class MotionDetectionModel_Resnet18_RNN(nn.Module):
    def __init__(self, input_size=1024, hidden_size=128, num_layers=1, num_classes=2):
        super(MotionDetectionModel_Resnet18_RNN, self).__init__()
        # Use ResNet18 as the base model for feature extraction
        base_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.base_cnn = nn.Sequential(*list(base_model.children())[:-1])  # Remove the last layer
        
        self.rnn = nn.RNN(input_size=input_size, 
                          hidden_size=hidden_size, 
                          num_layers=num_layers, 
                          batch_first=True)
        
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, image1, image2):
        # Feature extraction for each image
        features1 = self.base_cnn(image1)
        features2 = self.base_cnn(image2)
        
        # Flatten the features
        features1 = features1.view(features1.size(0), -1)
        features2 = features2.view(features2.size(0), -1)
        
        # Concatenate the features from the two images
        combined_features = torch.cat((features1, features2), dim=1)
        
        # Fully connected layers for classification
        out, _ = self.rnn(combined_features)
        # out = out.reshape(-1, out.shape[2])
        out = self.fc(out)
        
        return out
    

class MotionDetectionModel_Resnet50_RNN(nn.Module):
    def __init__(self, input_size=4096, hidden_size=128, num_layers=1, num_classes=2):
        super(MotionDetectionModel_Resnet50_RNN, self).__init__()
        # Use ResNet18 as the base model for feature extraction
        base_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.base_cnn = nn.Sequential(*list(base_model.children())[:-1])  # Remove the last layer
        
        self.rnn = nn.RNN(input_size=input_size, 
                          hidden_size=hidden_size, 
                          num_layers=num_layers, 
                          batch_first=True)
        
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, image1, image2):
        # Feature extraction for each image
        features1 = self.base_cnn(image1)
        features2 = self.base_cnn(image2)
        
        # Flatten the features
        features1 = features1.view(features1.size(0), -1)
        features2 = features2.view(features2.size(0), -1)
        
        # Concatenate the features from the two images
        combined_features = torch.cat((features1, features2), dim=1)
        
        # Fully connected layers for classification
        out, _ = self.rnn(combined_features)
        # out = out.reshape(-1, out.shape[2])
        out = self.fc(out)
        
        return out
    

class MotionDetectionModel_Resnet50_RNN_5(nn.Module):
    def __init__(self, input_size=2048 * 5, hidden_size=512, num_layers=1, num_classes=2):
        super(MotionDetectionModel_Resnet50_RNN_5, self).__init__()
        # Use ResNet18 as the base model for feature extraction
        base_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.base_cnn = nn.Sequential(*list(base_model.children())[:-1])  # Remove the last layer
        
        self.rnn = nn.RNN(input_size=input_size, 
                          hidden_size=hidden_size, 
                          num_layers=num_layers, 
                          batch_first=True)
        
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, image1, image2, image3, image4, image5):
        # Feature extraction for each image
        features1 = self.base_cnn(image1)
        features2 = self.base_cnn(image2)
        features3 = self.base_cnn(image3)
        features4 = self.base_cnn(image4)
        features5 = self.base_cnn(image5)
        
        
        # Flatten the features
        features1 = features1.view(features1.size(0), -1)
        features2 = features2.view(features2.size(0), -1)
        features3 = features3.view(features3.size(0), -1)
        features4 = features4.view(features4.size(0), -1)
        features5 = features2.view(features5.size(0), -1)
        
        # Concatenate the features from the two images
        combined_features = torch.cat((features1, features2, features3, features4, features5), dim=1)
        
        # Fully connected layers for classification
        out, _ = self.rnn(combined_features)
        # out = out.reshape(-1, out.shape[2])
        out = self.fc(out)
        
        return out