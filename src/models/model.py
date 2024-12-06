import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    """
    A simplified CNN for MNIST image classification.
    """
    def __init__(self):
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Input: (1, 28, 28) -> Output: (32, 28, 28)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Output: (64, 28, 28)
        
        # Max Pooling
        self.pool = nn.MaxPool2d(2, 2)  # Reduces spatial dimensions by half

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Input size after pooling (64, 7, 7) = 64*7*7
        self.fc2 = nn.Linear(128, 10)  # 10 classes for MNIST

        # Dropout
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Apply conv1 and pooling
        x = self.pool(F.relu(self.conv2(x)))  # Apply conv2 and pooling
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = F.relu(self.fc1(x))  # Fully connected layer 1
        x = self.dropout(x)  # Dropout layer
        x = self.fc2(x)  # Fully connected layer 2
        return F.log_softmax(x, dim=1)  # Softmax output for classification

