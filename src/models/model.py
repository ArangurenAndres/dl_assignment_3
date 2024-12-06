import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    """
    A CNN following the specified architecture for MNIST image classification.
    """
    def __init__(self):
        super(Net, self).__init__()

        # First Convolutional Layer: (batch, 1, 28, 28) -> (batch, 16, 28, 28)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)

        # Second Convolutional Layer: (batch, 16, 28, 28) -> (batch, 32, 14, 14)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

        # Third Convolutional Layer: (batch, 32, 14, 14) -> (batch, 64, 7, 7)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # Fully connected layer: Flattening the tensor (batch, 64*3*3) -> (batch, 10)
        self.fc1 = nn.Linear(64 * 3 * 3, 10)

        # Max Pooling layers
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # First Convolution + Max Pooling
        x = self.pool(F.relu(self.conv1(x)))

        # Second Convolution + Max Pooling
        x = self.pool(F.relu(self.conv2(x)))

        # Third Convolution + Max Pooling
        x = self.pool(F.relu(self.conv3(x)))

        # Flatten the output from convolution layers
        x = torch.flatten(x, 1)

        # Fully connected layer
        x = self.fc1(x)

        return F.log_softmax(x, dim=1)  # LogSoftmax output for classification
