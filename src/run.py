import ssl
import urllib.request
ssl._create_default_https_context = ssl._create_unverified_context
import os
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from data.load_data import Data, load_json
from models.model import Net

class Run(Data):
    def __init__(self, epochs=None, learning_rate=None, batch_size=None, transform=None):
        super().__init__(batch_size=batch_size, transform=transform)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def run_model(self):
        # Step 1: Load the train and test DataLoader objects
        self.train_loader, self.test_loader = self.load_data()

        # Step 2: Create in-memory datasets
        self.train_data, self.train_labels, self.val_data, self.val_labels, self.test_data, self.test_labels = self.create_in_memory_dataset()

        # Step 3: Create model
        model = Net()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # L2 regularization with weight decay
        



        
if __name__ == "__main__":
    run = Run(epochs=1, learning_rate=0.0001, batch_size=64, transform=transforms.ToTensor())
    run.run_model()
