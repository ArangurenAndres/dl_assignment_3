import ssl
import urllib.request
ssl._create_default_https_context = ssl._create_unverified_context
import os
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from data.load_data import Data, load_json
from models.model import Net
from train.train import train_model

class Run(Data):
    def __init__(self, transform=None, config_file=None):
        # Assign config file values to class attributes
        self.config_file = config_file
        self.epochs = self.config_file["epochs"]
        self.batch_size = self.config_file["batch_size"]
        self.learning_rate = self.config_file["lr"]
        super().__init__(batch_size=self.batch_size, transform=transform)

    def run_model(self):
        # Step 1: Load the train and test DataLoader objects
        self.train_loader, self.test_loader = self.load_data()

        # Step 2: Create in-memory datasets
        self.train_data, self.train_labels, self.val_data, self.val_labels, self.test_data, self.test_labels = self.create_in_memory_dataset()

        # Step 3: Create model
        model = Net()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-4)  
        train_loss,val_loss = train_model(epochs=self.epochs, model=model, criterion=criterion, optimizer=optimizer, 
                train_data=self.train_data, train_labels=self.train_labels, val_data=self.val_data, val_labels=self.val_labels, 
               batch_size=self.batch_size)
        return train_loss, val_loss
        

        
if __name__ == "__main__":
    config_path = "experiments/config_files/config.json"
    config_file = load_json(config_path)
    run = Run(transform=transforms.ToTensor(), config_file=config_file)
    train_loss,val_loss= run.run_model()
    # Plot the train loss accuracy vs epochs


    