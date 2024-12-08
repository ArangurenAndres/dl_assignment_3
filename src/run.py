import ssl
import urllib.request
ssl._create_default_https_context = ssl._create_unverified_context
import os
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from data.load_data import Data, load_json, save_list
from models.model import Net
from train.train import train_model


class Run(Data):
    def __init__(self, transform=None, config_file=None):
        # Assign config file values to class attributes
        self.config_file = config_file
        self.epochs = self.config_file["epochs"]
        self.batch_size = self.config_file["batch_size"]
        self.learning_rate = self.config_file["lr"]
        self.results_path = self.config_file["results_path"]
        self.exp_name = self.config_file["exp_name"]
        super().__init__(batch_size=self.batch_size, transform=transform)

    def run_model(self):
        # Step 1: Load the train and test DataLoader objects
        self.train_loader, self.test_loader = self.load_data()

        # Step 2: Create in-memory datasets
        self.train_data, self.train_labels, self.val_data, self.val_labels, self.test_data, self.test_labels = self.create_in_memory_dataset()

        # Step 3: Create model
        print(f"Loading model")
        model = Net()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        print(f"Training model over: {self.epochs} epochs")  
        train_loss,val_loss, val_accuracy = train_model(epochs=self.epochs, model=model, criterion=criterion, optimizer=optimizer, 
                train_data=self.train_data, train_labels=self.train_labels, val_data=self.val_data, val_labels=self.val_labels, 
               batch_size=self.batch_size)
        #save train and validation loss
        print(f"Saving training and validation losses")
        save_list(train_loss, self.results_path, self.exp_name,"train_loss.pkl")
        save_list(val_loss, self.results_path, self.exp_name,"val_loss.pkl")
        save_list(val_accuracy, self.results_path, self.exp_name,"val_accuracy.pkl")

        return train_loss, val_loss, val_accuracy
       
if __name__ == "__main__":
    config_path = "experiments/config_files/config_baseline.json"
    print(f"Loading model hpyerparameters")
    config_file = load_json(config_path)
    run = Run(transform=transforms.ToTensor(), config_file=config_file)
    #save the train, val loss 
    train_loss,val_loss= run.run_model()


    