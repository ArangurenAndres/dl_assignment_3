import ssl
import urllib.request

ssl._create_default_https_context = ssl._create_unverified_context
import os
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from data.load_data import load_json, save_list
from data.load_dataloader_part3 import get_data_loaders  # Import your new data loader
from models.model import Net
from train.train_loaders import train_model_with_dataloaders


class Run():
    def __init__(self, config_path=None, save_metrics=False):
        # Assign config file values to class attributes
        self.config_file = load_json(config_path)
        self.data_dir = self.config_file["data_dir"]  # New: Path to MNIST-varrez dataset
        self.augmentation = self.config_file["augmentation"]
        self.epochs = self.config_file["epochs"]
        self.batch_size = self.config_file["batch_size"]
        self.learning_rate = self.config_file["lr"]
        self.results_path = self.config_file["results_path"]
        self.exp_name = self.config_file["exp_name"]
        self.save_metrics = save_metrics

    def run_model(self):
        # Step 1: Load the train, validation, and test DataLoader objects
        print(f"2. Loading train, validation, and test datasets...")
        train_loader, val_loader, test_loader = get_data_loaders(data_dir=self.data_dir, batch_size=self.batch_size)

        # Step 2: Create model
        print(f"3. Loading model...")
        model = Net()

        # Define loss function
        criterion = nn.CrossEntropyLoss()

        # Define optimizer
        optimizer = optim.SGD(model.parameters(), lr=self.learning_rate)

        # Train the model
        print(f"4. Training model over {self.epochs} epochs...")
        train_loss, val_loss, val_accuracy = train_model_with_dataloaders(
            epochs=self.epochs,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader
        )

        # Save train and validation loss
        if self.save_metrics:
            print(f"5. Saving training and validation losses...")
            save_list(train_loss, self.results_path, self.exp_name, "train_loss.pkl")
            save_list(val_loss, self.results_path, self.exp_name, "val_loss.pkl")
            save_list(val_accuracy, self.results_path, self.exp_name, "val_accuracy.pkl")
            print(f"6. Metrics were saved. Training is concluded :)")
        else:
            print(f"Training concluded.")

        return train_loss, val_loss, val_accuracy


if __name__ == "__main__":
    # Specify the path of the config file you are implementing
    config_path = "/experiments/config_files_part3/config_baseline_part3.json"
    print(f"1. Loading model hyperparameters")
    run = Run(config_path=config_path, save_metrics=True)

    # Save the train and validation loss
    train_loss, val_loss, val_accuracy = run.run_model()
