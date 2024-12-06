import ssl
import urllib.request
ssl._create_default_https_context = ssl._create_unverified_context
import torchvision
import torch
import torchvision.transforms as transforms
import json
from torch.utils.data import DataLoader, TensorDataset

## Question 7. Part 1 , create Train, val datasets 

## Load the  config json file
def load_json(file_path):
    with open(file_path,'r') as file:
        data = json.load(file)
    return data

class Data:
    def __init__(self, batch_size=1, transform=None):
        self.batch_size = batch_size
        self.transform = transform
        self.trainloader = None
        self.testloader = None
        self.train_data, self.train_labels = None, None
        self.val_data, self.val_labels = None, None
        self.test_data, self.test_labels = None, None

    def load_data(self):
        train = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=self.transform
        )
        self.trainloader = torch.utils.data.DataLoader(
            train, batch_size=self.batch_size, shuffle=True, num_workers=2
        )
        test = torchvision.datasets.MNIST(
            root="./data", train=False, download=True, transform=self.transform
        )
        self.testloader = torch.utils.data.DataLoader(
            test, batch_size=self.batch_size, shuffle=True, num_workers=2
        )
        return self.trainloader, self.testloader

    def create_in_memory_dataset(self):
        train_data, train_labels = [], []
        for data, label in self.trainloader:
            train_data.append(data)
            train_labels.append(label)

        # Convert the lists to tensors
        train_data = torch.cat(train_data)  # Concatenate images into one tensor
        train_labels = torch.cat(train_labels)  # Concatenate labels into one tensor

        # Perform train-validation split
        self.train_data, self.val_data = train_data[:50000], train_data[50000:]
        self.train_labels, self.val_labels = train_labels[:50000], train_labels[50000:]

        test_data, test_labels = [], []
        for data, label in self.testloader:
            test_data.append(data)
            test_labels.append(label)

        # Convert test data and labels to tensors
        self.test_data = torch.cat(test_data)
        self.test_labels = torch.cat(test_labels)

        return self.train_data, self.train_labels, self.val_data, self.val_labels, self.test_data, self.test_labels
    def get_dataloaders(self):
        if self.train_data is None or self.val_data is None:
            raise ValueError("In-memory datasets are not initialized. Please call `create_in_memory_dataset()` first.")
        train_dataset = TensorDataset(self.train_data, self.train_labels)
        val_dataset = TensorDataset(self.val_data, self.val_labels)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader


if __name__ == "__main__":
    # Instantiate the class with desired parameters
    data = Data(batch_size=64, transform=transforms.ToTensor())

    # Step 1: Load the train and test DataLoader objects
    train_loader, test_loader = data.load_data()
    ## Question 7. Part 1

    # Step 2: Create in-memory datasets
    train_data, train_labels, val_data, val_labels, test_data, test_labels = data.create_in_memory_dataset()

    print(f"Train data shape: {train_data.shape}, Train labels shape: {train_labels.shape}")
    print(f"Validation data shape: {val_data.shape}, Validation labels shape: {val_labels.shape}")
    print(f"Test data shape: {test_data.shape}, Test labels shape: {test_labels.shape}")



    
