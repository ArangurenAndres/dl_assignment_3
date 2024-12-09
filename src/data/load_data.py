import ssl
import urllib.request
ssl._create_default_https_context = ssl._create_unverified_context
import torchvision
import torch
import torchvision.transforms as transforms
import json
from torch.utils.data import DataLoader, TensorDataset
import pickle
import os

## Question 7. Part 1 , create Train, val datasets 

## Load the  config json file
def load_json(file_path):
    with open(file_path,'r') as file:
        data = json.load(file)
    return data
import os
import pickle

def save_list(data_list, results_path, exp_name, filename):
    """
    Save a list to a file in a specific folder using pickle.

    Args:
        data_list (list): The list to save.
        results_path (str): The root folder for saving results.
        exp_name (str): The experiment folder name.
        filename (str): The name of the file (e.g., 'output.pkl').
    """
    # Ensure the experiment folder exists
    exp_path = os.path.join(results_path, exp_name)
    os.makedirs(exp_path, exist_ok=True)

    # Full path to the file
    file_path = os.path.join(exp_path, filename)

    # Save the list using pickle
    with open(file_path, 'wb') as f:
        pickle.dump(data_list, f)

    print(f"List saved to {file_path}")


class Data:
    def __init__(self, batch_size=1, augment=False):
        self.batch_size = batch_size
        self.trainloader = None
        self.testloader = None
        self.train_data, self.train_labels = None, None
        self.val_data, self.val_labels = None, None
        self.test_data, self.test_labels = None, None
        self.augment=augment
        
        self.transform_train = transforms.Compose([
                transforms.RandomRotation(20),  #Random rotation up to 20 degrees
                transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
                transforms.RandomCrop(28, padding=4),  # Random cropping with padding
                transforms.ToTensor()  # Convert to tensor
                    ])
        self.transform_test = transforms.Compose([
                transforms.ToTensor() 
                    ])
        self.transform = transforms.ToTensor()

    def load_data(self):
        # Load training dataset
        if self.augment:
            train = torchvision.datasets.MNIST(
                root="./data", train=True, download=True, transform=self.transform_train
            )

            # Load the test dataset
            test = torchvision.datasets.MNIST(
                root="./data", train=False, download=True, transform=self.transform_test
            )
        else:
            train = torchvision.datasets.MNIST(
                root="./data", train=True, download=True, transform=self.transform
            )
            # Load the test dataset
            test = torchvision.datasets.MNIST(
                root="./data", train=False, download=True, transform=self.transform
            )
        ## Once loaded the datasets create the dataloaders

        # Create train dataloader, set shuffle to true, this will help when performing the train validation split
        self.trainloader = torch.utils.data.DataLoader(
            train, batch_size=self.batch_size, shuffle=True, num_workers=2
        )
        # Create test dataloader
        self.testloader = torch.utils.data.DataLoader(
            test, batch_size=self.batch_size, shuffle=True, num_workers=2
        )
        return self.trainloader, self.testloader

    def create_in_memory_dataset(self):
        # Create empty lists for train_data and train_labels
        train_data, train_labels = [], []
        # Iterate over the dataloader
        for data, label in self.trainloader:
            #Append both data (image) and label
            train_data.append(data)
            train_labels.append(label)

        # Convert the lists of tensors into a four dimensional tensor of dimension (60000,1,28,28)
        train_data = torch.cat(train_data)  # Concatenate images into one tensor
        print(f"Shape",train_data.shape)
        # Apply the same to labels
        train_labels = torch.cat(train_labels)  # Concatenate labels into one tensor

        # Perform train-validation split as described in the guide we will take the first 50000 instances for training
        # Reamining 10000 will be used for validation
        self.train_data, self.val_data = train_data[:50000], train_data[50000:]
        self.train_labels, self.val_labels = train_labels[:50000], train_labels[50000:]
        # Apply same procedure described above to test dataset
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
    data = Data(batch_size=64, augment=True)

    # Step 1: Load the train and test DataLoader objects
    train_loader, test_loader = data.load_data()
    ## Question 7. Part 1

    # Step 2: Create in-memory datasets
    train_data, train_labels, val_data, val_labels, test_data, test_labels = data.create_in_memory_dataset()

    print(f"Train data shape: {train_data.shape}, Train labels shape: {train_labels.shape}")
    print(f"Validation data shape: {val_data.shape}, Validation labels shape: {val_labels.shape}")
    print(f"Test data shape: {test_data.shape}, Test labels shape: {test_labels.shape}")



    
