import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split, Subset
import torch
import matplotlib.pyplot as plt
import ssl
import urllib.request
ssl._create_default_https_context = ssl._create_unverified_context

class TransformData:

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.dataset[index]
        # Apply transform only if image is not already a tensor
        if not isinstance(image, torch.Tensor) and self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.dataset)

## Modify the transformation operation
def get_train_val_dataloaders(batch_size):
    train_transform = transforms.Compose([
        #transforms.RandomRotation(45),
        transforms.RandomHorizontalFlip(p=0.5),
        #transforms.RandomVerticalFlip(p=0.5),
        #transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.5, 2.0)),
        transforms.ToTensor()
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Load the raw dataset without applying transformations
    raw_dataset = MNIST(root="./data", train=True, download=True)

    # Split into training and validation subsets
    train_size = 50000
    val_size = len(raw_dataset) - train_size
    train_indices, val_indices = random_split(raw_dataset, [train_size, val_size])

    # Wrap subsets with TransformData to apply transformations dynamically
    train_dataset = TransformData(Subset(raw_dataset, train_indices.indices), transform=train_transform)
    val_dataset = TransformData(Subset(raw_dataset, val_indices.indices), transform=val_transform)

    # Load the test dataset
    test_dataset = MNIST(root="./data", train=False, download=True, transform=val_transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_train_val_dataloaders(batch_size=16)
        # Fetch a batch of images and labels from the train_loader
    images, labels = next(iter(train_loader))  # Fix: Use iter() to create an iterator

    # Plot a 3x3 grid of images
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    fig.suptitle("Data Augmentation Examples", fontsize=16)
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            img = images[i].squeeze(0).numpy()  # Remove channel dimension and convert to numpy
            ax.imshow(img, cmap="gray")
            ax.set_title(f"Label: {labels[i].item()}")
            ax.axis("off")
    plt.tight_layout()
    plt.show()
