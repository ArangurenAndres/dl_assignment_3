import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torch
import matplotlib.pyplot as plt

class TransformData:
    def __init__(self, dataset, transform=None):
        """
        A wrapper to dynamically apply transformations on a dataset.
        """
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.dataset[index]
        # Apply transform only if the image is not already a tensor
        if not isinstance(image, torch.Tensor) and self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.dataset)


def get_train_val_dataloaders(batch_size, target_size=None):
    """
    Prepares train, validation, and test dataloaders with dynamic transformations.

    Args:
        data_dir (str): Path to the dataset directory.
        batch_size (int): Batch size for the dataloaders.

    Returns:
        train_loader, val_loader, test_loader: DataLoader objects for training, validation, and testing.
    """
    # Define transformations for training and validation
    train_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])
    val_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    # Load datasets
    init_train_dataset = ImageFolder(root="mnist_var/train")
    init_test_dataset = ImageFolder(root=f"mnist_var/test", transform=val_transform)

    # Split training dataset into training and validation subsets
    train_size = int(0.8 * len(init_train_dataset))
    val_size = len(init_train_dataset) - train_size
    train_indices, val_indices = random_split(range(len(init_train_dataset)), [train_size, val_size])

    # Map indices to the original dataset with dynamic transformations
    train_dataset = TransformData(
        dataset=torch.utils.data.Subset(init_train_dataset, train_indices.indices),
        transform=train_transform
    )
    val_dataset = TransformData(
        dataset=torch.utils.data.Subset(init_train_dataset, val_indices.indices),
        transform=val_transform
    )

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(init_test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader




if __name__ == "__main__":
    data_dir = "./mnist_var"  # Path to the custom MNIST folder
    target_size = (32,32)
    train_loader, val_loader, test_loader = get_train_val_dataloaders(batch_size=16,target_size=target_size)
    # Fetch a batch of images and labels from the train_loader
    images, labels = next(iter(train_loader))  # Use iter() to create an iterator


    # Plot a 3x3 grid of images with varying sizes
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    fig.suptitle("Sample Images with Variable Sizes", fontsize=16)

    for i, ax in enumerate(axes.flat):
        if i < len(images):
            print(images[i].shape)
            img = images[i].permute(1, 2, 0).numpy()  # Convert tensor to numpy array (H, W, C)
            ax.imshow(img, cmap="gray")
            ax.set_title(f"Label: {labels[i].item()}")
            ax.axis("off")

    plt.tight_layout()
    plt.show()