import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from PIL import Image
import glob
import torch
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class MNISTVarrezDataset:
    """
    Dataset class for MNIST-varrez with train/test folders.
    """
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __getitem__(self, index):
        # Load image and label
        image_path = self.image_paths[index]
        label = int(image_path.split("/")[-2])  # Class label from folder name
        image = Image.open(image_path).convert("L")  # Convert to grayscale

        # Apply transformation (e.g., resizing, converting to tensor)
        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.image_paths)

def get_data_loaders(data_dir, batch_size):
    """
    Create DataLoaders for MNIST-varrez resizing all images to 28x28.
    Args:
        data_dir (str): Path to MNIST-varrez dataset root.
        batch_size (int): Batch size for DataLoader.

    Returns:
        tuple: Train, validation, and test DataLoaders.
    """
    # Define transformations
    train_transform = transforms.Compose([
        #transforms.RandomRotation(45),
        #transforms.RandomHorizontalFlip(p=0.5),
        #transforms.RandomVerticalFlip(p=0.5),
        #transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.5, 2.0)),
        transforms.Resize((28, 28)),  # Resize all images to 28x28
        transforms.ToTensor()        # Convert to tensor
    ])
    val_test_transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor()         # Convert to tensor
    ])

    # Load train data
    train_image_paths = glob.glob(f"{data_dir}/train/*/*.png")
    train_dataset = MNISTVarrezDataset(train_image_paths, transform=train_transform)

    # Split train dataset into train and validation sets
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    # Wrap validation set with test transform
    val_dataset = MNISTVarrezDataset([train_image_paths[i] for i in val_subset.indices], transform=val_test_transform)

    # Load test data
    test_image_paths = glob.glob(f"{data_dir}/test/*/*.png")
    test_dataset = MNISTVarrezDataset(test_image_paths, transform=val_test_transform)

    # Create DataLoaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Define constants
    data_dir = "/Users/irizabaranyanka/Desktop/Period_2/Deep_Learning/dl_assignment_3/mnist-varres"  # Path to MNIST-varrez dataset
    batch_size = 16

    # Get DataLoaders
    train_loader, val_loader, test_loader = get_data_loaders(data_dir, batch_size)

    # Fetch a batch of images and labels from the train_loader
    images, labels = next(iter(train_loader))

    # Plot a 3x3 grid of images
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    fig.suptitle("Resized Data Examples (28x28)", fontsize=16)
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            img = images[i].squeeze(0).numpy()  # Remove channel dimension and convert to numpy
            ax.imshow(img, cmap="gray")
            ax.set_title(f"Label: {labels[i]}")
            ax.axis("off")
    plt.tight_layout()
    plt.show()
