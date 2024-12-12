import os
import glob
import torch
from torchvision.io import read_image
from torchvision.transforms import ToTensor
from PIL import Image  # Import PIL for reading images
import numpy as np

def load_data_by_resolution(root_dir, val_split=0.2, random_seed=42):
    # Initialize three empty lists for each resolution
    data_32, data_48, data_64 = [], [], []
    labels_32, labels_48, labels_64 = [], [], []

    # Iterate over each number subdirectory (0-9)
    for number in range(10):
        number_dir = os.path.join(root_dir, 'train', str(number))
        paths = glob.glob(os.path.join(number_dir, '*.png'))  # Get all .png images in the folder

        # Iterate over each image path
        for path in paths:
            img = Image.open(path)  # Read the image as a PIL Image
            img_tensor = ToTensor()(img)  # Convert to tensor   
            path_sections = path.split('/')
            label = int(path_sections[-2])  # Extract the number (label) from the path
            label_tensor = torch.tensor(label)  # Convert the label to tensor

            # Determine resolution based on image size
            if img_tensor.shape[-2:] == (32, 32):  # 32x32 resolution
                data_32.append(img_tensor)
                labels_32.append(label_tensor)
            elif img_tensor.shape[-2:] == (48, 48):  # 48x48 resolution
                data_48.append(img_tensor)
                labels_48.append(label_tensor)
            elif img_tensor.shape[-2:] == (64, 64):  # 64x64 resolution
                data_64.append(img_tensor)
                labels_64.append(label_tensor)

    # Convert lists to tensors
    data_32_tensor = torch.stack(data_32)  # Stack into a tensor
    data_48_tensor = torch.stack(data_48)  # Stack into a tensor
    data_64_tensor = torch.stack(data_64)  # Stack into a tensor
    labels_32_tensor = torch.stack(labels_32)  # Stack into a tensor
    labels_48_tensor = torch.stack(labels_48)  # Stack into a tensor
    labels_64_tensor = torch.stack(labels_64)  # Stack into a tensor

    # Verify label consistency
    for label_list, data_list in zip([labels_32, labels_48, labels_64], [data_32, data_48, data_64]):
        assert len(label_list) == len(data_list), "Label count does not match data count"

    # Split data into train and validation sets using PyTorch's random_split
    def split_data(data, labels, val_split=0.2, random_seed=42):
        num_samples = len(data)
        indices = torch.randperm(num_samples, generator=torch.Generator().manual_seed(random_seed))
        split = int(np.floor(val_split * num_samples))
        train_indices, val_indices = indices[split:], indices[:split]
        train_data, val_data = data[train_indices], data[val_indices]
        train_labels, val_labels = labels[train_indices], labels[val_indices]
        return train_data, train_labels, val_data, val_labels

    data_32_train, labels_32_train, data_32_val, labels_32_val = split_data(data_32_tensor, labels_32_tensor, val_split, random_seed)
    data_48_train, labels_48_train, data_48_val, labels_48_val = split_data(data_48_tensor, labels_48_tensor, val_split, random_seed)
    data_64_train, labels_64_train, data_64_val, labels_64_val = split_data(data_64_tensor, labels_64_tensor, val_split, random_seed)

    return (data_32_train, labels_32_train, data_32_val, labels_32_val,
            data_48_train, labels_48_train, data_48_val, labels_48_val,
            data_64_train, labels_64_train, data_64_val, labels_64_val)

# Example usage:
root_dir = 'mnist_var'
(data_32_train, labels_32_train, data_32_val, labels_32_val,
 data_48_train, labels_48_train, data_48_val, labels_48_val,
 data_64_train, labels_64_train, data_64_val, labels_64_val) = load_data_by_resolution(root_dir)

# Display shapes

# Display shapes
print(f'Resolution 32x32 Train Tensor Shape: {data_32_train.shape}, Labels Shape: {labels_32_train.shape}')
print(f'Resolution 32x32 Validation Tensor Shape: {data_32_val.shape}, Labels Shape: {labels_32_val.shape}')
print(f'Resolution 48x48 Train Tensor Shape: {data_48_train.shape}, Labels Shape: {labels_48_train.shape}')
print(f'Resolution 48x48 Validation Tensor Shape: {data_48_val.shape}, Labels Shape: {labels_48_val.shape}')
print(f'Resolution 64x64 Train Tensor Shape: {data_64_train.shape}, Labels Shape: {labels_64_train.shape}')
print(f'Resolution 64x64 Validation Tensor Shape: {data_64_val.shape}, Labels Shape: {labels_64_val.shape}')