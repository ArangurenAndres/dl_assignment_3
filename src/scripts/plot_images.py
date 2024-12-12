import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Path to the directory containing MNIST images
data_dir = 'mnist_var/train/5'

# Function to load and filter images based on size
def load_images(data_dir, target_size=(64, 64),resize_shape = (28,28)):
    images = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.png'):  # assuming the images are in PNG format
            img_path = os.path.join(data_dir, filename)
            img = Image.open(img_path)
            if img.size == target_size:
                img_resized = img.resize(resize_shape)
                images.append(np.array(img_resized))  # convert PIL image to numpy array
    return images

# Load images with the desired dimensions
images = load_images(data_dir=data_dir,target_size=(64, 64),resize_shape = (28,28))

# Plotting 9 images in a 3x3 grid
fig, axs = plt.subplots(3, 3, figsize=(8, 8))
for i, ax in enumerate(axs.flat):
    if i < len(images):
        ax.imshow(images[i], cmap='gray')
    ax.axis('off')
plt.tight_layout()
plt.show()
