import matplotlib.pyplot as plt


def plot_data_examples(images,labels):
    # Step 3: Plot a grid of images
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