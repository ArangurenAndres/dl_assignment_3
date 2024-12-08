import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle

def plot_losses_from_pkl(directory, title="Loss Curves"):
    """
    Plots loss curves stored as .pkl files in the given directory.
    
    Args:
        directory (str): Path to the directory containing .pkl files with loss data.
        title (str): Title for the plot.
    """
    # Set Seaborn style
    sns.set(style="whitegrid")
    color_palette = sns.color_palette("husl", as_cmap=False)

    # Prepare figure
    plt.figure(figsize=(10, 6))

    # Iterate through files in the directory
    for idx, file in enumerate(sorted(os.listdir(directory))):
        if file.endswith(".pkl"):
            # Load the losses
            with open(os.path.join(directory, file), "rb") as f:
                loss_data = pickle.load(f)

            # Plot the losses
            label = os.path.splitext(file)[0]  # Use filename (without extension) as label
            plt.plot(loss_data, label=label, color=color_palette[idx % len(color_palette)])

    # Add legend, labels, and title
    plt.legend(title="Experiments", loc="best")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(title)

    # Show the plot
    plt.tight_layout()
    plt.show()