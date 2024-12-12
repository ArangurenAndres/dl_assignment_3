import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
def plot_losses_accuracies(train_loss_path, val_loss_path, val_accuracy_path):
    # Load the data
    with open(train_loss_path, 'rb') as f:
        train_loss_per_res = pickle.load(f)

    with open(val_loss_path, 'rb') as f:
        val_loss_per_res = pickle.load(f)

    with open(val_accuracy_path, 'rb') as f:
        val_accuracy_per_res = pickle.load(f)
        print(val_accuracy_per_res)

    # Calculate and print mean validation accuracy for each resolution
    final_val_acc = []
    for resolution, accuracies in val_accuracy_per_res.items():
        last_accuracy = accuracies[-1]
        final_val_acc.append(last_accuracy)
    mean_acc_res = np.mean(final_val_acc)
    print(f"Mean Validation Accuracy for: {mean_acc_res:.4f}")

    # Set Seaborn style
    sns.set(style="whitegrid")

    # Define marker styles and colors for resolutions
    styles = {
        '32x32': {'marker': 'o', 'color_train': 'blue', 'color_val': 'orange'},
        '48x48': {'marker': 's', 'color_train': 'green', 'color_val': 'red'},
        '64x64': {'marker': 'D', 'color_train': 'purple', 'color_val': 'brown'}
    }

    # Plotting
    fig, axs = plt.subplots(2, 1, figsize=(12, 14))

    # Plot 1: Train Loss and Validation Loss
    for resolution, style in styles.items():
        sns.lineplot(
            x=range(len(train_loss_per_res[resolution])), 
            y=train_loss_per_res[resolution], 
            marker=style['marker'], label=f'Train Loss ({resolution})', 
            ax=axs[0], color=style['color_train']
        )
        sns.lineplot(
            x=range(len(val_loss_per_res[resolution])), 
            y=val_loss_per_res[resolution], 
            marker=style['marker'], label=f'Val Loss ({resolution})', 
            ax=axs[0], color=style['color_val']
        )

    axs[0].set_title('Train and Validation Loss per Resolution', fontsize=14)
    axs[0].set_xlabel('Epoch', fontsize=12)
    axs[0].set_ylabel('Loss', fontsize=12)
    axs[0].legend(fontsize=10)

    # Plot 2: Validation Accuracy
    for resolution, style in styles.items():
        sns.lineplot(
            x=range(len(val_accuracy_per_res[resolution])), 
            y=val_accuracy_per_res[resolution], 
            marker=style['marker'], label=f'Val Accuracy ({resolution})', 
            ax=axs[1], color=style['color_val']
        )

    axs[1].set_title('Validation Accuracy per Resolution', fontsize=14)
    axs[1].set_xlabel('Epoch', fontsize=12)
    axs[1].set_ylabel('Accuracy', fontsize=12)
    axs[1].legend(fontsize=10)

    plt.tight_layout()
    plt.show()

# Example usage:
# plot_losses_accuracies_seaborn('train_loss.pkl', 'val_loss.pkl', '

# Example usage:
# plot_losses_accuracies_seaborn('train_loss.pkl', 'val_loss.pkl', 'val_accuracy.pkl')

# Example usage:
dir_path = "experiments/results/results_var_res/exp_1_avg"
train_loss_path = os.path.join(dir_path,"train_loss_per_res.pkl")
val_loss_path = os.path.join(dir_path,"val_loss_per_res.pkl")
val_accuracy_path = os.path.join(dir_path,"val_accuracy_per_res.pkl")
plot_losses_accuracies(train_loss_path=train_loss_path, val_loss_path=val_loss_path, val_accuracy_path=val_accuracy_path)
