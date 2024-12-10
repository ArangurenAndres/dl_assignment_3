from tqdm import tqdm
import torch.optim as optim
import torch
import numpy as np
import torch.nn.functional as F


def train_model_with_dataloaders(epochs=1, model=None, criterion=None, optimizer=None,
                                 train_loader=None, val_loader=None):
    """
    Training function using PyTorch DataLoaders for train and validation datasets.
    Args:
        epochs (int): Number of training epochs.
        model (torch.nn.Module): The model to train.
        criterion (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.

    Returns:
        tuple: Lists of training losses, validation losses, and validation accuracies for each epoch.
    """
    # Initialize empty lists to store training loss, validation loss, and accuracy
    run_loss = []
    val_loss = []
    val_accuracy = []

    # Loop over epochs
    for epoch in tqdm(range(epochs), desc="Training Progress", unit="epoch"):
        model.train()  # Set the model in training mode
        running_loss = 0.0

        # Create a progress bar for the training batches
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{epochs} - Training", position=1,
                  leave=False) as pbar_train:
            for inputs, labels in train_loader:
                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                pbar_train.update(1)  # Update the batch progress bar

        # Compute average training loss
        avg_train_loss = running_loss / len(train_loader)
        run_loss.append(avg_train_loss)

        # Validation loop
        model.eval()  # Set the model in evaluation mode
        with torch.no_grad():
            val_running_loss = 0.0
            correct_preds = 0
            total_preds = 0

            # Create a progress bar for the validation batches
            with tqdm(total=len(val_loader), desc=f"Epoch {epoch + 1}/{epochs} - Validation", position=2,
                      leave=False) as pbar_val:
                for inputs, labels in val_loader:
                    # Forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_running_loss += loss.item()

                    # Compute validation accuracy
                    preds = torch.argmax(outputs, dim=1)
                    correct_preds += (preds == labels).sum().item()
                    total_preds += labels.size(0)

                    pbar_val.update(1)  # Update the validation progress bar

            # Compute average validation loss
            avg_val_loss = val_running_loss / len(val_loader)
            val_loss.append(avg_val_loss)

            # Compute validation accuracy
            val_accuracy_temp = correct_preds / total_preds
            val_accuracy.append(val_accuracy_temp)

        # Print progress for the current epoch
        tqdm.write(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f}, "
                   f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy_temp:.4f}")

    print("Finished Training")
    return run_loss, val_loss, val_accuracy