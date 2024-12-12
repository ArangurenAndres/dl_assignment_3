import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

def train_model_with_resolutions(epochs=1, model=None, criterion=None, optimizer=None, batch_size=None,
                                  data_32_train=None, labels_32_train=None, data_32_val=None, labels_32_val=None,
                                  data_48_train=None, labels_48_train=None, data_48_val=None, labels_48_val=None,
                                  data_64_train=None, labels_64_train=None, data_64_val=None, labels_64_val=None):
    """
    Training loop with an inner loop over resolutions (32x32, 48x48, 64x64).
    Tracks per-resolution loss as well as combined loss and accuracy.
    """
    # Initialize lists to store losses and accuracies
    train_loss_per_res = {'32x32': [], '48x48': [], '64x64': []}
    val_loss_per_res = {'32x32': [], '48x48': [], '64x64': []}
    val_accuracy_per_res = {'32x32': [], '48x48': [], '64x64': []}
    combined_train_loss = []
    combined_val_loss = []
    combined_val_accuracy = []

    for epoch in tqdm(range(epochs), desc="Training Progress", unit="epoch"):
        model.train()
        running_loss = {'32x32': 0.0, '48x48': 0.0, '64x64': 0.0}
        num_batches = {'32x32': len(data_32_train) // batch_size, 
                       '48x48': len(data_48_train) // batch_size, 
                       '64x64': len(data_64_train) // batch_size}

        # Training loop over resolutions
        for resolution, (data_train, labels_train) in zip(['32x32', '48x48', '64x64'],
                                                          [(data_32_train, labels_32_train),
                                                           (data_48_train, labels_48_train),
                                                           (data_64_train, labels_64_train)]):
            with tqdm(total=num_batches[resolution], desc=f"Epoch {epoch + 1}/{epochs} - Training ({resolution})", position=1, leave=False) as pbar_train:
                for i in range(num_batches[resolution]):
                    start_idx = i * batch_size
                    end_idx = start_idx + batch_size

                    inputs = data_train[start_idx:end_idx]
                    targets = labels_train[start_idx:end_idx]

                    optimizer.zero_grad()

                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    running_loss[resolution] += loss.item()
                    pbar_train.update(1)

            avg_loss = running_loss[resolution] / num_batches[resolution]
            train_loss_per_res[resolution].append(avg_loss)

        # Combined training loss
        combined_loss = sum(running_loss.values()) / sum(num_batches.values())
        combined_train_loss.append(combined_loss)

        # Validation loop
        model.eval()
        running_val_loss = {'32x32': 0.0, '48x48': 0.0, '64x64': 0.0}
        correct_predictions = {'32x32': 0, '48x48': 0, '64x64': 0}
        total_samples = {'32x32': len(data_32_val), '48x48': len(data_48_val), '64x64': len(data_64_val)}

        with torch.no_grad():
            for resolution, (data_val, labels_val) in zip(['32x32', '48x48', '64x64'],
                                                          [(data_32_val, labels_32_val),
                                                           (data_48_val, labels_48_val),
                                                           (data_64_val, labels_64_val)]):
                val_batches = len(data_val) // batch_size
                with tqdm(total=val_batches, desc=f"Epoch {epoch + 1}/{epochs} - Validation ({resolution})", position=2, leave=False) as pbar_val:
                    for i in range(val_batches):
                        start_idx = i * batch_size
                        end_idx = start_idx + batch_size

                        inputs = data_val[start_idx:end_idx]
                        targets = labels_val[start_idx:end_idx]

                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        running_val_loss[resolution] += loss.item()

                        _, predicted = torch.max(outputs.data, 1)
                        correct_predictions[resolution] += (predicted == targets).sum().item()

                        pbar_val.update(1)

                avg_val_loss = running_val_loss[resolution] / val_batches
                val_loss_per_res[resolution].append(avg_val_loss)
                accuracy = correct_predictions[resolution] / total_samples[resolution]
                val_accuracy_per_res[resolution].append(accuracy)

            combined_val_loss.append(sum(running_val_loss.values()) / sum(len(data) // batch_size for data in [data_32_val, data_48_val, data_64_val]))
            combined_accuracy = sum(correct_predictions.values()) / sum(total_samples.values())
            combined_val_accuracy.append(combined_accuracy)

        # Print epoch summary
        print(f"Epoch {epoch + 1}/{epochs} - Combined Training Loss: {combined_train_loss[-1]:.4f}, Combined Validation Loss: {combined_val_loss[-1]:.4f}, Combined Validation Accuracy: {combined_val_accuracy[-1]:.4f}")

    return train_loss_per_res, val_loss_per_res, val_accuracy_per_res, combined_train_loss, combined_val_loss, combined_val_accuracy
