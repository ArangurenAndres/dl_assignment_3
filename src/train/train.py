import torch.optim as optim
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

## Question 7
## Split the training data into 50000 trianing instances and 10000 validation instances. 
## Write the training loop that loops over batches of 

def train_model(epochs=2, model=None, criterion=None, optimizer=None, 
                train_data=None, train_labels=None, val_data=None, val_labels=None, 
                batch_size=32):

    run_loss = []
    val_loss = []
    for epoch in tqdm(range(epochs), desc="Training Progress", unit="epoch"):
        model.train()  
        running_loss = 0.0
        #Get the number of batches
        num_batches = len(train_data) // batch_size
        for i in range(num_batches):
            # Get the batch index start and end
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            inputs = train_data[start_idx:end_idx]
            labels = train_labels[start_idx:end_idx]
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Compute average training loss
        avg_train_loss = running_loss / num_batches
        run_loss.append(avg_train_loss)

        # Validation loop
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            val_running_loss = 0.0
            val_num_batches = len(val_data) // batch_size
            for i in range(val_num_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                inputs = val_data[start_idx:end_idx]
                labels = val_labels[start_idx:end_idx]

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()

            # Compute average validation loss
            avg_val_loss = val_running_loss / val_num_batches
            val_loss.append(avg_val_loss)

        # Print progress
        tqdm.write(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    print("Finished Training")
    return run_loss, val_loss
