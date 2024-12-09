from tqdm import tqdm
import torch.optim as optim
import torch
import numpy as np
import torch.nn.functional as F
## Question 7. training loop that loops voer batches of 16 instances at a time
def train_model(epochs=1, model=None, criterion=None, optimizer=None, 
                train_data=None, train_labels=None, val_data=None, val_labels=None, 
                batch_size=16):
    # Initialize empty list to store the training loss , validation loss and validation accuracy
    run_loss = []
    val_loss = []
    val_accuracy = []
    
    # Loop over epochs we will use tqdm package which allows to see the progressbar in a defined unit 
    for epoch in tqdm(range(epochs), desc="Training Progress", unit="epoch"):
        #set the model in training mode
        model.train()  
        running_loss = 0.0
        
        # Get the number of batches for this question the batch size was defined as 16.
        # thus number of iterations will be just the length of the data divided by the size of the batch
        num_batches = len(train_data) // batch_size
        
        # Create a progress bar for the training batches
        with tqdm(total=num_batches, desc=f"Epoch {epoch + 1}/{epochs} - Training", position=1, leave=False) as pbar_train:
            for i in range(num_batches):
                #Since we are iterating over in memory instances instead of dataloaders, 
                # we will apply indexing to the train and validaiton lists 
                start_idx = i * batch_size
                # move unit based on the batch size 
                end_idx = start_idx + batch_size
                # Get the batch images
                inputs = train_data[start_idx:end_idx]
                # Get the batch labels
                labels = train_labels[start_idx:end_idx]
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs)
                # the loss function is also a parameter for this case cross-entropy loss
                loss = criterion(outputs, labels)
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                pbar_train.update(1)  # Update the batch progress bar

        # Compute average training loss
        avg_train_loss = running_loss / num_batches
        run_loss.append(avg_train_loss)

        # Validation loop
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            val_running_loss = 0.0
            val_num_batches = len(val_data) // batch_size
            correct_preds = 0
            total_preds = 0
            
            # Create a progress bar for the validation batches
            with tqdm(total=val_num_batches, desc=f"Epoch {epoch + 1}/{epochs} - Validation", position=2, leave=False) as pbar_val:
                for i in range(val_num_batches):
                    start_idx = i * batch_size
                    end_idx = start_idx + batch_size
                    inputs = val_data[start_idx:end_idx]
                    labels = val_labels[start_idx:end_idx]

                    # Forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_running_loss += loss.item()

                    # Compute the validation accuracy
                    preds = torch.argmax(outputs,dim=1)
                    correct_preds+=(preds==labels).sum().item()
                    total_preds+=labels.size(0)
                    
                    pbar_val.update(1)  # Update the validation progress bar

            # Compute average validation loss
            avg_val_loss = val_running_loss / val_num_batches
            val_loss.append(avg_val_loss)

            # Compute validation accuracy
            val_accuracy_temp = correct_preds / total_preds
            val_accuracy.append(val_accuracy_temp)

        # Print progress for the current epoch

    print("Finished Training")
    return run_loss, val_loss, val_accuracy
