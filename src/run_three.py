import ssl
import urllib.request
ssl._create_default_https_context = ssl._create_unverified_context
import os
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from data.load_data import Data, load_json, save_list, save_model
from models.model_part_three import Net_resolution, count_parameters
from train.train_three import train_model_with_resolutions
from data.load_data_three import load_data_by_resolution


class Run_resolutions():
    def __init__(self, config_path=None, save_metrics=False, save_model=True):
        # Assign config file values to class attributes
        self.config_file = load_json(config_path)
        self.augmentation = self.config_file["augmentation"]
        self.epochs = self.config_file["epochs"]
        self.batch_size = self.config_file["batch_size"]
        self.learning_rate = self.config_file["lr"]
        self.results_path = self.config_file["results_path"]
        self.model_path = self.config_file["model_path"]
        self.exp_name = self.config_file["exp_name"]   
        self.save_model = save_model
        self.save_metrics = save_metrics
        self.root_dir = "mnist_var"
    def run_model(self):
        # Step 1: Load and prepare datasets for different resolutions
        print(f"2. Loading datasets for multiple resolutions...")
        
        data_32_train, labels_32_train, data_32_val, labels_32_val,data_48_train, labels_48_train, data_48_val, labels_48_val,data_64_train, labels_64_train, data_64_val, labels_64_val = load_data_by_resolution(root_dir=self.root_dir)
        print(f'Resolution 32x32 Tensor Shape: {data_32_train.shape}')
        print(f'Resolution 32x32 Labels Shape: {labels_32_train.shape}')
        print(f'Resolution 48x48 Tensor Shape: {data_48_train.shape}')
        print(f'Resolution 32x32 Labels Shape: {labels_48_train.shape}')
        print(f'Resolution 64x64 Tensor Shape: {data_64_train.shape}')
        print(f'Resolution 32x32 Labels Shape: {labels_64_train.shape}')

        # Step 2: Create model
        print(f"3. Loading model...")
        model = Net_resolution(N=81,dropout=0.5)
        num_param = count_parameters(model)
        print(f"Training model with: {num_param} parameters")
        print(model.parameters())
        criterion = nn.CrossEntropyLoss()
        #optimizer = optim.SGD(model.parameters(), lr=self.learning_rate)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-4)  # Adjust weight_decay as needed
        
        print(f"4. Training model over: {self.epochs} epochs...")  
        
        train_loss_per_res, val_loss_per_res, val_accuracy_per_res, combined_train_loss, combined_val_loss, combined_val_accuracy = train_model_with_resolutions(
                                    epochs=self.epochs, model=model, criterion=criterion,
                                    optimizer=optimizer,batch_size=self.batch_size,
                                    data_32_train=data_32_train, labels_32_train=labels_32_train, data_32_val=data_32_val, labels_32_val=labels_32_val,
                                    data_48_train=data_48_train, labels_48_train=labels_48_train, data_48_val=data_48_val, labels_48_val=labels_48_val,
                                    data_64_train=data_64_train, labels_64_train=labels_64_train, data_64_val=data_64_val, labels_64_val=labels_64_val
                                    )

        # Save the model
        if self.save_model:
            print(f"Saving model weights...")
            save_model(model, self.model_path, self.exp_name, "model_weights.pth")

        # Save training and validation losses
        if self.save_metrics:
            print(f"5. Saving training and validation losses...")
            save_list(train_loss_per_res, self.results_path, self.exp_name, "train_loss_per_res.pkl")
            save_list(val_loss_per_res, self.results_path, self.exp_name, "val_loss_per_res.pkl")
            save_list(val_accuracy_per_res, self.results_path, self.exp_name, "val_accuracy_per_res.pkl")
            save_list(combined_train_loss, self.results_path, self.exp_name, "combined_train_loss.pkl")
            save_list(combined_val_loss, self.results_path, self.exp_name, "combined_val_loss.pkl")
            save_list(combined_val_loss, self.results_path, self.exp_name, "combined_val_accuracy.pkl")

            print(f"6. Metrics were saved. Training is concluded :)")
        else:
            print(f"Training concluded.")

        return train_loss_per_res, val_loss_per_res, val_accuracy_per_res,combined_train_loss, combined_val_loss, combined_val_accuracy
       
if __name__ == "__main__":
    # Specify the path of the config file you are implementing
    config_path = "experiments/config_var_res/config_exp_2.json"
    print(f"1. Loading model hyperparameters") 
    run = Run_resolutions(config_path=config_path, save_metrics=True, save_model=True)
    run.run_model()