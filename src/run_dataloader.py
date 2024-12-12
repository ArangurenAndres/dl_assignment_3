import ssl
import urllib.request
ssl._create_default_https_context = ssl._create_unverified_context
import os
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from data.load_data import load_json, save_list, save_model
#from data.load_data_dataloaders import get_train_val_dataloaders
from data.load_data_var import get_train_val_dataloaders
#from models.model import Net
from models.model_var import Net_var
from train.train_loaders import train_model_with_dataloaders

class Run():
    def __init__(self,config_path = None, save_metrics = False,save_model=True):
        # Assign config file values to class attributes
        self.config_file = load_json(config_path)
        self.augmentation = self.config_file["augmentation"]
        self.epochs = self.config_file["epochs"]
        self.batch_size = self.config_file["batch_size"]
        self.learning_rate = self.config_file["lr"]
        self.results_path = self.config_file["results_path"]
        self.model_path = self.config_file["model_path"]
        self.exp_name = self.config_file["exp_name"]
        self.target_size = self.config_file["target_size"]   
        self.save_model = save_model  
        self.save_metrics = save_metrics   

    def run_model(self):
        # Step 1: Load the train and test DataLoader objects
        print(f"2. Loading train and validation datasets...")
        train_loader,val_loader,_ = get_train_val_dataloaders(batch_size=self.batch_size,target_size=(self.target_size,self.target_size))
        # Step 2: Create model
        print(f"3. Loading model...")
        model = Net_var()
        # Define loss function
        criterion = nn.CrossEntropyLoss()
        #for now lets not include momentum nor weight decay
        # Define optimizer
        optimizer = optim.SGD(model.parameters(), lr=self.learning_rate)
        #optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        print(f"4. Training model over: {self.epochs} epochs...")  
        train_loss,val_loss, val_accuracy = train_model_with_dataloaders(epochs=self.epochs, model=model, criterion=criterion, optimizer=optimizer, 
                                 train_loader=train_loader, val_loader=val_loader)
        # Aftre training save the model
        if self.save_model:
            print(f"Saving model weights...")
            save_model(model,self.model_path, self.exp_name,"model_weights.pth")
        #save train and validation loss
        if self.save_metrics:
            print(f"5.Saving training and validation losses...")
            save_list(train_loss, self.results_path, self.exp_name,"train_loss.pkl")
            save_list(val_loss, self.results_path, self.exp_name,"val_loss.pkl")
            save_list(val_accuracy, self.results_path, self.exp_name,"val_accuracy.pkl")
            print(f"6. Metrics were saved. Training is concluded :)")
        else:
            print(f"Training concluded.")
    
        return train_loss, val_loss, val_accuracy
       
if __name__ == "__main__":
    # In this line specify the path of the config file you are implementing
    config_path = "experiments/config_files_var/config_baseline.json"
    print(f"1. Loading model hpyerparameters") 
    run = Run(config_path=config_path,save_metrics=True)
    #save the train, val loss 
    train_loss,val_loss, val_accuracy= run.run_model()


     