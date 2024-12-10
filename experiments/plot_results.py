import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle


import pickle
import seaborn as sns
import matplotlib.pyplot as plt
# Use this function to plot the train-validation loss vs epochs graph
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns



class Plot:
    def __init__(self,results_path,exp_name):
        self.results_path = results_path
        self.exp_name = exp_name
        self.files_path = os.path.join(self.results_path,self.exp_name)

    def load_pickle_file(self,name):
        # Load the loss values from .pkl files
        with open(os.path.join(self.files_path, name), 'rb') as f:
            list_value = pickle.load(f)
        
        return list_value
    
    def plot_accuracy(self):
        val_acc = self.load_pickle_file("val_accuracy.pkl")
        #Print the last accuracy
        print(f"Validation accuracy: {val_acc[-1]} ")
        # Plot the losses directly
        sns.set(style="whitegrid")
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(val_acc) + 1)
        plt.plot(epochs, val_acc, color="orange" ,label="Validation accuracy", marker='o',markersize=5)
        plt.title("Validation accuracy Over Epochs", fontsize=14)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.show()


    

    def plot_loss(self):
        """
        Plots training and validation loss over epochs.

        Args:
        results_path (str): Path to the folder containing experiment results.
        exp_name (str): Experiment name (subfolder in results_path).
        """
        train_loss = self.load_pickle_file("train_loss.pkl")
        val_loss = self.load_pickle_file("val_loss.pkl")
        # Ensure the lengths match
        assert len(train_loss) == len(val_loss), "Mismatch in number of epochs for train and validation loss."
        # Use Seaborn colors for nicer tones
        palette = sns.color_palette("coolwarm", n_colors=2)
        train_color = palette[0]  # A cool blue
        val_color = palette[1]    # A warm red
        # Plot the losses directly
        sns.set(style="whitegrid")
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(train_loss) + 1)
        plt.plot(epochs, train_loss, color="blue" ,label="Train Loss", marker='o', markersize = 5)
        plt.plot(epochs, val_loss, color="orange" ,label="Validation Loss", marker='o', markersize=5)
        plt.title("Training and Validation Loss Over Epochs", fontsize=14)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.show()



if __name__ == "__main__":
    #insert here the results path folder containing all the experiments resutls
    results_path = "/Users/irizabaranyanka/Desktop/Period_2/Deep_Learning/dl_assignment_3/experiments/results"
    exp_name = "exp_baseline_part3"
    data = Plot(results_path,exp_name)
    accuracy = data.load_pickle_file("val_accuracy.pkl")
    print(accuracy)
    print(f"Validation accuracy: {accuracy[-1]}")
    data.plot_loss()
    data.plot_accuracy()
