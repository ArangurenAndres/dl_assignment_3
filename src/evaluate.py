import torch
import os
from data.load_data import Data
from models.model import Net
import warnings
import numpy as np


class Evaluate:
    def __init__(self, results_path, exp_name):
        """
        Initialize evaluation with paths to results and experiment name.
        Args:
            results_path (str): Path to the directory containing the model weights.
            exp_name (str): Name of the experiment (model weight file name).
        """
        self.results_path = results_path
        self.exp_name = exp_name
        self.weights_path = os.path.join(self.results_path, self.exp_name,"model_weights.pth")
        print(f"Model weights path: {self.weights_path}")

    def load_model(self):
        """
        Load the model and its weights from the specified path.

        Returns:
            model (Net): PyTorch model loaded with weights.
        """
        # Create model instance
        model = Net()
        # Check if the weights file exists
        if os.path.exists(self.weights_path):
            model.load_state_dict(torch.load(self.weights_path,weights_only=True))
            model.eval()  # Set the model to evaluation mode
            print("Model successfully loaded.")
        else:
            warnings.warn(f"Could not find model in specified path: {self.weights_path}")
        return model

    def evaluate_model(self):
        """
        Evaluate the model on the test dataset.

        Returns:
            float: Test accuracy of the model.
        """
        # Load test dataset
        data = Data(batch_size=16)  # Ensure batch_size is properly set in Data class
        _, test_loader = data.load_data()

        # Initialize counters
        correct_preds = 0
        total_preds = 0

        # Load model
        model = self.load_model()

        # Iterate through test data
        for images, labels in test_loader:
            # Ensure the input tensor is on the same device as the model (e.g., CPU/GPU)
            images, labels = images.to(torch.device("cpu")), labels.to(torch.device("cpu"))

            # Forward pass
            with torch.no_grad():
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                correct_preds += (preds == labels).sum().item()
                total_preds += labels.size(0)

        # Calculate test accuracy
        test_accuracy = correct_preds / total_preds
        print(f"Test Accuracy: {test_accuracy:.4f}")
        return test_accuracy


if __name__ == "__main__":
    # Specify the experiment for which you want to test the model
    results_path = "experiments/models"
    exp_name = "exp_1_baseline"  # Ensure this matches your saved weight file name
    evaluator = Evaluate(results_path, exp_name)
    test_accuracy = evaluator.evaluate_model()
    print(f"Final Test Accuracy: {test_accuracy:.4f}")