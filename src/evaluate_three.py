import os
import torch
from torchvision.transforms import ToTensor
from PIL import Image
from models.model_part_three import Net_resolution

def load_test_data(test_dir):
    """Loads test data and labels without resizing."""
    test_data, test_labels = [], []

    for label in range(10):  # Assuming subfolders are named 0-9
        label_dir = os.path.join(test_dir, str(label))
        for image_path in os.listdir(label_dir):
            img = Image.open(os.path.join(label_dir, image_path))
            img_tensor = ToTensor()(img)  # Convert to tensor without resizing
            test_data.append(img_tensor)
            test_labels.append(label)

    return test_data, test_labels

def evaluate_model(weights_path, test_data, test_labels):
    """Evaluates the model on the test data."""
    # Load model
    model = Net_resolution(N=81)
    model.load_state_dict(torch.load(weights_path,weights_only=True))
    model.eval()  # Set model to evaluation mode

    # Move model to appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    correct_preds = 0
    total_preds = 0

    # Process each image individually
    for img_tensor, label in zip(test_data, test_labels):
        # Move image and label to the device
        img_tensor = img_tensor.unsqueeze(0).to(device)  # Add batch dimension
        label_tensor = torch.tensor([label]).to(device)

        # Make prediction
        with torch.no_grad():
            output = model(img_tensor)
            _, prediction = torch.max(output, 1)

        # Update accuracy
        correct_preds += (prediction == label_tensor).sum().item()
        total_preds += 1

    accuracy = correct_preds / total_preds
    return accuracy

if __name__ == "__main__":
    # Define paths
    test_dir = "mnist_var/test"  # Path to test data
    weights_path = "experiments/models_var_res/exp_dropout/model_weights.pth"  # Path to saved model weights

    # Load test data
    test_data, test_labels = load_test_data(test_dir)

    # Evaluate model
    test_accuracy = evaluate_model(weights_path, test_data, test_labels)
    print(f"Test Accuracy: {test_accuracy:.2%}")
