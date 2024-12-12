import torch
import torch.nn as nn
import torch.nn.functional as F

class Net_resolution(nn.Module):
    # We define the model with the number of output classes.
    # parameter N:
    # defines:
    #  - number of output channels in second conv layer
    #  - number of units in linear layer -> which then outputs (units= num_classes)

    def __init__(self, num_classes=10, N=64, dropout=0.5):
        super(Net_resolution, self).__init__()

        # First Conv layer
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(dropout)

        # Second Conv layer
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout2d(dropout)

        # Third Conv layer
        self.conv3 = nn.Conv2d(32, N, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout2d(dropout)

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Using AdaptiveAvgPool2d

        # Fully connected layer
        self.fc = nn.Linear(N, num_classes)

    def forward(self, x):
        # First block
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        # Second block
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        # Third block
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        # Global pooling
        x = self.global_pool(x)  # Output size: (batch, N, 1, 1)
        x = torch.flatten(x, 1)  # Flatten to (batch, N)

        # Fully connected layer
        x = self.fc(x)

        return x

# Function to count total trainable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Example usage
if __name__ == "__main__":

    model = Net_resolution(num_classes=10, N=64, dropout=0.5)
    var_res_params = count_parameters(model)
    target_params =  29066
    print(f"The model contains initially: {var_res_params} parameters")
    print(f"The fixed resolution model has: {target_params} parameters")
    e = 40
    lower_bound = target_params - e
    upper_bound = target_params + e
    print(lower_bound, upper_bound)
    for n_test in range(10, 100):
        t_model = Net_resolution(num_classes=10, N=n_test, dropout=0.5)
        t_params = count_parameters(t_model)
        if lower_bound <= t_params <= upper_bound:
            print(f"Required n: {n_test}")
            print(f"Using N equals {n_test} the variable resolution model has {t_params} parameters, which is the closest approximation to the fixed resolution model")
