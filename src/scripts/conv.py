import torch
from torch import nn

class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3),
                 stride=1, padding=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Initialize kernel weights (out_channels, in_channels, k_h, k_w)
        self.weights = nn.Parameter(
            torch.randn(out_channels, in_channels, *kernel_size)
        )

    def apply_padding(self, x):
        """
        Apply zero padding manually to the input tensor.

        Parameters:
            x (torch.Tensor): Input tensor of shape (b, c, h, w).

        Returns:
            torch.Tensor: Padded tensor of shape (b, c, h + 2 * padding, w + 2 * padding).
        """
        b, c, h, w = x.shape
        padded_h = h + 2 * self.padding
        padded_w = w + 2 * self.padding

        # Initialize a zero tensor for padding
        x_padded = torch.zeros((b, c, padded_h, padded_w), device=x.device, dtype=x.dtype)

        # Copy the original image into the padded tensor
        x_padded[:, :, self.padding:self.padding + h, self.padding:self.padding + w] = x

        return x_padded

    def forward(self, input_batch):
        b, c, h, w = input_batch.shape
        k_h, k_w = self.kernel_size

        # Calculate output dimensions
        h_output = (h - k_h + 2 * self.padding) // self.stride + 1
        w_output = (w - k_w + 2 * self.padding) // self.stride + 1

        # Apply padding to the input
        image_padded = self.apply_padding(input_batch)

        # Initialize output tensor
        output = torch.zeros((b, self.out_channels, h_output, w_output), device=input_batch.device)

        # Fold the input tensor into patches (b * p, k)
        patches = []
        for i in range(h_output):
            for j in range(w_output):
                start_i = i * self.stride
                start_j = j * self.stride

                # Extract patch of shape (b, c, k_h, k_w)
                patch = image_padded[
                    :, :, start_i:start_i + k_h, start_j:start_j + k_w
                ]  # Shape: (b, in_channels, k_h, k_w)

                # Flatten the patch to (b, k), where k = c * k_h * k_w
                patch_flat = patch.reshape(b, -1)   # Shape: (b, c * k_h * k_w)
                patches.append(patch_flat)

        # Stack all patches into a tensor of shape (b * p, k)
        patches = torch.stack(patches, dim=1).view(b * h_output * w_output, -1)

        # Perform batched matrix multiplication: (b * p, k) x (k, l) -> (b * p, l)
        kernel_flat = self.weights.reshape(self.out_channels, -1).t()  # Shape: (k, l)
        Y = torch.matmul(patches, kernel_flat)  # Shape: (b * p, out_channels)

        # Reshape Y back to (b, p, l) and permute to (b, out_channels, h_output, w_output)
        Y = Y.view(b, h_output, w_output, self.out_channels).permute(0, 3, 1, 2)

        # Optionally, add assertions to check the dimensions
        assert Y.shape == (b, self.out_channels, h_output, w_output), "Output shape mismatch"

        return Y


# Testing the module
if __name__ == "__main__":
    # Instantiate the convolution layer
    conv = Conv2D(in_channels=3, out_channels=32, kernel_size=(3, 3),
                  stride=1, padding=1)
    
    # Create an input batch
    input_batch = torch.randn(16, 3, 32, 32)
    
    # Apply the convolution
    output_batch = conv(input_batch)
    print(output_batch.shape)  # Expected output shape: (16, 32, 32, 32)
