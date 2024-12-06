

# This function assumes squared images
def get_output_dimension(input_dim, kernel, stride,padding):
    out = ((input_dim-kernel+2*padding)//stride)+1
    return out
if __name__ == "__main__":
    input_dim = 32
    kernel = 3
    stride = 1
    padding = 1
    out = get_output_dimension(input_dim, kernel, stride,padding)
    print(f"Expected output dimension:{out,out}")
