

# This function assumes squared images
def get_output_dimension(input_dim=None, input_channels=1,output_channels=3,kernel=3, stride=1,padding=1):
    b=1
    h,w = input_dim[0],input_dim[1]
    if h==w:
        out = ((h-kernel+2*padding)//stride)+1
        return (b,output_channels,out,out)
    else:
        h_out = ((h-kernel+2*padding)//stride)+1
        w_out = ((w-kernel+2*padding)//stride)+1
    return (b,output_channels,h_out,w_out)
if __name__ == "__main__":
    input_dim = (1024,768)
    input_channels = 3
    output_channels = 16
    kernel = 5
    stride = 2
    padding = 1
    out = get_output_dimension(input_dim,input_channels,output_channels, kernel, stride,padding)
    print(f"Expected output dimension:{out}")
