import numpy as np
import matplotlib.pyplot as plt
import torch

def apply_padding(x=None,p=1):
    b,c,h,w = x.shape
    x_padded = np.zeros((b,c,h+2*p,w+2*p))
    x_padded[:,:,p:p+h,p:p+w] = x
    return x_padded

def unfold_operation(x = None,k = None, s=1,p=1):
    #apply padding
    x_padded = apply_padding(x)

    b,c,h,w = x.shape
    k_h,k_w = k.shape
    h_output = ((h-k_h+2*p)//s)+1
    w_output = ((w-k_w+2*p)//s)+1

    # Get the number of patches
    num_p = h_output*w_output
    # Values per patch
    k_values = k_h*k_w
    #initialize the array that will hold the patches
    patches = np.zeros((b,k_values,num_p))
    patch_idx = 0
    for i in range(h_output):
        start_i = i*s
        for j in range(w_output):
            start_j = j*s
            #Extract the patch from the input image
            patch = x_padded[:,:,start_i:start_i+k_h,start_j:start_j+k_w]
            patch_flat = patch.reshape(b,-1)
            patches[:,:,patch_idx]=patch_flat

            patch_idx+=1
    return patches




if __name__ == "__main__":
    sample = torch.randn(1, 1, 9, 9)
    kernel = np.ones((3,3))
    patches = unfold_operation(x = sample,k = kernel, s=1,p=1)


    



