# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 11:37:05 2019

@author: szb18149

Practicing reading in the .mat files for training and testing datasets.
"""

import numpy as np
from scipy import io
import matplotlib.pyplot as plt
import torch

    

if __name__ == '__main__':
    
    # Tell python where the file lives, and what its called
    path_targets = "E:\\USB_backup_EngDwork\\EngD_work\\Cell_Unet_multicell\\trainingData\\training_targets.mat"
    
    # Now use .loadmat to get the data and store in a dict.
    targets = io.loadmat(path_targets)
    
    # Check how the array looks
    print(type(targets)) # which tells us its a dict
    print(targets.keys()) # which stores the name of the cell array as a key.
    
    # Get the mask data and ignore the other keys
    targets = targets['Train_Mask_DataSet'] 
    print(type(targets)) # which is a numpy.ndarray
    print(targets.shape) # which is (100, 1)
    
    # Extract the first image ground truths
    G = np.squeeze(targets)
    
    C = G[0] # store 5x1 cell array for image 1
    C = np.squeeze(C) # treat this now as the C = np.squeeze(C) from that python tutorial
    
    # Create empty array to store 3D array 
    X = np.empty((C.shape[0], C[0].shape[0], C[0].shape[1]))
    
    for i in range(X.shape[0]):  # for each image
        X[i] = C[i]
    print(X.shape)  
    
    # Convert to Torch tensor 
    train_target_tensor = torch.from_numpy(X).float()
    
    # check size should be [5, 512, 512]: 
    print(train_target_tensor.size())
    
    # NOW X CONTAINS THE 512X512 IMAGE OF EACH CELL, SO HAS THE SHAPE:
    #   -> (5, 512, 512)
    # where each cell can be accessed separately by doing:
    # e.g.  cell_1 = X[0]
    # and plotted by: plt.imshow(cell_1)
    
#    img1_cell1 = img1_gts[0] # get the first cell 
#    length = img1_cell1[0].shape[0] 
#    height = img1_cell1[0].shape[1]
#    
#    