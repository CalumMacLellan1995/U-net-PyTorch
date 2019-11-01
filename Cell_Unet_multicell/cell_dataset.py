#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 1/11/2019: 2pm

@author: calmac

This script is required for calling the images and targets for the multi-cell
segmentation problem.

Updated 01/11/19:
    - the overlap classes were doing sod all to get the cells segmented.
      decided to just load each ground truth of each cell for all images 
      and use them to train the network. 
    - dont need the to_one_hot() function anymore, so its removed.
    - recoded the targetfile_list in _init_() to load .mat file straight from MATLAB.
        --> repeated all of that for images as well.
    - added the create_tensor() function to tidy up the code. This function basically 
      stores all the classes (i.e. cell GTs) into one tensor, effectively creating a OHE tensor.
        --> don't need to send input images to create_tensor(), since we are only calling one
            file at a time. Only need to convert to tensor then we're done!
          
"""


from __future__ import print_function
import numpy as np
import random # for shuffling the dataset while creating training/validation data
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import transforms 
from scipy import io  # for loading matlab (.mat) files 
import matplotlib.pyplot as plt

class TrainingDataset(data.Dataset):
    
    def __init__(self, image_dir, targets_dir):
        
        # Inputs (i.e. images from .mat file)
        self.imagefile_list  = io.loadmat(image_dir)     # INPUTS: read MATLAB .mat file containing 100 examples
        self.inputs = self.imagefile_list['Train_DataSet'] # the name of the variable in .mat containing data is 'Train_DataSet'
        self.inputs = np.squeeze(self.inputs) # squeezes object into one column   
        
        # Targets
        self.targetfile_list = io.loadmat(target_dir)    # TARGETS: read MATLAB .mat file containing 100 examples 
        self.targets = self.targetfile_list['Train_Mask_DataSet'] # same as above: use the name as the key to get data from dict.
        self.targets = np.squeeze(self.targets) # squeezes object into one column   
        
    def __getitem__(self, index):
        
        """
        -------------------------
        Input images:    
        -------------------------
        """
        
        # Get image at index i (from DataLoader in main loop)
        inputfile_i = self.inputs[index]
        
        # Convert to torch tensor from numpy.
        train_img_tensor = torch.from_numpy(inputfile_i).float()
        
        """
        ------------------------
        Targets: first alteration on 01.11.19.
        ------------------------
        """

        # Get the ith target file from the list of 100. 
        targetfile_i = self.targets[index]
        
        # Need to squeeze again because the targets file has 100x1 cell (hence first squeeze in __init__())
        # which contains separate 5x1 cell arrays. 
        # But then need to be squeezed in order to access inner info (ie 512x512 array of 1/0s)
        targetfile_i = np.squeeze(targetfile_i) 
        
        # Send the file to create_tensor(). 
        train_target_tensor = self.create_tensor(targetfile_i)
        
        return [train_img_tensor, train_target_tensor]
    
   
    def create_tensor(self, file): 
        """
        create_tensor():
            
        Given a matlab file (.mat), spit out a torch tensor.
        This is for the targets only, which are already one-hot encoded.
        
        Since we need the targets to be in the form [n, 512, 512], where n is 
        the number of cells in the image, this function takes in a file, figures out 
        how many cells are in it (using C.shape[0]), and the image dimensions (512, 512)
        it then assigns each 512x512 array to each class using X[i] = C[i].

        """
        
        # Currently file is still an object, which we cant do anything with.
        # Create a numpy array by cycling through each image and pulling out the targets.
        C = file # assign to variable C for easier reading.
        X = np.empty((C.shape[0], C[0].shape[0], C[0].shape[1])) # preallocate numpy array of size equivalent to number of cells in that image.
        
        # Assign one-hot encoded (OHE) targets to numpy array, X.
        for i in range(X.shape[0]):  # for each image (ie cell 1 to 5)
            X[i] = C[i]              # assign each (512, 512) array of 1/0 information to its own OHE class.
        # which gives a one-hot encoded numpy array for each cell class (shape = (5, 512, 512) )

        # Need to have as tensor and in float form:
        tensor = torch.from_numpy(X).float()
        return tensor
        
    
def split_train_val(dataset, val_percent):
    dataset = list(dataset)
    length = len(dataset)
    n = int(length * val_percent)
    random.shuffle(dataset)  
    return {'train': dataset[:-n], 'val': dataset[-n:]} # train = all data minus number of validation examples
                                                        # val   = the remaining number of examples
    
if __name__ == '__main__':
            
    batch_size = 1
    image_dir = "E:\\USB_backup_EngDwork\\EngD_work\\Cell_Unet_multicell\\trainingData\\training_images.mat"
    target_dir = "E:\\USB_backup_EngDwork\\EngD_work\\Cell_Unet_multicell\\trainingData\\training_targets.mat"
    train_dataset = TrainingDataset(image_dir, target_dir)
    
    traindataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=0)
 
    #print(next(iter(traindataloader)))

    # Check the dataset has been loaded correctly by checking its size:
    print(len(train_dataset))

    # Cool it works: so now we need to just replicate what the DataLoader will do with enumerate(), which
    # is load the inputs/targets (ie image/target pair) one set at a time.
    # It gets the input/target pair from __getitem__ which returns the tensors separately. 

    # Try this for the first image in the dataset:
    image, targets = train_dataset[10]
    print(image.size())
    print(targets.size())

    # To visualise the image/target pair, need to squeeze the first dimension (Channel number) 
    # out and then convert to numpy.
    image = image.squeeze(0).numpy()
    target = targets[5].squeeze(0).numpy() # extract a particular cell's ground truth labels.
    plt.imshow(image)
    plt.imshow(target)

   
