#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/10/2019: 10:00

@author: calmac

This script is required for calling the images and targets for the multi-cell
segmentation problem.

Updated 01/11/19:
    - the overlap classes were doing sod all to get the cells segmented.
      decided to just load each ground truth of each cell for all images 
      and use them to train the network. 
    - recoded the targetfile_list in _init_() to load .mat file straight from MATLAB.
    

"""


from __future__ import print_function
import numpy as np
import random # for shuffling the dataset while creating training/validation data
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import transforms 
from scipy import io  # for loading matlab (.mat) files 
import cv2 # need to run: < pip install opencv-python > in cmd prompt before importing this package
import glob # module for finding all the pathnames matching a user defined pattern 
import matplotlib.pyplot as plt

class TrainingDataset(data.Dataset):
    
    def __init__(self, image_dir, targets_dir):
        
        self.trainfile_list = sorted(glob.glob(image_dir))     # assign the directory of the training images 
        self.targetfile_list = io.loadmat(target_dir)          # read MATLAB .mat file containing 100 examples 
        self.targets = self.targetfile_list['Train_Mask_DataSet']
        self.targets = np.squeeze(self.targets) # does something   
        
    def __getitem__(self, index):
        
        """
        -------------------------
        Input images:    
        -------------------------
        """
        # Get the file path from the current directory.
        # Get the image associated with index in grayscale form (0): save as numpy array with cv2 module.
        # Then resize according to required dimens.
        train_img = cv2.imread(self.trainfile_list[index], 0) # get the ith training image from the list of training images using the index
        
        # Now convert the numpy array to a tensor for Unet
        train_img_tensor = torch.from_numpy(train_img).float()
        train_img_tensor = train_img_tensor.unsqueeze(0) # ANOTHER WAY to add the 1' channels dimension to beginning of tensor
        
        """
        ------------------------
        Targets: first alteration on 01.11.19.
        ------------------------
        """

        # Get the ith target file from the list of 100. 
        targetfile_i = self.targets[index]
        targetfile_i = np.squeeze(targetfile_i)
        
        # Currently still an object, which we cant do anything with.
        # Create a numpy array by cycling through each image and pulling out the targets.
        C = targetfile_i # assign as C: makes for easier reading.
        X = np.empty((C.shape[0], C[0].shape[0], C[0].shape[1]))
    
        # Assign one-hot encoded (OHE) targets to numpy array.
        for i in range(X.shape[0]):  # for each image (ie cell 1 to 5)
            X[i] = C[i]              # assign each (512, 512) array of 1/0 information to its own OHE class.
        
        # which gives a one-hot encoded numpy array for each cell class (shape = (5, 512, 512) )
        
        # Need to have as tensor:
        train_target_tensor = torch.from_numpy(X).float()
        
        return [train_img_tensor, train_target_tensor]
    
    # Given a matlab file (.mat), this function spits out a torch tensor.
    def create_tensor(self, file): 
        C = file # assign to variable C for easier reading.
        X = np.empty((C.shape[0], C[0].shape[0], C[0].shape[1])) # preallocate.
        
        # Assign one-hot encoded (OHE) targets to numpy array, X.
        for i in range(X.shape[0]):  # for each image (ie cell 1 to 5)
            X[i] = C[i]              # assign each (512, 512) array of 1/0 information to its own OHE class.
        # which gives a one-hot encoded numpy array for each cell class (shape = (5, 512, 512) )

        # Need to have as tensor and in float form:
        tensor = torch.from_numpy(X).float()
        return tensor
        
    # Use the torch.eye identity matrix to compute the one hot encoded version 
    # of the labels (ohe_labels)
    def to_one_hot(self, labels):
        nclasses = int(np.amax(labels) + 1)   # get number of classes in that image 
        ohe_labels = torch.eye(nclasses)[labels].float() # create an identity matrix torch.eye of size (nclasses,nclasses).   
        return ohe_labels
    
    
    def __len__(self):
        #assert len(self.trainfile_list) == len(self.targetfile_list)
        return len(self.trainfile_list)
    
def split_train_val(dataset, val_percent):
    dataset = list(dataset)
    length = len(dataset)
    n = int(length * val_percent)
    random.shuffle(dataset)  
    return {'train': dataset[:-n], 'val': dataset[-n:]} # train = all data minus number of validation examples
                                                        # val   = the remaining number of examples
    
if __name__ == '__main__':
            
    batch_size = 1
    image_dir = "./trainingData/subset_train/train_images/*"
    target_dir = "E:\\USB_backup_EngDwork\\EngD_work\\Cell_Unet_multicell\\trainingData\\training_targets.mat"
    train_dataset = TrainingDataset(image_dir, target_dir)
    
    traindataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=0)
 
    print(next(iter(traindataloader)))

    # Check the dataset has been loaded correctly by checking its size:
    print(len(train_dataset))

    # Cool it works: so now we need to just replicate what the DataLoader will do with enumerate(), which
    # is load the inputs/targets (ie image/target pair) one set at a time.
    # It gets the input/target pair from __getitem__ which returns the tensors separately. 

    # Try this for the first image in the dataset:
    image1, targets1 = train_dataset[1]
    print(image1.size())
    print(targets1.size())

    # To visualise the image/target pair, need to squeeze the first dimension (Channel number) 
    # out and then convert to numpy.
    img = image1.squeeze(0).numpy()
    target = targets1[1].squeeze(0).numpy()
    plt.imshow(img)
    plt.imshow(target)

   