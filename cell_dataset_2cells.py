#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/10/2019: 10:00

@author: calmac

PREAMBLE (23/10):
    
I was getting bogged down by the details of implementing the network on data
with varying numbers of classes; some images contained only 2 classes (ie no overlap), others
had 3 (only 2 cells overlapping), and then some had up to 6 classes (5 cells overlapping).

I couldn't get my head around how to structure Unet to deal with this (ie how many outputs should the last
layer have if the expected number (eg 6) is met with only 3 classes from a certain image) 
and got confuse.  

So in this dataset.py file I'm stripping it back to basics: run Unet on the same kind of data as the Masters project.
Focus on training it on lots of images (100s) with 2 cells overlapping with varying overlap degrees. 

Make changes (ie more complex data) on the basis of this working. This is also a good idea 
of proving that the future work insights I suggested in my thesis (ie to explore deep learning) were 
wise, and show the DL performs better than what I proposed previously. 


Updates: 29/10/19
    - 28/10/19:  
        Since we only have a batch_size = 1, and apparently use group normalisation (and not batch normalisation)
        there is no need to resize inputs or targets. 
        
        Thus, the __init__() arguments taking in resize dimensions have been removed. 
        
        Also, since I'm trying OHE vectors, which only work with the Dice loss function, this 
        version has uncommented the code to produce the OHE targets, which are in .float() form. 
     
    - 29/10/19: 
        
   
    
"""


from __future__ import print_function
import numpy as np
import random # for shuffling the dataset while creating training/validation data
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import transforms 
import cv2 # need to run: < pip install opencv-python > in cmd prompt before importing this package
import glob # module for finding all the pathnames matching a user defined pattern 
import matplotlib.pyplot as plt

class TrainingDataset(data.Dataset):
    
    def __init__(self, image_dir, targets_dir):
        
        self.trainfile_list = sorted(glob.glob(image_dir))     # assign the directory of the training images 
        self.targetfile_list = sorted(glob.glob(targets_dir))  # assign the directory of the target images
        
            
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
        Targets:
        ------------------------
        """
        # Get the ith target file from the directory. 
        targetfile_i = self.targetfile_list[index]
        
        # Load labels file as a numpy array as integer: data must be unsigned 8-bit integer otherwise cv2.resize doesnt work!
        labels = np.loadtxt(targetfile_i, dtype='uint8', delimiter=",") 
        labels = np.int8(labels)
        train_target_tensor = self.to_one_hot(labels) # send labels_resized to the to_one_hot() function to be one-hot encoded.  
        
        return [train_img_tensor, train_target_tensor]
    
    # Use the torch.eye identity matrix to compute the one hot encoded version 
    # of the labels (ohe_labels)
    def to_one_hot(self, labels):
        nclasses = int(np.amax(labels) + 1)   # get number of classes in that image 
        ohe_labels = torch.eye(nclasses)[labels].float() # create an identity matrix torch.eye of size (nclasses,nclasses). 
        
        return ohe_labels
    
    
    def __len__(self):
        assert len(self.trainfile_list) == len(self.targetfile_list)
        
        return len(self.trainfile_list)
    
   
def showtensor(inp, title=None):
    """Imshow for Tensor (ie input/target images converted to numpy then tensor)."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    
def split_train_val(dataset, val_percent):
    dataset = list(dataset)
    length = len(dataset)
    n = int(length * val_percent)
    random.shuffle(dataset)
    
    return {'train': dataset[:-n], 'val': dataset[-n:]} # train = all data minus number of validation examples
                                                        # val   = the remaining number of examples
    
#if __name__ == '__main__':
#    
#        
#    batch_size = 1
#    image_dir = "./Ready_twocells_data/subset_train/training_images/*"
#    target_dir = "./Ready_twocells_data/subset_train/training_targets/*"
#    train_dataset = TrainingDataset(image_dir, target_dir)
#    
#    traindataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=0)
# 
#    print(next(iter(traindataloader)))
#
#    # Check the dataset has been loaded correctly by checking its size:
#    print(len(train_dataset))
#
#    # Cool it works: so now we need to just replicate what the DataLoader will do with enumerate(), which
#    # is load the inputs/targets (ie image/target pair) one set at a time.
#    # It gets the input/target pair from __getitem__ which returns the tensors separately. 
#
#    # Try this for the first image in the dataset:
#    image1, targets1 = train_dataset[1]
#    print(image1.size())
#    print(targets1.size())
#
#    # To visualise the image/target pair, need to squeeze the first dimension (Channel number) 
#    # out and then convert to numpy.
#    img = image1.squeeze(0).numpy()
#    target = targets1.squeeze(0).numpy()
#    plt.imshow(img)
#    plt.imshow(target)
#
#   