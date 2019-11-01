#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 17/10/2019: 11:53

@author: calmac

THIS IS NOW THE UPDATED VERSION OF THE CELLDATASET FILE (as of 17/10).

WHATS CHANGED:
    - incorporated the TargetDataset() class from celldataset_v1.2.py into the TrainingDataset() class
        -> this includes all the functions, such as to_one_hot() and the __getitem__ lines.
    
"""


from __future__ import print_function
import numpy as np
import torch
import torch.utils.data as data
import cv2 # need to run: < pip install opencv-python > in cmd prompt before importing this package
import glob # module for finding all the pathnames matching a user defined pattern 
import matplotlib.pyplot as plt

class TrainingDataset(data.Dataset):
    
    def __init__(self, image_dir, targets_dir, input_resize, target_resize):
        
        self.trainfile_list = glob.glob(image_dir)     # assign the directory of the training images 
        self.targetfile_list = glob.glob(targets_dir)  # assign the directory of the target images
        self.input_resize = input_resize            # make the input images the size required for the Unet architecture (572, 572)
        self.target_resize = target_resize          # same for output images (388, 388)
    
    def __getitem__(self, index):
        
        """
        Input images:
            
        """
        # Get the file path from the current directory.
        # Get the image associated with index in grayscale form (0): save as numpy array with cv2 module.
        # Then resize according to required dimens.
        train_img = cv2.imread(self.trainfile_list[index], 0) # get the ith training image from the list of training images using the index
        train_img = cv2.resize(train_img, self.input_resize, interpolation = cv2.INTER_AREA)
        
        # Now convert the numpy array to a tensor for Unet
        train_img_tensor = torch.from_numpy(train_img).float()
        train_img_tensor = train_img_tensor.unsqueeze(0) # ANOTHER WAY to add the 1' channels dimension to beginning of tensor
        #train_img_tensor = train_img_tensor.view(1, *self.input_resize)  # reshape to CHW format (channels, height, width)
        
        """
        Targets:
            
        """
        
        targetfile_i = self.targetfile_list[index]
        
        # Load labels file as a numpy array as integer: data must be unsigned 8-bit integer otherwise cv2.resize doesnt work!
        labels = np.loadtxt(targetfile_i, dtype='uint8', delimiter=",") 
        labels_resized = cv2.resize(labels, self.target_resize, interpolation=cv2.INTER_AREA)
        labels_resized = np.int8(labels_resized) # now convert back to int8 otherwise we get a shape error when trying to do torch.eye()[] in to_one_hot
        # ^new lines for resizing target array to (388,388)
        #labels_resized = np.transpose(labels_resized, axes=[2, 0, 1])
        ohe_targets = self.to_one_hot(labels_resized) # send labels_resized to the to_one_hot() function to be one-hot encoded.  
        return [train_img_tensor, ohe_targets]
    
    # Use the torch.eye identity matrix to compute the one hot encoded version 
    # of the labels (ohe_labels)
    def to_one_hot(self, labels_resized):
        nclasses = int(np.amax(labels_resized) + 1)   # get number of classes in that image 
        ohe_labels = torch.eye(nclasses)[labels_resized] # create an identity matrix torch.eye of size (nclasses,nclasses). 
        return ohe_labels#.view(nclasses, *self.target_resize) # need to have tensors in CHW (channels, height, width) format
    
    
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
    
if __name__ == '__main__':
    # (18/10) Tryout the DataLoader to see if it works 
    from torch.utils.data import DataLoader
    
    batch_size = 1
    input_resize = (572, 572)
    target_resize = (388, 388)
    image_dir = './Calum_unet_celldata_20_09/trainingData/trainingImages/*'
    target_dir = './Calum_unet_celldata_20_09/trainingData/trainingTargets/*'
    train_dataset = TrainingDataset(image_dir, target_dir, input_resize, target_resize)
    
    traindataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=0)
    
#    for i, batch in enumerate(traindataloader):
#        print(i, batch)
      
    print(next(iter(traindataloader)))
    
    # Check the dataset has been loaded correctly by checking its size:
    print(len(train_dataset))

    # Cool it works: so now we need to just replicate what the DataLoader will do with enumerate(), which
    # is load the inputs/targets (ie image/target pair) one set at a time.
    # It gets the input/target pair from __getitem__ which returns the tensors separately. 
    
    # Try this for the first image in the dataset:
    image1, targets1 = train_dataset[0]
    
    # Check the sizes to make sure it was loaded properly.
    image1.size()
    # -> Out[10]: torch.Size([1, 572, 572])
    
    targets1.size()
    # -> Out[11]: torch.Size([388, 388, 5])

    # To visualise the result, we need to squeeze the tensor's channel dimension to 
    # make the numpy array into a 2D size. 
    img = image1.squeeze(0).numpy()
    target = targets1.squeeze(2).numpy()
    plt.imshow(img)

    