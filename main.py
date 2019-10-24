# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 11:33:55 2019

@author: szb18149
"""

from datetime import datetime
import argparse
import os
import os.path
import torch
import torch.optim as optim
import torch.utils.data
from celldataset import TrainingDataset
import unet_update as unet
import multiprocessing
import classifier

def main():
    
    log_root = "./log"
    
    if not os.path.exists(log_root): os.mkdir(log_root)
    LOG_FOUT = open(os.path.join(log_root, 'train.log'), 'w')
    
    def log_string(out_str):
        LOG_FOUT.write(out_str+'\n')
        LOG_FOUT.flush()
        
    os.system('mkdir {0}'.format('model_checkpoint'))
    
    parser = argparse.ArgumentParser(description = '2D u-net')
    parser.add_argument('--model', default='UNet', type=str, help='choose a type of model')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.99, help='momentum in optimizer') # same as Unet paper 
    parser.add_argument('--batch_size', type=int, default=1, help='batch size') # same as unet paper
    parser.add_argument('--epochs', type=int, default=5, help='epochs to train')
    parser.add_argument('--out', type=str, default='./model_checkpoint', help='path to save model checkpoints')
    parser.add_argument('--lr-mode', type=str, default='step')
    parser.add_argument('--step', type=int, default=20)
    
    args = parser.parse_args()
    #save_dir = os.path.join('models', args.model+'_')
    #if not os.path.exists(save_dir): os.mkdir(save_dir)
    
    
    """
    Hyperparameters
    
    """
    batch_size = args.batch_size
    epochs = args.epochs
    input_img_resize = (572, 572)
    output_img_resize = (388, 388)
    n_channels = 1  # set to 1 for grayscale images
    n_classes  = 5 # the number of classes to segment
    
    # Other parameters/pathways
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainImage_dir  = './Calum_unet_celldata_20_09/trainingData/trainingImages/*'
    targetLabels_dir = './Calum_unet_celldata_20_09/trainingData/targetLabels/*'
    
    # Get the dataset using TrainingDataset() class from celldataset.py, 
    print('===> Building dataset')
    train_dataset = TrainingDataset(trainImage_dir, targetLabels_dir, train=True)
    
    # and load with the .Dataloader() class  from PyTorch
    traindataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=0)
    
    # Build the model from UNet, and assign number of input channels (1 here since
    # our images are grayscale), and number of classes
    print('===> Building model')
    net = unet.UNet(n_channels, n_classes, *input_img_resize, *output_img_resize)
    
    # Assign the classifier from classifier.py and the optimizer for updating the weights
    cellclassifier = classifier.CellClassifier(net, epochs)
    cellclassifier.to(device)
    optimizer = optim.Adam(cellclassifier.parameters(), lr=args.lr, weight_decay = 1e-4)
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [50,100], gamma=0.1)

    # Call the training module (.train) within the CellClassifier() class from classifier.py
    cellclassifier.train(traindataloader, optimizer, epochs)
        
    
    
if __name__ == '__main__':
    multiprocessing.set_method('spawn', force=True)
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    