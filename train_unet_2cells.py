#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The 'main' file for calling all the classes/functions to train the network on 
the cervical cell data.

Latest update: 29/10/19.
Description: updated the Dice loss function to incorporate one-hot encoded prediction and target maps.

@author: Calum
"""
from datetime import datetime
import argparse
import os
import os.path
import torch
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from cell_dataset_2cells import TrainingDataset
from unet import UNet
#from unet_update import UNet
from utils import compute_average_dice, AverageMeter
import loss  
from torch.autograd import Variable
import multiprocessing


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
parser.add_argument('--momentum', type=float, default=0.99, help='momentum in optimizer')
parser.add_argument('--batch_size', type=int, default=1, help='batch size') # choose batch_size of 1 initially
parser.add_argument('--epochs', type=int, default=5, help='epochs to train')
parser.add_argument('--out', type=str, default='./model_checkpoint', help='path to save model checkpoints')
parser.add_argument('--lr-mode', type=str, default='step')
parser.add_argument('--step', type=int, default=20)


# Hyperparameters
args = parser.parse_args()
#save_dir = os.path.join('models', args.model+'_')
#if not os.path.exists(save_dir):
#    os.mkdir(save_dir)

batch_size = args.batch_size 
epochs = args.epochs   
n_channels = 1              # grayscale images thus only 1 channel      
n_classes = 3              # Two cell problem has 3 classes: 0=background, 1=cytoplasm, 2=overlap.
input_resize = (572, 572)    # we need the input images to be 572x572 for max pooling to be valid.
output_resize = (388, 388)   # the output segmentation map will be upsampled to 388x388, so the target array must be matched in size.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

# Directories for paths to data.
"""
This version has used a larger dataset from the subset_train folder.
Just to see if the performance improves at all...
"""
image_dir = "./Ready_twocells_data/subset_train/training_images/*"    # use ./ notation for glob() class
target_dir = "./Ready_twocells_data/subset_train/training_targets/*"  # "" ""
    
if __name__ == '__main__':
    
    train_dataset = TrainingDataset(image_dir, target_dir)
    traindataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=0)
    print('===> Built dataset')

    classifier = UNet(n_channels, n_classes)
    classifier.to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=args.lr, weight_decay = 1e-4)
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [50,100], gamma=0.1)
    l_dice = loss.DiceLoss()
    print('===> Built model')
    print('===> Start training')

    for epoch in range(epochs):
        print('===> EPOCH %03d ' % (epoch+1))
        log_string('**** EPOCH %03d ****' % (epoch+1))
        log_string(str(datetime.now()))
        print(optimizer.param_groups[0]['lr'])
        
        # Initialise storage for loss results.
        # I think the different Dice scores are for distinct classes (eg Dice_1 = Dice score for 
        # segmenting left ventricle (class=1), Dice_2 = DC for right ventricle (class=2), and so on. But for us, 
        # this is for the different overlap regions).
        train_loss_epoch, train_dice_epoch = [], []
        train_loss1_epoch, train_loss2_epoch = [], []   # loss 1 = average dice for that epoch
        Dice = AverageMeter()       # Average of the averages (ie overall average DC score for all classes)
        Dice_bgd = AverageMeter()   # Store average DC for class 1 (background pixels)
        Dice_cyto = AverageMeter()     # Store average DC for class 2 (cytoplasm)
        Dice_ovlp = AverageMeter()     # Store average DC for class 3 (overlap region) 
        
        for i, (inputs, targets) in enumerate(traindataloader):
            
            # For each image-target pair, extract the input and target tensors from cell_dataset.py.
            # We want the One-hot encoded tensor targets for the Dice loss function, but since NLL doesnt
            # like them in that form
            inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets = Variable(inputs), Variable(targets)
            
            optimizer.zero_grad()           # Zero the gradients to prevent future gradients being added to existing gradients
            classifier = classifier.train() # call the classifier (Unet) and set it to training mode 
            pred = classifier(inputs)       # send a batch of input images through network and compute segmentation map
#            print('Prediction map original/Dice size:')
#            print(pred.size())
#            
            pred_dice = pred                # use this original prediction map for the dice loss.
            
            # Need to put targets used for dice loss in same shape as pred_dice.
            # Use .permute() for this. 
            # Since pred_dice shape is [1, 3, 512, 512], and targets is [1, 512, 512, 3]
            # we switch the 3 with the 512, 512 so it is target_dice = [1, 3, 512, 512] like pred_dice.
            target_dice = targets.permute(0, 3, 1, 2).float()   # use for dice loss with pred_dice; make targets into integers of unlimited length . this is actually redundant now!!
#            print('Target map original size:')
#            print(targets.size())
#            print('Target map Dice size:')
#            print(target_dice.size())
            # permute(0,2,3,1): this reorders the dimensions of the tensor using the indices specified by the user. 
            # (eg (0, 1, 2, 3) can be reshaped to (0, 2, 3, 1) meaning that in this case, He is moving the image dimensions 
            # in positions 2 and 3 over to positions 1 and 2, but moving the number of channels to position 3. Not sure what
            # the fourth dimension is for -> Batch size (which is == 1 in our case) 
            
            # .contiguous(): to do with memory, doesnt do anything to tensor. 
            
            pred = pred.permute(0,2,3,1).contiguous()
#            print('Prediction map permuted size:')
#            print(pred.size())
            
            # .view(-1, num_classes) -> Resize output map according number of classes there are in that image. (ie size is inferred based on pred and number of classes).
            # important for me since I'll be dealing with images of varying class sizes. 
            # This happens in He's paper, where one of the training images contains only background as a class, even though 
            # his network has been designed to output 4 classes. 
            pred_nll = pred.view(-1, n_classes)     # reshape segmentation map.
#            print('Prediction map .view size:')
#            print(pred_nll.size())
            
            # Targets need to be back in their class indices form, as NLL or Cross entropy loss functions
            # dont like them as one-hot encoded vectors.
            # Need to convert back to class indices form:
            target_nll = torch.max(target_dice, 1)[1]
            target_nll = target_nll.view(-1).long()   # make same shape as pred; flattens target tensor to (N, 1) size -> need .long() 
#            print('Target map .view size:')
#            print(target_nll.size())

            loss1 = l_dice(pred_dice, target_dice)       # compute dice loss between prediction and target
            loss2 = F.nll_loss(pred_nll, target_nll)    # compute negative log likelihood loss 
            loss = loss1+loss2                  # sum the two losses to get custom loss result 
            loss.backward()                     # backpropagate errors through network
            optimizer.step()                    # update parameters 
            
            pred_choice = pred_nll.data.max(1)[1]   # get the index of the max log-probability
            pred_seg = pred_choice.cpu().numpy()   # convert prediction torch tensor back into numpy array...
            label_seg = target_nll.data.cpu().numpy() # ...and again for targets
            
            # Compute Average DC for all classes, class 1 (LV), class 2 (RV), and class 3 (myocard.)
            dice_score, dice1, dice2, dice3 = compute_average_dice(pred_seg, label_seg, n_classes)
            Dice.update(dice_score) # update Dice() with average DC for that image
            Dice_bgd.update(dice1)
            Dice_cyto.update(dice2)
            Dice_ovlp.update(dice3)
            train_loss_epoch.append(loss.detach().cpu().numpy())    # loss for that epoch = total loss (DC + NLL)
            train_loss1_epoch.append(loss1.detach().cpu().numpy())  # loss1 = Dice: so this is the cumulative Dice score for this epoch
            train_loss2_epoch.append(loss2.detach().cpu().numpy())  # loss2 = NLL: so this is the cumulative NLL for this epoch
            train_dice_epoch.append(Dice)                           # 
        
        # Print results for average DC of each class.
        # Averaged across the N images in the training dataset.        
        print(('epoch %d | Avg. overall train dice: %f')      % (epoch+1, Dice.avg))
        print(('epoch %d | Avg. train dice (background): %f') % (epoch+1, Dice_bgd.avg))
        print(('epoch %d | Avg. train dice (cytoplasm): %f')  % (epoch+1, Dice_cyto.avg))
        print(('epoch %d | Avg. train dice (overlap): %f')    % (epoch+1, Dice_ovlp.avg))
        print(('epoch %d | Mean train loss: %f')              % (epoch+1, np.mean(train_loss_epoch)))
        print(('epoch %d | Mean train loss_dice: %f')         % (epoch+1, np.mean(train_loss1_epoch)))
        print(('epoch %d | Mean train loss_CE: %f')           % (epoch+1, np.mean(train_loss2_epoch)))
        
        # Write the results 
        log_string(('epoch %d | Avg. overall train dice: %f')        % (epoch+1, Dice.avg))
        log_string(('epoch %d | Avg. train dice (background): %f')   % (epoch+1, Dice_bgd.avg))
        log_string(('epoch %d | Avg. train dice (cytoplasm): %f')    % (epoch+1, Dice_cyto.avg))
        log_string(('epoch %d | Avg. train dice (overlap): %f')      % (epoch+1, Dice_ovlp.avg))
        log_string(('epoch %d | Mean train loss: %f')                % (epoch+1, np.mean(train_loss_epoch)))
        
        # Save the model at every epoch. 
        # This lets us decide which model performed the best so that we can keep it for 
        # future use. 
        torch.save(classifier.state_dict(), '%s/%s_model_%d.pth' % (args.out, 'cellnet', epoch))
        

    
    
    
    
    
    