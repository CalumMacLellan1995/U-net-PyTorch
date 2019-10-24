#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 10:50:00 2019

@author: hesun
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 14:03:29 2019

@author: hesun
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
from unet_update import UNet
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
save_dir = os.path.join('models', args.model+'_')
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

batch_size = args.batch_size 
epochs = args.epochs         
num_classes = 3              # Two cell problem has 3 classes: 0=background, 1=cytoplasm, 2=overlap.
input_resize = (572, 572)    # we need the input images to be 572x572 for max pooling to be valid.
output_resize = (388, 388)   # the output segmentation map will be upsampled to 388x388, so the target array must be matched in size.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

# Directories for paths to data 
image_dir = "./Ready_twocells_data/train_data/trainingImages/*"    # use ./ notation for glob() class
target_dir = "./Ready_twocells_data/train_data/trainingTargets/*"  # "" ""
    
if __name__ == '__main__':
    
    train_dataset = TrainingDataset(image_dir, target_dir, input_resize, output_resize)
    traindataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=0)
    print('===> Built dataset')

    classifier = UNet(n_channels = 1, n_classes = num_classes)
    classifier.to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=args.lr, weight_decay = 1e-4)
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [50,100], gamma=0.1)
    l_dice = loss.DiceLoss()
    print('===> Built model')

    for epoch in range(epochs):
        print('===> Start training')
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
        Dice_1 = AverageMeter()     # Average DC for class 1 (cytoplasm)
        Dice_2 = AverageMeter()     # Average DC for class 2 (overlap region)
        
        for i, (inputs, targets) in enumerate(traindataloader):
            
            # For each image-target pair, extract the input and target tensors from cell_dataset.py
            inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets = Variable(inputs), Variable(targets)
            
            optimizer.zero_grad()           # Zero the gradients to prevent future gradients being added to existing gradients
            classifier = classifier.train() # call the classifier (Unet) and set it to training mode 
            pred = classifier(inputs)       # send a batch of input images through network and compute segmentation map
            pred1 = pred
            target1 = targets.long()        # make targets into integers of unlimited length . this is actually redundant now!!
            
            # permute(0,2,3,1): this reorders the dimensions of the tensor using the indices specified by the user. 
            # (eg (0, 1, 2, 3) can be reshaped to (0, 2, 3, 1) meaning that in this case, He is moving the image dimensions 
            # in positions 2 and 3 over to positions 1 and 2, but moving the number of channels to position 3. Not sure what
            # the fourth dimension is for -> need to ask He when he gets back. 
            
            # .contiguous(): to do with memory, doesnt do anything to tensor. 
            
            pred = pred.permute(0,2,3,1).contiguous()
            
            # .view(-1, num_classes) -> Resize output map according number of classes there are in that image. (ie size is inferred based on pred and number of classes).
            # important for me since I'll be dealing with images of varying class sizes. 
            # This happens in He's paper, where one of the training images contains only background as a class, even though 
            # his network has been designed to output 4 classes. 
            pred = pred.view(-1, num_classes)   # reshape segmentation map. 
            targets = targets.view(-1).long()   # make same shape as pred; flattens target tensor to (N, 1) size. also more integer conversion: I actually think .long() is redundant now in Python 3 
            loss1 = l_dice(pred1,target1)       # compute dice loss between prediction and target
            loss2 = F.nll_loss(pred,targets)    # compute negative log likelihood loss 
            loss = loss1+loss2                  # sum the two losses to get custom loss result 
            loss.backward()                     # backpropagate errors through network
            optimizer.step()                    # update parameters 
            
            pred_choice = pred.data.max(1)[1]   # choose the label the network thinks is present (based on highest/max energy level)
            pred_seg = pred_choice.cpu().numpy()   # convert prediction torch tensor back into numpy array...
            label_seg = targets.data.cpu().numpy() # ...and again for targets
            
            # Compute Average DC for all classes, class 1 (LV), class 2 (RV), and class 3 (myocard.)
            dice_score, dice1, dice2, dice3 = compute_average_dice(pred_seg,label_seg)
            Dice.update(dice_score) # update Dice() with average DC for that image
            Dice_1.update(dice1)
            Dice_2.update(dice2)
            train_loss_epoch.append(loss.detach().cpu().numpy())    # loss for that epoch = total loss (DC + NLL)
            train_loss1_epoch.append(loss1.detach().cpu().numpy())  # loss1 = Dice: so this is the cumulative Dice score for this epoch
            train_loss2_epoch.append(loss2.detach().cpu().numpy())  # loss2 = NLL: so this is the cumulative NLL for this epoch
            train_dice_epoch.append(Dice)                           # 
        
        # Print results.        
        print(('epoch %d | train dice: %f') % (epoch+1, Dice.avg))
        print(('epoch %d | train dice 1: %f') % (epoch+1, Dice_1.avg))
        print(('epoch %d | train dice 2: %f') % (epoch+1, Dice_2.avg))
        print(('epoch %d | mean train loss: %f') % (epoch+1, np.mean(train_loss_epoch)))
        print(('epoch %d | mean train loss_dice: %f') % (epoch+1, np.mean(train_loss1_epoch)))
        print(('epoch %d | mean train loss_CE: %f') % (epoch+1, np.mean(train_loss2_epoch)))
        
        # Write the results 
        log_string(('epoch %d | train dice: %f') % (epoch+1, Dice.avg))
        log_string(('epoch %d | train dice 1: %f') % (epoch+1, Dice_1.avg))
        log_string(('epoch %d | train dice 2: %f') % (epoch+1, Dice_2.avg))
        log_string(('epoch %d | mean train loss: %f') % (epoch+1, np.mean(train_loss_epoch)))
        
        # Save the model at every epoch. 
        # This lets us decide which model performed the best so that we can keep it for 
        # future use. 
        torch.save(classifier.state_dict(), '%s/%s_model_%d.pth' % (args.out, 'acdc', epoch))
        

    
    
    
    
    
    
