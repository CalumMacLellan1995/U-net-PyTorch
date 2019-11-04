#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable

class DiceLoss(nn.Module):
    def __init__(self, class_num, smooth=0.5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.class_num = class_num

    def forward(self, output, target, class_num):
        output = torch.exp(output)   # returns the exponential of the log probabilties generated from Unet by the softmax function
#        self.smooth = 0.5
        device = torch.device("cuda")
        Dice = Variable(torch.Tensor([0]).float())
        Dice = Dice.to(device) # need this line otherwise the loss wont be loaded to the GPU; spits out an error
        for i in range(0,self.class_num):
            output_i = output[:, i, :, :]     # extract each prediction map and compare with target
            target_i = target[:, i, :, :]   # extract the OHE vector for each class 
            intersect = (output_i*target_i).sum()
            union = torch.sum(output_i) + torch.sum(target_i)
            if target_i.sum() == 0:
                dice = Variable(torch.Tensor([1]).float())
            else:
                dice = (2 * intersect + self.smooth) / (union + self.smooth)
            Dice += dice
        dice_loss = 1 - Dice/(self.class_num - 1)
        return dice_loss