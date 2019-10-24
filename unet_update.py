# Create the Unet model and classes representing the operations and encoder/decoder blocks

import torch
import torch.nn.functional as F
import torch.nn as nn



"""
----------------------------------------
UNet class:
    where the architecture of the network is constructed layer by layer, using 
    the Encoder/Decoder classes and ConvBnRelu operation
----------------------------------------
"""

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        
        self.down1 = EncoderBlock(n_channels, 64)
        self.down2 = EncoderBlock(64, 128)
        self.down3 = EncoderBlock(128, 256)
        self.down4 = EncoderBlock(256, 512)        
        
        self.bottom = nn.Sequential(
                ConvBnRelu(512,  1024, kernel_size=(3,3), stride=1, padding=0),
                ConvBnRelu(1024, 1024, kernel_size=(3,3), stride=1, padding=0)
        )
        
        self.up1 = DecoderBlock(1024, 512,  upsample_size=(56,  56))          
        self.up2 = DecoderBlock(512,  256,  upsample_size=(104, 104))             
        self.up3 = DecoderBlock(256,  128,  upsample_size=(200, 200))           
        self.up4 = DecoderBlock(128,  64,   upsample_size=(392, 392))
        
        # 1x1 convolution at the output layer: gives the output segmentation map
        # for comparing with target segmentation map
        self.out_segmap = nn.Conv2d(64, n_classes, kernel_size=(1,1), stride=1, padding=0)

    def forward(self, x):
        
        # Contracting pathway: 
        #   output from convbnrelu + maxpool (x), as well as pre-maxpool output (x_copy) from block n
        x, x_copy1 = self.down1(x)
        x, x_copy2 = self.down2(x)
        x, x_copy3 = self.down3(x)
        x, x_copy4 = self.down4(x)
        
        # Valley floor path:
        x = self.bottom(x)
        
        # Expansive pathway:
        #   combine/concatenate upsampled tensor (x) with copy from corresponding n layer (x_copy)
        x = self.up1(x, x_copy4)
        x = self.up2(x, x_copy3)
        x = self.up3(x, x_copy2)
        x = self.up4(x, x_copy1)
        
        # Output segmentation map
        x = self.out_segmap(x)
        
        return F.log_softmax(x, dim=1)
    
    
 
"""
----------------------------------------
EncoderBlock:
    Describe
----------------------------------------
"""
    
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        
        self.convBnRe1 = ConvBnRelu(in_channels, out_channels, kernel_size=(3,3), stride=1, padding=0)
        self.convBnRe2 = ConvBnRelu(in_channels, out_channels, kernel_size=(3,3), stride=1, padding=0)
        self.maxpool   = nn.MaxPool2d(kernel_size=(2,2), stride=1)
        
    def forward(self, x):
        x = self.convBnRe1(x)   # do Convolution, Batch normalisation, and Relu on x
        x = self.convBnRe2(x)   # and a second time to reduce resolution 
        x_copy = x              # save a copy of the convolved feature map for the crop_concat in the expansion/decoder block
        x = self.maxpool(x)     # then do max pooling on x to pass to next level down
        
        return x, x_copy        # return the output from the block as well as the copy. pass to lower level.
    
    
"""
----------------------------------------
DecoderBlock:
    Describe
    
----------------------------------------
"""
    
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_size):
        super(DecoderBlock, self).__init__()
        
        self.upsample  = nn.Upsample(size=upsample_size, scale_factor=(2,2), mode="bilinear")
        self.convBnRe1 = ConvBnRelu(in_channels, out_channels, kernel_size=(3,3), stride=1, padding=0)
        self.convBnRe2 = ConvBnRelu(in_channels, out_channels, kernel_size=(3,3), stride=1, padding=0)
        
    def _crop_concat(self, x_upsampled, x_copy):
    
        # input is upsampled tensor from x = self.upsample(x) line in forward()
        # also the copied tensor from the corresponding contracting layer, which we
        # want to crop to the size of x_upsampled, and then concatenate with x_upsampled.
        
        diffY = x_copy.size()[2] - x_upsampled.size()[2]
        diffX = x_copy.size()[3] - x_upsampled.size()[3]

        x_upsampled = F.pad(x_upsampled, (diffX // 2, diffX - diffX//2,
                                diffY // 2, diffY - diffY//2))
        
        x = torch.cat([x_copy, x_upsampled], dim=1)
        
        """
        Alternative way to crop/concat: from carvana unet code
        """
#        c = ( x_copy.size()[2] - x_upsampled.size()[2] ) // 2
#        x_copy = F.pad(x_upsampled, (-c, -c, -c, -c))
#        return torch.cat((x_upsampled, x_copy), 1)

        return x
            

    def forward(self, x, x_across):
        x_up = self.upsample(x)
        x = self._crop_concat(x_up, x_across)  # x_up = upsampled tensor, x_across = x_copy from corresponding encoder level 
        x = self.convBnRe1(x)
        x = self.convBnRe2(x)
        return x
    
    
"""
----------------------------------------
ConvBnRelu:
    
    
----------------------------------------
"""    
    
class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d()
        self.bn   = nn.BatchNorm2d()
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
        
        
        
        
        
        
        
        
        
        
        
        
        
        