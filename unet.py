#!/usr/bin/python
# full assembly of the sub-parts to form the complete net

import torch
import torch.nn as nn
import torch.nn.functional as F

# python 3 confusing imports :(
from unet_parts import *

'''
class UNet(nn.Module):
    def __init__(self, n_channels, shrink):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64 // shrink)
        self.down1 = down(64 // shrink, 128 // shrink)
        self.down2 = down(128 // shrink, 256 // shrink)
        self.down3 = down(256 // shrink, 512 // shrink)
        self.down4 = down(512 // shrink, 512 // shrink)
        self.up1 = up(1024 // shrink, 256 // shrink)
        self.up2 = up(512 // shrink, 128 // shrink)
        self.up3 = up(256 // shrink, 64 // shrink)
        self.up4 = up(128 // shrink, 64 // shrink)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        out1 = self.up1(x5, x4) #8
        out2 = self.up2(out1, x3) #16
        out3 = self.up3(out2, x2) #32
        return [out1, out2, out3]
'''

class UNet(nn.Module):
    def __init__(self, n_channels, shrink):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64 // shrink) #64
        self.down1 = down(64 // shrink, 128 // shrink) #32
        self.down2 = down(128 // shrink, 256 // shrink) #16
        self.down3 = down(256 // shrink, 512 // shrink) #8
        self.down4 = down(512 // shrink, 1024 // shrink) #4
        self.down5 = down(1024 // shrink, 1024 // shrink) #2
        self.up1 = up(2048 // shrink, 512 // shrink) #4
        self.up2 = up(1024 // shrink, 256 // shrink) #8
        self.up3 = up(512 // shrink, 128 // shrink) #16
        self.up4 = up(256 // shrink, 64 // shrink) #32
        self.up5 = up(128 // shrink, 64 // shrink) #64

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        out1 = self.up1(x6, x5) #4
        out2 = self.up2(out1, x4) #8
        out3 = self.up3(out2, x3) #16
        out4 = self.up4(out3, x2) #32
        #print(out1.shape,out2.shape,out3.shape,out4.shape)
        return [out1, out2, out3, out4]
