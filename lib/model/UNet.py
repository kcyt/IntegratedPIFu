

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 









class LargeKernelOutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LargeKernelOutConv, self).__init__()
        self.conv_initial = nn.Conv2d(in_channels, 32, kernel_size=5, padding=2)
        self.conv_final = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv_final( self.conv_initial(x) )



class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)




class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(scale_factor),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, scale_factor=2):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=scale_factor, stride=scale_factor)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)








class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, last_op=None):
        super(UNet, self).__init__()
        self.n_channels = n_channels

        self.chn = 64

        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(self.n_channels, self.chn)
        self.down1 = Down(self.chn, self.chn*2)



        self.down2 = Down(self.chn*2, self.chn*4)
        self.down3 = Down(self.chn*4, self.chn*8)
        factor = 2 if bilinear else 1
        self.down4 = Down(self.chn*8, self.chn*16 // factor)
        self.up1 = Up(self.chn*16, self.chn*8 // factor, bilinear)
        self.up2 = Up(self.chn*8, self.chn*4 // factor, bilinear)
        self.up3 = Up(self.chn*4, self.chn*2 // factor, bilinear)
        self.up4 = Up(self.chn*2, self.chn, bilinear)
        self.outc = OutConv(self.chn, n_classes)

        self.last_op = last_op





    def forward(self, x):

        x1 = self.inc(x)
        x2 = self.down1(x1)


        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        if self.last_op is None:
            logits = self.outc(x)
        else:
            logits = self.last_op(self.outc(x))
        return logits








class DifferenceUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, last_op=None, scale_factor=2):
        super(DifferenceUNet, self).__init__()
        self.n_channels = n_channels

        self.n_classes = n_classes
        self.bilinear = bilinear

        self.scale_factor = scale_factor
        self.chn = 64


        self.inc = DoubleConv(self.n_channels, self.chn)
        self.down1 = Down(self.chn, self.chn*2, scale_factor=self.scale_factor)
        self.down2 = Down(self.chn*2, self.chn*4, scale_factor=self.scale_factor)
        self.down3 = Down(self.chn*4, self.chn*8, scale_factor=self.scale_factor)
        factor = 2 if bilinear else 1
        self.down4 = Down(self.chn*8, self.chn*16 // factor, scale_factor=self.scale_factor)
        self.up1 = Up(self.chn*16, self.chn*8 // factor, bilinear, scale_factor=self.scale_factor)
        self.up2 = Up(self.chn*8, self.chn*4 // factor, bilinear, scale_factor=self.scale_factor)
        self.up3 = Up(self.chn*4, self.chn*2 // factor, bilinear, scale_factor=self.scale_factor)
        self.up4 = Up(self.chn*2, self.chn, bilinear, scale_factor=self.scale_factor)
    

        self.outc = LargeKernelOutConv(self.chn, n_classes)
        self.last_op = last_op


    def forward(self, x):
        temp_x = x 


        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)



        if self.last_op is None:

            logits = temp_x[:,3:4,:,:] + self.outc(x)
        else:

            logits = temp_x[:,3:4,:,:]  + self.last_op(self.outc(x))
        return logits








