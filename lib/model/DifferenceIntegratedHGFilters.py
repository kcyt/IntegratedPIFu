'''
MIT License

Copyright (c) 2019 Shunsuke Saito, Zeng Huang, and Ryota Natsume

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
import torch
import torch.nn as nn 
import torch.nn.functional as F 
from ..net_util import conv3x3

class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, norm='batch'):
        super(ConvBlock, self).__init__()

        # conv3x3 will not change the size (height and width) of the input feature maps 
        self.conv1 = conv3x3(in_planes, int(out_planes / 2))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4))

        if norm == 'batch':
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.bn2 = nn.BatchNorm2d(int(out_planes / 2))
            self.bn3 = nn.BatchNorm2d(int(out_planes / 4))
            self.bn4 = nn.BatchNorm2d(in_planes)
        elif norm == 'group':
            self.bn1 = nn.GroupNorm(32, in_planes)
            self.bn2 = nn.GroupNorm(32, int(out_planes / 2))
            self.bn3 = nn.GroupNorm(32, int(out_planes / 4))
            self.bn4 = nn.GroupNorm(32, in_planes)
        
        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                self.bn4,
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes,
                          kernel_size=1, stride=1, bias=False),
            )
        else:
            self.downsample = None
    
    def forward(self, x):
        residual = x

        out1 = self.conv1(F.relu(self.bn1(x), True))
        out2 = self.conv2(F.relu(self.bn2(out1), True))
        out3 = self.conv3(F.relu(self.bn3(out2), True))

        out3 = torch.cat([out1, out2, out3], 1)

        if self.downsample is not None:
            residual = self.downsample(residual)
        
        out3 += residual

        return out3

class HourGlass(nn.Module):
    def __init__(self, depth, n_features, norm='batch'):
        super(HourGlass, self).__init__()
        self.depth = depth
        self.features = n_features
        self.norm = norm

        self._generate_network(self.depth)

    def _generate_network(self, level):
        self.add_module('b1_' + str(level), ConvBlock(self.features, self.features, norm=self.norm))
        self.add_module('b2_' + str(level), ConvBlock(self.features, self.features, norm=self.norm))

        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvBlock(self.features, self.features, norm=self.norm))

        self.add_module('b3_' + str(level), ConvBlock(self.features, self.features, norm=self.norm))

    def _forward(self, level, inp):
        # upper branch
        up1 = inp 
        up1 = self._modules['b1_' + str(level)](up1)

        # lower branch
        low1 = F.avg_pool2d(inp, 2, stride=2)
        low1 = self._modules['b2_' + str(level)](low1)

        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = low1
            low2 = self._modules['b2_plus_' + str(level)](low2)

        low3 = low2
        low3 = self._modules['b3_' + str(level)](low3)

        up2 = F.interpolate(low3, scale_factor=2, mode='bicubic', align_corners=True)
        # up2 = F.interpolate(low3, scale_factor=2, mode='bilinear')

        return up1 + up2
    
    def forward(self, x):
        return self._forward(self.depth, x)
        



class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(mid_channels),
            nn.Tanh(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.Tanh()
        )

    def forward(self, x):
        return self.double_conv(x)



class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, x1_in_channels, out_channels, x1_intermediate_channels = 192 , x2_in_channels = 128 , scale_factor=2):
        super().__init__()

        self.up = nn.ConvTranspose2d(x1_in_channels, x1_intermediate_channels, kernel_size=scale_factor, stride=scale_factor)

        self.conv = DoubleConv(x1_intermediate_channels + x2_in_channels,  out_channels)

    def forward(self, x1, x2):
        # x1 and x2 must each have no. of channels == in_channels. x1 is the one that will be upsampled
        x1 = self.up(x1)
        # input is CHW

        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)






class DifferenceIntegratedHGFilter(nn.Module):
    def __init__(self, stack, depth, in_ch, last_ch, norm='batch', down_type='conv64', use_sigmoid=True, no_first_down_sampling = False):
        super(DifferenceIntegratedHGFilter, self).__init__()
        self.n_stack = stack
        self.use_sigmoid = use_sigmoid
        self.depth = depth
        self.last_ch = last_ch 
        self.norm = norm
        self.down_type = down_type
        self.no_first_down_sampling = no_first_down_sampling

        self.stem = nn.Conv2d(in_ch, 256 , kernel_size=7, stride=2, padding=3) # will downsample the feature map width and height by half

        self.upscale = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        
        self.conv1 = nn.Conv2d(256, 128 , kernel_size=3, stride=1, padding=1)
 
        last_ch = self.last_ch

        self.bn0 = nn.InstanceNorm2d(256 )
        self.bn1 = nn.InstanceNorm2d(128 )

        self.bn3 = nn.InstanceNorm2d(64+256)

        self.conv4 = nn.Conv2d(128+256, 64+256, kernel_size=3, stride=1, padding=1)

        self.conv5 = nn.Conv2d(64+256, 256, kernel_size=1, stride=1, padding=0)





    def forward(self, x, netG_output_map):


        x = F.leaky_relu(self.bn0(self.stem(x)))
        
        x = F.leaky_relu(self.bn1(self.conv1(x)))

        upscaled_netG_output_map = self.upscale(netG_output_map)
        x = torch.cat( [x, upscaled_netG_output_map], dim=1 )

        normx = 0 

        x = F.leaky_relu(self.bn3(self.conv4(x)) )
        x = self.conv5(x) + upscaled_netG_output_map 

        outputs = [x]
 
        return outputs, normx
    