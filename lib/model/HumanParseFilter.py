
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
from ..net_util import init_net
import cv2





class HumanParseFilter(nn.Module):


    def __init__(self, 
                 opt, 
                 criteria={'err': nn.CrossEntropyLoss() }
                 ):
        super(HumanParseFilter, self).__init__()

        self.name = 'humanparsefilter'
        self.criteria = criteria

        self.opt = opt
        
        from .UNet import UNet
        n_channels = 3 
        if self.opt.use_normal_map_for_parse_training:
            n_channels = n_channels + 3



        self.image_filter = UNet(n_channels=n_channels, n_classes=7, bilinear=False )



        self.im_feat_list = []

        init_net(self) # initialise weights 


 


    def filter(self, images):
        '''
        apply a fully convolutional network to images.
        the resulting feature will be stored.
        args:
            images: [B, C, H, W]
        '''

        self.im_feat_list  = self.image_filter(images) 

        


    def get_im_feat(self):

        return self.im_feat_list

    def generate_parse_map(self):

        im_feat = self.im_feat_list # [B, C, H, W]
        im_feat = torch.argmax(im_feat, dim=1) # [B,H,W]

        return im_feat



    def get_error(self):
        '''
        return the loss given the ground truth labels and prediction
        '''
        error = {}

        error['Err'] = self.criteria['err'](self.im_feat_list, self.groundtruth_parsemap)

        return error


    def forward(self, images, groundtruth_parsemap ):


        self.filter(images)

        self.groundtruth_parsemap = torch.argmax(groundtruth_parsemap, dim=1)  # [B, H, W] 
            
        err = self.get_error()

        return err
