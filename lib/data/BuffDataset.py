


import os
import random

import numpy as np 
from PIL import Image, ImageOps
import cv2
import torch
import json
import trimesh
import logging

from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn.functional as F


produce_normal_maps = True 
produce_coarse_depth_maps = True
produce_fine_depth_maps = True
produce_parse_maps = True



class BuffDataset(Dataset):


    def __init__(self, opt):
        self.opt = opt
        self.projection_mode = 'orthogonal'
        self.subjects = np.loadtxt("buff_subject_testing.txt", dtype=str).tolist()

        self.root = "buff_dataset/buff_rgb_images"



        # PIL to tensor
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(), #  ToTensor converts input to a shape of (C x H x W) in the range [0.0, 1.0] for each dimension
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalise with mean of 0.5 and std_dev of 0.5 for each dimension. Finally range will be [-1,1] for each dimension
        ])


    def __len__(self):
        return len(self.subjects)









    def get_item(self, index):

        subject = self.subjects[index]

        param_path = os.path.join(self.root, "rendered_params_" +  subject + ".npy" ) 
        render_path = os.path.join(self.root, "rendered_image_" +  subject + ".png" ) 
        mask_path = os.path.join(self.root, "rendered_mask_" +  subject + ".png" ) 

        if produce_normal_maps:
            nmlF_high_res_path =  os.path.join( "buff_dataset/buff_normal_maps" , "rendered_nmlF_" + subject + ".npy"  )
            nmlB_high_res_path =  os.path.join( "buff_dataset/buff_normal_maps" , "rendered_nmlB_" + subject + ".npy" )

        if produce_coarse_depth_maps:
            coarse_depth_map_path =  os.path.join( "buff_dataset/buff_depth_maps" , "rendered_coarse_depthmap_" + subject + ".npy"  )
        
        if produce_fine_depth_maps:
            fine_depth_map_path =  os.path.join( "buff_dataset/buff_depth_maps" , "rendered_depthmap_" + subject + ".npy"  )

        if produce_parse_maps:
            parse_map_path =  os.path.join( "buff_dataset/buff_parse_maps" , "rendered_parse_" + subject + ".npy"  ) 


        load_size_associated_with_scale_factor = 1024

        # get params
        param = np.load(param_path, allow_pickle=True)  # param is a np.array that looks similar to a dict.  # ortho_ratio = 0.4 , e.g. scale or y_scale = 0.961994278, e.g. center or vmed = [-1.0486  92.56105  1.0101 ]
        center = param.item().get('center') # is camera 3D center position in the 3D World point space (without any rotation being applied).
        R = param.item().get('R')   # R is used to rotate the CAD model according to a given pitch and yaw.
        scale_factor = param.item().get('scale_factor') # is camera 3D center position in the 3D World point space (without any rotation being applied).



        b_range = load_size_associated_with_scale_factor / scale_factor 
        b_center = center
        b_min = b_center - b_range/2
        b_max = b_center + b_range/2



        # extrinsic is used to rotate the 3D points according to our specified pitch and yaw
        translate = -center.reshape(3, 1)
        extrinsic = np.concatenate([R, translate], axis=1)  # when applied on the 3D pts, the rotation is done first, then the translation
        extrinsic = np.concatenate([extrinsic, np.array([0, 0, 0, 1]).reshape(1, 4)], 0)
        
        scale_intrinsic = np.identity(4)
        scale_intrinsic[0, 0] = 1.0 * scale_factor  
        scale_intrinsic[1, 1] = -1.0 * scale_factor  
        scale_intrinsic[2, 2] = 1.0 * scale_factor   

        # Match image pixel space to image uv space   
        uv_intrinsic = np.identity(4)
        uv_intrinsic[0, 0] = 1.0 / float(load_size_associated_with_scale_factor // 2)  
        uv_intrinsic[1, 1] = 1.0 / float(load_size_associated_with_scale_factor // 2)  
        uv_intrinsic[2, 2] = 1.0 / float(load_size_associated_with_scale_factor // 2) 

        mask = Image.open(mask_path).convert('L')  
        render = Image.open(render_path).convert('RGB')


        intrinsic = np.matmul(uv_intrinsic, scale_intrinsic)
        calib = torch.Tensor(np.matmul(intrinsic, extrinsic)).float() 
        extrinsic = torch.Tensor(extrinsic).float()

        mask = transforms.ToTensor()(mask).float()

        render = self.to_tensor(render)  # normalize render from [0,255] to [-1,1]
        render = mask.expand_as(render) * render


        # resize the 1024 x 1024 image to 512 x 512 for the low-resolution pifu
        render_low_pifu = F.interpolate(torch.unsqueeze(render,0), size=(self.opt.loadSizeGlobal,self.opt.loadSizeGlobal) )
        mask_low_pifu = F.interpolate(torch.unsqueeze(mask,0), size=(self.opt.loadSizeGlobal,self.opt.loadSizeGlobal) )
        render_low_pifu = render_low_pifu[0]
        mask_low_pifu = mask_low_pifu[0]

        if produce_normal_maps:
            nmlF_high_res = np.load(nmlF_high_res_path) # shape of [3, 1024,1024]
            nmlB_high_res = np.load(nmlB_high_res_path) # shape of [3, 1024,1024]
            nmlF_high_res = torch.Tensor(nmlF_high_res)
            nmlB_high_res = torch.Tensor(nmlB_high_res)
            nmlF_high_res = mask.expand_as(nmlF_high_res) * nmlF_high_res
            nmlB_high_res = mask.expand_as(nmlB_high_res) * nmlB_high_res

            nmlF  = F.interpolate(torch.unsqueeze(nmlF_high_res,0), size=(self.opt.loadSizeGlobal,self.opt.loadSizeGlobal) )
            nmlF = nmlF[0]
            nmlB  = F.interpolate(torch.unsqueeze(nmlB_high_res,0), size=(self.opt.loadSizeGlobal,self.opt.loadSizeGlobal) )
            nmlB = nmlB[0]
        else:
            nmlF_high_res = nmlB_high_res = 0
            nmlF = nmlB = 0 

        if produce_coarse_depth_maps:
            coarse_depth_map = np.load(coarse_depth_map_path)
            coarse_depth_map = torch.Tensor(coarse_depth_map)
            coarse_depth_map = mask.expand_as(coarse_depth_map) * coarse_depth_map # shape of [C,H,W]

        else:
            coarse_depth_map = 0

        if produce_fine_depth_maps:
            fine_depth_map = np.load(fine_depth_map_path)
            fine_depth_map = torch.Tensor(fine_depth_map)
            fine_depth_map = mask.expand_as(fine_depth_map) * fine_depth_map # shape of [C,H,W]
            depth_map_low_res = F.interpolate(torch.unsqueeze(fine_depth_map,0), size=(self.opt.loadSizeGlobal,self.opt.loadSizeGlobal) )
            depth_map_low_res = depth_map_low_res[0] 
        else: 
            fine_depth_map = 0
            depth_map_low_res = 0 

        if produce_parse_maps:
            human_parse_map = np.load(parse_map_path) # shape of (1024,1024)
            human_parse_map = torch.Tensor(human_parse_map)
            human_parse_map = torch.unsqueeze(human_parse_map,0) # shape of (1,1024,1024)
            human_parse_map = mask.expand_as(human_parse_map) * human_parse_map # shape of [1,H,W]

            human_parse_map_0 = (human_parse_map == 0).float()
            human_parse_map_1 = (human_parse_map == 1).float()
            human_parse_map_2 = (human_parse_map == 2).float() 
            human_parse_map_3 = (human_parse_map == 3).float() 
            human_parse_map_4 = (human_parse_map == 4).float() 
            human_parse_map_5 = (human_parse_map == 5).float()
            human_parse_map_6 = (human_parse_map == 6).float()
            human_parse_map_list = [human_parse_map_0, human_parse_map_1, human_parse_map_2, human_parse_map_3, human_parse_map_4, human_parse_map_5, human_parse_map_6]

            human_parse_map = torch.cat(human_parse_map_list, dim=0)
            human_parse_map = F.interpolate(torch.unsqueeze(human_parse_map,0), size=(self.opt.loadSizeGlobal,self.opt.loadSizeGlobal) )
            human_parse_map = human_parse_map[0] 
        else:
            human_parse_map = 0




        center_indicator = np.zeros([1,1024,1024])
        center_indicator[:, 511:513,511:513] = 1.0
        center_indicator = torch.Tensor(center_indicator).float()




        return {
            'name': subject,
            'render_path':render_path,
            'render_low_pifu': render_low_pifu,
            'mask_low_pifu': mask_low_pifu,
            'original_high_res_render':render,
            'mask':mask,
            'calib': calib,
            'nmlF_high_res':nmlF_high_res,
            'nmlB_high_res':nmlB_high_res,
            'center_indicator':center_indicator,
            'coarse_depth_map':coarse_depth_map,
            'depth_map':fine_depth_map,
            'depth_map_low_res':depth_map_low_res,
            'human_parse_map':human_parse_map,
            'nmlF':nmlF,
            'nmlB':nmlB,
            'b_min':b_min,
            'b_max':b_max
                }



    def __getitem__(self, index):
        return self.get_item(index)




    












