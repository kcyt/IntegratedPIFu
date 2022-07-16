


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





class NormalDataset(Dataset):


    def __init__(self, opt, evaluation_mode=False):
        self.opt = opt
        self.training_subject_list = np.loadtxt("/mnt/lustre/kennard.chan/getTestSet/train_set_list.txt", dtype=str).tolist()


        if evaluation_mode:
            print("Overwriting self.training_subject_list!")
            self.training_subject_list = np.loadtxt("/mnt/lustre/kennard.chan/getTestSet/test_set_list.txt", dtype=str).tolist()
            self.is_train = False


        self.groundtruth_normal_map_directory = "/mnt/lustre/kennard.chan/render_THuman_with_blender/buffer_normal_maps_of_full_mesh"
        self.root = "/mnt/lustre/kennard.chan/render_THuman_with_blender/buffer_fixed_full_mesh"


        self.subjects = self.training_subject_list   



        self.img_files = []
        for training_subject in self.subjects:
            subject_render_folder = os.path.join(self.root, training_subject)
            subject_render_paths_list = [  os.path.join(subject_render_folder,f) for f in os.listdir(subject_render_folder) if "image" in f   ]
            self.img_files = self.img_files + subject_render_paths_list
        self.img_files = sorted(self.img_files)



        # PIL to tensor
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(), #  ToTensor converts input to a shape of (C x H x W) in the range [0.0, 1.0] for each dimension
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalise with mean of 0.5 and std_dev of 0.5 for each dimension. Finally range will be [-1,1] for each dimension
        ])


    def __len__(self):
        return len(self.img_files)







    def get_item(self, index):

        img_path = self.img_files[index]
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        # get yaw
        yaw = img_name.split("_")[-1]
        yaw = int(yaw)

        # get subject
        subject = img_path.split('/')[-2] # e.g. "0507"

        render_path = os.path.join(self.root, subject, "rendered_image_" + "{0:03d}".format(yaw) + ".png"  )
        mask_path = os.path.join(self.root, subject, "rendered_mask_" + "{0:03d}".format(yaw) + ".png"  )
        

        nmlF_high_res_path =  os.path.join(self.groundtruth_normal_map_directory, subject, "rendered_nmlF_" + "{0:03d}".format(yaw) + ".exr"  )
        nmlB_high_res_path =  os.path.join(self.groundtruth_normal_map_directory, subject, "rendered_nmlB_" + "{0:03d}".format(yaw) + ".exr"  )



        mask = Image.open(mask_path).convert('L') 
        render = Image.open(render_path).convert('RGB')

        mask = transforms.ToTensor()(mask).float()

        render = self.to_tensor(render)  # normalize render from [0,255] to [-1,1]
        render = mask.expand_as(render) * render


        # resize the 1024 x 1024 image to 512 x 512 for the low-resolution pifu
        render_low_pifu = F.interpolate(torch.unsqueeze(render,0), size=(self.opt.loadSizeGlobal,self.opt.loadSizeGlobal) )
        mask_low_pifu = F.interpolate(torch.unsqueeze(mask,0), size=(self.opt.loadSizeGlobal,self.opt.loadSizeGlobal) )
        render_low_pifu = render_low_pifu[0]
        mask_low_pifu = mask_low_pifu[0]


        nmlF_high_res = cv2.imread(nmlF_high_res_path, cv2.IMREAD_UNCHANGED).astype(np.float32) # numpy of [1024,1024,3]
        nmlB_high_res = cv2.imread(nmlB_high_res_path, cv2.IMREAD_UNCHANGED).astype(np.float32) 
        nmlB_high_res = nmlB_high_res[:,::-1,:].copy()
 

        
        nmlF_high_res = np.transpose(nmlF_high_res, [2,0,1]  ) # change to shape of [3,1024,1024]
        nmlF_high_res = torch.Tensor(nmlF_high_res)
        nmlF_high_res = mask.expand_as(nmlF_high_res) * nmlF_high_res

        nmlB_high_res = np.transpose(nmlB_high_res, [2,0,1]  ) # change to shape of [3,1024,1024]
        nmlB_high_res = torch.Tensor(nmlB_high_res)
        nmlB_high_res = mask.expand_as(nmlB_high_res) * nmlB_high_res



        return {
            'name': subject,
            'render_path':render_path,
            'render_low_pifu': render_low_pifu,
            'mask_low_pifu': mask_low_pifu,
            'original_high_res_render':render,
            'nmlB_high_res':nmlB_high_res,
            'nmlF_high_res':nmlF_high_res

                }



    def __getitem__(self, index):
        return self.get_item(index)




    












