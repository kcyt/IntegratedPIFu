
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
from .BasePIFuNet import BasePIFuNet
from .MLP import MLP
from .DepthNormalizer import DepthNormalizer
from .HGFilters import HGFilter
from ..net_util import init_net
from ..net_util import CustomBCELoss
from ..networks import define_G
import cv2

class HGPIFuNetwNML(BasePIFuNet):
    '''
    HGPIFu uses stacked hourglass as an image encoder.
    '''

    def __init__(self, 
                 opt, 
                 projection_mode='orthogonal',
                 criteria={'occ': nn.MSELoss()},
                 use_High_Res_Component = False
                 ):
        super(HGPIFuNetwNML, self).__init__(
            projection_mode=projection_mode,
            criteria=criteria)

        self.name = 'hg_pifu_low_res'

        self.opt = opt

        self.use_High_Res_Component = use_High_Res_Component

        in_ch = 3
        try:
            if opt.use_front_normal: 
                in_ch += 3
            if opt.use_back_normal: 
                in_ch += 3
        except:
            pass


        if self.opt.use_depth_map and self.opt.depth_in_front:
            if not self.use_High_Res_Component:
                in_ch += 1
            elif self.use_High_Res_Component and self.opt.allow_highres_to_use_depth:
                in_ch += 1
            else:
                pass

        if self.opt.use_human_parse_maps:
            if not self.use_High_Res_Component:
                if self.opt.use_groundtruth_human_parse_maps:
                    in_ch += 6
                else:
                    in_ch += 7
            else:
                pass
        



        if self.use_High_Res_Component:
            from .DifferenceIntegratedHGFilters import DifferenceIntegratedHGFilter
            self.image_filter = DifferenceIntegratedHGFilter(1, 2, in_ch, 256,   
                                         opt.norm, opt.hg_down, False) 
        else:
            self.image_filter = HGFilter(opt.num_stack_low_res, opt.hg_depth_low_res, in_ch, opt.hg_dim_low_res,   
                                         opt.norm, opt.hg_down, False) 
       
        if self.opt.use_depth_map and not self.opt.depth_in_front:
            self.opt.mlp_dim_low_res[0] = self.opt.mlp_dim_low_res[0] + 1 # plus 1 for the depthmap. 
            print("Overwriting self.opt.mlp_dim_low_res to add in 1 dim for depth map!")



        self.mlp = MLP(
            filter_channels=self.opt.mlp_dim_low_res,  
            merge_layer=self.opt.merge_layer_low_res,  
            res_layers=self.opt.mlp_res_layers_low_res,   
            norm="no_norm",
            last_op=nn.Sigmoid())

        self.spatial_enc = DepthNormalizer(opt)

        self.im_feat_list = []
        self.tmpx = None
        self.normx = None
        self.phi = None

        self.intermediate_preds_list = []

        init_net(self) # initialise weights  

        self.netF = None
        self.netB = None


        self.nmlF = None
        self.nmlB = None

        self.gamma = None

        self.current_depth_map  = None





    def filter(self, images, nmlF=None, nmlB = None, current_depth_map = None, netG_output_map = None, human_parse_map=None, mask_low_res_tensor=None, mask_high_res_tensor=None):
        '''
        apply a fully convolutional network to images.
        the resulting feature will be stored.
        args:
            images: [B, C, H, W]
        '''
        if self.opt.use_depth_map and not self.opt.depth_in_front:
            self.current_depth_map = current_depth_map

        self.mask_high_res_tensor = mask_high_res_tensor
        self.mask_low_res_tensor = mask_low_res_tensor


        nmls = []
        # if you wish to train jointly, remove detach etc.
        with torch.no_grad():
            if self.opt.use_front_normal:
                if nmlF == None:
                    raise Exception("NORMAL MAPS ARE MISSING!!")

                self.nmlF = nmlF
                nmls.append(self.nmlF)
            if self.opt.use_back_normal:
                if nmlB == None:
                    raise Exception("NORMAL MAPS ARE MISSING!!")

                self.nmlB = nmlB
                nmls.append(self.nmlB)
        
        # Concatenate the input image with the two normals maps together
        if len(nmls) != 0:
            nmls = torch.cat(nmls,1)
            if images.size()[2:] != nmls.size()[2:]:
                nmls = nn.Upsample(size=images.size()[2:], mode='bilinear', align_corners=True)(nmls)
            images = torch.cat([images,nmls],1)

        if self.opt.use_depth_map and self.opt.depth_in_front and (current_depth_map is not None):
            images = torch.cat([images, current_depth_map], 1) 

        if self.opt.use_human_parse_maps and (human_parse_map is not None) :
            images = torch.cat([images, human_parse_map], 1) 


        if self.use_High_Res_Component: 
            self.im_feat_list, self.normx = self.image_filter(images, netG_output_map) 
        else:
            self.im_feat_list, self.normx = self.image_filter(images) 

        if not self.training:
            self.im_feat_list = [self.im_feat_list[-1]]
        
    def query(self, points, calibs, transforms=None, labels=None, update_pred=True, update_phi=True):
        '''
        given 3d points, we obtain 2d projection of these given the camera matrices.
        filter needs to be called beforehand.
        the prediction is stored to self.preds
        args:
            points: [B, 3, N] 3d points in world space
            calibs: [B, 3, 4] calibration matrices for each image. If calibs is [B,3,4], it is fine as well.
            transforms: [B, 2, 3] image space coordinate transforms
            labels: [B, C, N] ground truth labels (for supervision only)
        return:
            [B, C, N] prediction
        '''
        xyz = self.projection(points, calibs, transforms) # [B, 3, N]
        xy = xyz[:, :2, :] # [B, 2, N]

        if self.use_High_Res_Component and self.opt.use_mask_for_rendering_high_res and (self.mask_high_res_tensor is not None):
            mask_values = self.index(self.mask_high_res_tensor , xy) 
        if (not self.use_High_Res_Component) and self.opt.use_mask_for_rendering_low_res and (self.mask_low_res_tensor is not None):
            mask_values = self.index(self.mask_low_res_tensor , xy) 

        # if the point is outside bounding box, return outside.
        in_bb = (xyz >= -1) & (xyz <= 1) # [B, 3, N]
        in_bb = in_bb[:, 0, :] & in_bb[:, 1, :] & in_bb[:, 2, :] # [B, N]
        in_bb = in_bb[:, None, :].detach().float() # [B, 1, N]

        is_zero_bool = (xyz == 0) # [B, 3, N]; remove the (0,0,0) point that has been used to discard unwanted sample pts
        is_zero_bool = is_zero_bool[:, 0, :] & is_zero_bool[:, 1, :] & is_zero_bool[:, 2, :] # [B, N]
        not_zero_bool = torch.logical_not(is_zero_bool)
        not_zero_bool = not_zero_bool[:, None, :].detach().float() # [B, 1, N]

        if labels is not None:
            self.labels = in_bb * labels # [B, 1, N]
            self.labels = not_zero_bool * self.labels

            size_of_batch = self.labels.shape[0]

        sp_feat = self.spatial_enc(xyz, calibs=calibs) # sp_feat is the normalized z value. (x and y are removed)

        intermediate_preds_list = []

        phi = None
        for i, im_feat in enumerate(self.im_feat_list):

            if self.opt.use_depth_map and not self.opt.depth_in_front:
                point_local_feat_list = [self.index(im_feat, xy), self.index(self.current_depth_map , xy) ,sp_feat] 
            else:
                point_local_feat_list = [self.index(im_feat, xy), sp_feat] # z_feat has already gone through a round of indexing. 'point_local_feat_list' should have shape of [batch_size, 272, num_of_points]     
            point_local_feat = torch.cat(point_local_feat_list, 1)
            pred, phi = self.mlp(point_local_feat) # phi is activations from an intermediate layer of the MLP
            pred = in_bb * pred
            pred = not_zero_bool * pred
            if self.use_High_Res_Component and self.opt.use_mask_for_rendering_high_res and (self.mask_high_res_tensor is not None):
                pred = mask_values * pred
            if (not self.use_High_Res_Component) and self.opt.use_mask_for_rendering_low_res and (self.mask_low_res_tensor is not None):
                pred = mask_values * pred

            intermediate_preds_list.append(pred)
        
        if update_phi:
            self.phi = phi

        if update_pred:
            self.intermediate_preds_list = intermediate_preds_list
            self.preds = self.intermediate_preds_list[-1]

    def calc_normal(self, points, calibs, transforms=None, labels=None, delta=0.01, fd_type='forward'):
        '''
        return surface normal in 'model' space.
        it computes normal only in the last stack.
        note that the current implementation use forward difference.
        args:
            points: [B, 3, N] 3d points in world space
            calibs: [B, 3, 4] calibration matrices for each image
            transforms: [B, 2, 3] image space coordinate transforms
            delta: perturbation for finite difference
            fd_type: finite difference type (forward/backward/central) 
        '''
        pdx = points.clone()
        pdx[:,0,:] += delta
        pdy = points.clone()
        pdy[:,1,:] += delta
        pdz = points.clone()
        pdz[:,2,:] += delta

        if labels is not None:
            self.labels_nml = labels

        points_all = torch.stack([points, pdx, pdy, pdz], 3)
        points_all = points_all.view(*points.size()[:2],-1)
        xyz = self.projection(points_all, calibs, transforms)
        xy = xyz[:, :2, :]

        im_feat = self.im_feat_list[-1]
        sp_feat = self.spatial_enc(xyz, calibs=calibs)

        point_local_feat_list = [self.index(im_feat, xy), sp_feat]            
        point_local_feat = torch.cat(point_local_feat_list, 1)

        pred = self.mlp(point_local_feat)[0]

        pred = pred.view(*pred.size()[:2],-1,4) # (B, 1, N, 4)

        # divide by delta is omitted since it's normalized anyway
        dfdx = pred[:,:,:,1] - pred[:,:,:,0]
        dfdy = pred[:,:,:,2] - pred[:,:,:,0]
        dfdz = pred[:,:,:,3] - pred[:,:,:,0]

        nml = -torch.cat([dfdx,dfdy,dfdz], 1)
        nml = F.normalize(nml, dim=1, eps=1e-8)

        self.nmls = nml

    def get_im_feat(self):
        '''
        return the image filter in the last stack
        return:
            [B, C, H, W]
        '''
        return self.im_feat_list[-1]


    def get_error(self,points=None):
        '''
        return the loss given the ground truth labels and prediction
        '''
        error = {}
        error['Err(occ)'] = 0
        for preds in self.intermediate_preds_list:
            error['Err(occ)'] += self.criteria['occ'](preds, self.labels)
        
        error['Err(occ)'] /= len(self.intermediate_preds_list)
        
        if self.nmls is not None and self.labels_nml is not None:
            error['Err(nml)'] = self.criteria['nml'](self.nmls, self.labels_nml)


        return error

    def forward(self, images, points, calibs, labels, points_nml=None, labels_nml=None, nmlF = None, nmlB = None, current_depth_map=None, netG_output_map = None, human_parse_map=None, mask_low_res_tensor=None, mask_high_res_tensor=None):
        self.filter(images, nmlF = nmlF, nmlB = nmlB, current_depth_map = current_depth_map, netG_output_map = netG_output_map, human_parse_map=human_parse_map, mask_low_res_tensor=mask_low_res_tensor, mask_high_res_tensor=mask_high_res_tensor)
        self.query(points, calibs, labels=labels)
        if points_nml is not None and labels_nml is not None:
            self.calc_normal(points_nml, calibs, labels=labels_nml)
        res = self.get_preds()
            
        err = self.get_error()

        return err, res




