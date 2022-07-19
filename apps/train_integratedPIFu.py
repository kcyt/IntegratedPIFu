
import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
import io

from lib.options import BaseOptions
from lib.model import HGPIFuNetwNML 
from lib.data import TrainDataset
from lib.mesh_util import save_obj_mesh_with_color, reconstruction
from lib.geometry import index


seed = 0 
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


parser = BaseOptions()
opt = parser.parse()
gen_test_counter = 0


debug_mode = True # For internal use, not for publicly released version.
test_script_activate = False # Set to True to generate the test subject meshes
test_script_activate_option_use_BUFF_dataset = False # will only work if test_script_activate is True


# Whether to load model weights
load_model_weights = True
load_model_weights_for_high_res_too = False
load_model_weights_for_low_res_finetuning_config = 0 # 0 == No Load weights; 1 == Load, load optimizerG weights; 2 == Load, load optimizer_lowResFineTune weights
checkpoint_folder_to_load_low_res = 'apps/checkpoints/Date_15_Jul_22_Time_10_51_45' # Date_15_Jul_22_Time_10_51_45 is folder to load
checkpoint_folder_to_load_high_res = 'apps/checkpoints/Date_28_Jun_22_Time_02_49_38' # Date_28_Jun_22_Time_02_49_38 is folder to load
epoch_to_load_from_low_res = 24
epoch_to_load_from_high_res = 2



if opt.use_High_Res_Component:
    opt.sigma_low_resolution_pifu = opt.High_Res_Component_sigma
    print("Modifying sigma_low_resolution_pifu to {0} for high resolution component!".format(opt.High_Res_Component_sigma) )




class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def save_samples_truncted_prob(fname, points, prob):
    '''
    Save the visualization of sampling to a ply file.
    Red points represent positive predictions.
    Green points represent negative predictions.
    :param fname: File name to save
    :param points: [N, 3] array of points
    :param prob: [N, 1] array of predictions in the range [0~1]
    :return:
    '''
    r = (prob >= 0.5).reshape([-1, 1]) * 255
    g = (prob < 0.5).reshape([-1, 1]) * 255
    b = np.zeros(r.shape)

    to_save = np.concatenate([points, r, g, b], axis=-1)
    return np.savetxt(fname,
                      to_save,
                      fmt='%.6f %.6f %.6f %d %d %d',
                      comments='',
                      header=(
                          'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header').format(
                          points.shape[0])
                      )




def adjust_learning_rate(optimizer, epoch, lr, schedule, learning_rate_decay):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in schedule:
        lr *= learning_rate_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr





def gen_mesh(resolution, net, device, data, save_path, thresh=0.5, use_octree=True):
    
    calib_tensor = data['calib'].to(device=device)
    calib_tensor = torch.unsqueeze(calib_tensor,0)
    
    b_min = data['b_min']
    b_max = data['b_max']

    # low-resolution image that is required by both models
    image_low_tensor = data['render_low_pifu'].to(device=device)  
    image_low_tensor = image_low_tensor.unsqueeze(0)

    if opt.use_front_normal:
        nmlF_low_tensor = data['nmlF'].to(device=device)
        nmlF_low_tensor = nmlF_low_tensor.unsqueeze(0)
    else:
        nmlF_low_tensor = None


    if opt.use_back_normal:
        nmlB_low_tensor = data['nmlB'].to(device=device)
        nmlB_low_tensor = nmlB_low_tensor.unsqueeze(0)
    else:
        nmlB_low_tensor = None


    if opt.use_depth_map:
        depth_map_low_res = data['depth_map_low_res'].to(device=device)
        depth_map_low_res = depth_map_low_res.unsqueeze(0)
    else: 
        depth_map_low_res = None



    if opt.use_human_parse_maps:
        human_parse_map = data['human_parse_map'].to(device=device)
        human_parse_map = human_parse_map.unsqueeze(0)
    else:
        human_parse_map=None





    if opt.use_High_Res_Component:
        netG, highRes_netG = net
        net = highRes_netG

        image_high_tensor = data['original_high_res_render'].to(device=device)  # the renders. Shape of [Batch_size, Channels, Height, Width]
        image_high_tensor = torch.unsqueeze(image_high_tensor,0)
        
        if opt.use_front_normal:
            nmlF_high_tensor = data['nmlF_high_res'].to(device=device)
            nmlF_high_tensor = nmlF_high_tensor.unsqueeze(0)
        else:
            nmlF_high_tensor = None


        if opt.use_back_normal:
            nmlB_high_tensor = data['nmlB_high_res'].to(device=device)
            nmlB_high_tensor = nmlB_high_tensor.unsqueeze(0)
        else:
            nmlB_high_tensor = None




        if opt.use_depth_map and opt.allow_highres_to_use_depth:
            depth_map_high_res = data['depth_map'].to(device=device)
            depth_map_high_res = depth_map_high_res.unsqueeze(0)

        else: 
            depth_map_high_res = None




        if opt.use_mask_for_rendering_high_res:
            mask_high_res_tensor = data['mask'].to(device=device)
            mask_high_res_tensor = mask_high_res_tensor.unsqueeze(0)
        else:
            mask_high_res_tensor = None





        netG.filter( image_low_tensor, nmlF=nmlF_low_tensor, nmlB = nmlB_low_tensor, current_depth_map = depth_map_low_res, human_parse_map=human_parse_map  ) # forward-pass using only the low-resolution PiFU
        netG_output_map = netG.get_im_feat() # should have shape of [B, 256, H, W]

        net.filter( image_high_tensor, nmlF=nmlF_high_tensor, nmlB = nmlB_high_tensor,current_depth_map = depth_map_high_res, netG_output_map = netG_output_map , mask_low_res_tensor=None, mask_high_res_tensor=mask_high_res_tensor ) # forward-pass 
        image_tensor = image_high_tensor



    else:


        if opt.use_mask_for_rendering_low_res:
            mask_low_res_tensor = data['mask_low_pifu'].to(device=device)
            mask_low_res_tensor = mask_low_res_tensor.unsqueeze(0)
        else:
            mask_low_res_tensor = None


        net.filter( image_low_tensor, nmlF=nmlF_low_tensor, nmlB = nmlB_low_tensor, current_depth_map = depth_map_low_res, human_parse_map=human_parse_map  , mask_low_res_tensor=mask_low_res_tensor, mask_high_res_tensor=None ) # forward-pass 
        image_tensor = image_low_tensor



    try:
        save_img_path = save_path[:-4] + '.png'
        save_img_list = []
        for v in range(image_tensor.shape[0]):
            save_img = (np.transpose(image_tensor[v].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0
            save_img_list.append(save_img)
        save_img = np.concatenate(save_img_list, axis=1)
        cv2.imwrite(save_img_path, save_img)

        verts, faces, _, _ = reconstruction(
            net, device, calib_tensor, resolution, thresh, use_octree=use_octree, num_samples=50000, b_min=b_min , b_max=b_max )


        verts_tensor = torch.from_numpy(verts.T).unsqueeze(0).to(device=device).float()

        xyz_tensor = net.projection(verts_tensor, calib_tensor)  
        uv = xyz_tensor[:, :2, :]
        color = index(image_tensor, uv).detach().cpu().numpy()[0].T
        color = color * 0.5 + 0.5


        save_obj_mesh_with_color(save_path, verts, faces, color)


    except Exception as e:
        print(e)
        print("Cannot create marching cubes at this time.")






def train(opt):
    global gen_test_counter
    currently_epoch_to_update_low_res_pifu = True


    if torch.cuda.is_available():
        # set cuda
        device = 'cuda:0'

    else:
        device = 'cpu'

    print("using device {}".format(device) )

    if debug_mode:
        opt.debug_mode = True
    else:
        opt.debug_mode = False
    
    if test_script_activate:
        if test_script_activate_option_use_BUFF_dataset:
            from lib.data.BuffDataset import BuffDataset
            train_dataset = BuffDataset(opt)
        else:
            train_dataset = TrainDataset(opt, projection='orthogonal', phase = 'train', evaluation_mode = True)
    else:
        train_dataset = TrainDataset(opt, projection='orthogonal', phase = 'train')
    
    projection_mode = train_dataset.projection_mode

    if debug_mode:
        train_dataset.normal_directory_high_res = "/mnt/lustre/kennard.chan/specialized_pifuhd/trained_normal_maps"
        train_dataset.depth_map_directory = "/mnt/lustre/kennard.chan/specialized_pifuhd/trained_refined_depth_maps_usingNormalOnly"
        train_dataset.human_parse_map_directory = "/mnt/lustre/kennard.chan/specialized_pifuhd/trained_parse_maps"


    train_data_loader = DataLoader(train_dataset, 
                                   batch_size=opt.batch_size, shuffle=not opt.serial_batches,
                                   num_workers=opt.num_threads, pin_memory=opt.pin_memory)


    print('train loader size: ', len(train_data_loader))


    if opt.useValidationSet:
        validation_dataset = TrainDataset(opt, projection='orthogonal', phase = 'validation', evaluation_mode=False, validation_mode=True)

        validation_epoch_cd_dist_list = []
        validation_epoch_p2s_dist_list = []

        validation_graph_path = os.path.join(opt.results_path, opt.name, 'ValidationError_Graph.png')

    if debug_mode:
        validation_dataset.normal_directory_high_res = "/mnt/lustre/kennard.chan/specialized_pifuhd/trained_normal_maps"
        validation_dataset.depth_map_directory = "/mnt/lustre/kennard.chan/specialized_pifuhd/trained_refined_depth_maps_usingNormalOnly"
        validation_dataset.human_parse_map_directory = "/mnt/lustre/kennard.chan/specialized_pifuhd/trained_parse_maps"

    
    netG = HGPIFuNetwNML(opt, projection_mode, use_High_Res_Component = False)

    if opt.use_High_Res_Component:
        highRes_netG = HGPIFuNetwNML(opt, projection_mode, use_High_Res_Component = True)





    if (not os.path.exists(opt.checkpoints_path) ):
        os.makedirs(opt.checkpoints_path)
    if (not os.path.exists(opt.results_path) ):
        os.makedirs(opt.results_path)
    if (not os.path.exists('%s/%s' % (opt.checkpoints_path, opt.name))  ):
        os.makedirs('%s/%s' % (opt.checkpoints_path, opt.name))
    if (not os.path.exists('%s/%s' % (opt.results_path, opt.name)) ):
        os.makedirs('%s/%s' % (opt.results_path, opt.name))


    print('opt.name:', opt.name)


    if load_model_weights:

        # load weights for low-res model
        modelG_path = os.path.join( checkpoint_folder_to_load_low_res ,"netG_model_state_dict_epoch{0}.pickle".format(epoch_to_load_from_low_res) )

        print('Resuming from ', modelG_path)

        if device == 'cpu':
            with open(modelG_path, 'rb') as handle:
               netG_state_dict = CPU_Unpickler(handle).load()
        else:
            with open(modelG_path, 'rb') as handle:
               netG_state_dict = pickle.load(handle)

        netG.load_state_dict( netG_state_dict , strict = True )
        
        
        
        # load weights for high-res model
        if opt.use_High_Res_Component and load_model_weights_for_high_res_too:
            
            modelhighResG_path = os.path.join( checkpoint_folder_to_load_high_res, "highRes_netG_model_state_dict_epoch{0}.pickle".format(epoch_to_load_from_high_res) )

            print('Resuming from ', modelhighResG_path)

            if device == 'cpu':
                with open(modelhighResG_path, 'rb') as handle:
                   highResG_state_dict = CPU_Unpickler(handle).load()
            else:
                with open(modelhighResG_path, 'rb') as handle:
                   highResG_state_dict = pickle.load(handle)
            highRes_netG.load_state_dict( highResG_state_dict , strict = True )
            

        
    if test_script_activate:
        # testing script
        with torch.no_grad():

            print('generate mesh (test) ...')
            train_dataset.is_train = False
            netG = netG.to(device=device)
            netG.eval()

            if opt.use_High_Res_Component:
                highRes_netG = highRes_netG.to(device=device)
                highRes_netG.eval()

            if test_script_activate_option_use_BUFF_dataset:
                len_to_iterate = len(train_dataset)
            else:
                len_to_iterate = 72
            for gen_idx in tqdm(range(len_to_iterate)):

                if test_script_activate_option_use_BUFF_dataset:
                    index_to_use = gen_idx
                else:
                    index_to_use = gen_test_counter % len(train_dataset)
                gen_test_counter += 10 # 10 is the number of images for each class
                train_data = train_dataset.get_item(index=index_to_use) 
                save_path = '%s/%s/test_%s.obj' % (
                    opt.results_path, opt.name, train_data['name'])


                if opt.use_High_Res_Component:
                    gen_mesh(resolution=opt.resolution, net=[netG, highRes_netG] , device = device, data = train_data, save_path = save_path)
                else:
                    gen_mesh(resolution=opt.resolution, net=netG, device = device, data = train_data, save_path = save_path)

        print("Testing is Done! Exiting...")
        return
        
         
       
        
    
    
    



    opt_log = os.path.join(opt.results_path, opt.name, 'opt.txt')
    with open(opt_log, 'w') as outfile:
        outfile.write(json.dumps(vars(opt), indent=2))





    netG = netG.to(device=device)
    lr_G = opt.learning_rate_G
    optimizerG = torch.optim.RMSprop(netG.parameters(), lr=lr_G, momentum=0, weight_decay=0)


    
    # load saved weights for optimizerG
    if load_model_weights:
        # load saved weights for optimizerG
        optimizerG_path = os.path.join(checkpoint_folder_to_load_low_res, "optimizerG_epoch{0}.pickle".format(epoch_to_load_from_low_res) )
        with open(optimizerG_path, 'rb') as handle:
           optimizerG_state_dict = pickle.load(handle)
        optimizerG.load_state_dict( optimizerG_state_dict )

    

    if opt.use_High_Res_Component:
        highRes_netG = highRes_netG.to(device=device)
        lr_highRes = opt.learning_rate_MR
        optimizer_highRes = torch.optim.RMSprop(highRes_netG.parameters(), lr=lr_highRes, momentum=0, weight_decay=0)


        if load_model_weights and load_model_weights_for_high_res_too:
            # load highRes optimizer weights
            optimizer_highRes_path = os.path.join(checkpoint_folder_to_load_high_res, "optimizer_highRes_epoch{0}.pickle".format(epoch_to_load_from_high_res) )
            
            with open(optimizer_highRes_path, 'rb') as handle:
                optimizer_highRes_state_dict = pickle.load(handle)
            optimizer_highRes.load_state_dict( optimizer_highRes_state_dict )
            
        
        if opt.update_low_res_pifu:
            optimizer_lowResFineTune = torch.optim.RMSprop(netG.parameters(), lr=opt.learning_rate_low_res_finetune, momentum=0, weight_decay=0)

            if load_model_weights and (load_model_weights_for_low_res_finetuning_config != 0):
                # load optimizer_lowResFineTune weights

                if load_model_weights_for_low_res_finetuning_config == 1:
                    optimizer_lowResFineTune_path = os.path.join(checkpoint_folder_to_load_low_res, "optimizerG_epoch{0}.pickle".format(epoch_to_load_from_low_res) )
                elif load_model_weights_for_low_res_finetuning_config == 2:
                    optimizer_lowResFineTune_path = os.path.join(checkpoint_folder_to_load_high_res, "optimizer_lowResFineTune_epoch{0}.pickle".format(epoch_to_load_from_high_res)  )          
                else:
                    raise Exception('Incorrect use of load_model_weights_for_low_res_finetuning_config!')
                
                with open(optimizer_lowResFineTune_path, 'rb') as handle:
                   optimizer_lowResFineTune_state_dict = pickle.load(handle)
                optimizer_lowResFineTune.load_state_dict( optimizer_lowResFineTune_state_dict )
                


        

    start_epoch = 0
    for epoch in range(start_epoch, opt.num_epoch):



        print("start of epoch {}".format(epoch) )


        netG.train()
        if opt.use_High_Res_Component:
            if opt.update_low_res_pifu:
                if (epoch < opt.epoch_to_start_update_low_res_pifu):
                    currently_epoch_to_update_low_res_pifu = False 
                    print("currently_epoch_to_update_low_res_pifu remains at False for this epoch")
                elif (epoch >= opt.epoch_to_end_update_low_res_pifu):
                    currently_epoch_to_update_low_res_pifu = False
                    print("No longer updating low_res_pifu! In the Finetune Phase") 
                elif (epoch % opt.epoch_interval_to_update_low_res_pifu == 0):
                    currently_epoch_to_update_low_res_pifu = not currently_epoch_to_update_low_res_pifu
                    print("Updating currently_epoch_to_update_low_res_pifu to: ",currently_epoch_to_update_low_res_pifu)
                else:
                    pass



            if opt.update_low_res_pifu and currently_epoch_to_update_low_res_pifu:
                netG.train()
                highRes_netG.eval()
            else:
                netG.eval()
                highRes_netG.train()





        train_len = len(train_data_loader)
        for train_idx, train_data in enumerate(train_data_loader):
            print("batch {}".format(train_idx) )

            # retrieve the data
            calib_tensor = train_data['calib'].to(device=device) # the calibration matrices for the renders ( is np.matmul(intrinsic, extrinsic)  ). Shape of [Batchsize, 4, 4]


            if opt.use_High_Res_Component:
                render_low_pifu_tensor = train_data['render_low_pifu'].to(device=device) 
                render_pifu_tensor = train_data['original_high_res_render'].to(device=device)  # the renders. Shape of [Batch_size, Channels, Height, Width]
                
                if opt.use_front_normal:
                    nmlF_low_tensor = train_data['nmlF'].to(device=device)
                    nmlF_tensor = train_data['nmlF_high_res'].to(device=device)
                else:
                    nmlF_tensor = None

                if opt.use_back_normal:
                    nmlB_low_tensor = train_data['nmlB'].to(device=device)
                    nmlB_tensor = train_data['nmlB_high_res'].to(device=device)
                else:
                    nmlB_low_tensor = None
                    nmlB_tensor = None
            else:
                # low-resolution image that is required by both models
                render_pifu_tensor = train_data['render_low_pifu'].to(device=device)  # the renders. Shape of [Batch_size, Channels, Height, Width]
                
                if opt.use_front_normal:
                    nmlF_tensor = train_data['nmlF'].to(device=device)
                else:
                    nmlF_tensor = None

                if opt.use_back_normal:
                    nmlB_tensor = train_data['nmlB'].to(device=device)
                else:
                    nmlB_tensor = None



            if opt.use_depth_map:
                current_depth_map = train_data['depth_map'].to(device=device)
                if opt.depth_in_front and (not opt.use_High_Res_Component) :
                    current_depth_map = train_data['depth_map_low_res'].to(device=device)
                if opt.use_High_Res_Component:
                    current_low_depth_map = train_data['depth_map_low_res'].to(device=device)
                    if not opt.allow_highres_to_use_depth:
                        current_depth_map = None
            else: 
                current_depth_map = None
                current_low_depth_map = None


            if opt.use_human_parse_maps:
                human_parse_map = train_data['human_parse_map'].to(device=device)
            else:
                human_parse_map = None


 
            samples_low_res_pifu_tensor = train_data['samples_low_res_pifu'].to(device=device)  # contain inside and outside points. Shape of [Batch_size, 3, num_of_points]
            labels_low_res_pifu_tensor = train_data['labels_low_res_pifu'].to(device=device)  # tell us which points in sample_tensor are inside and outside in the surface. Should have shape of [Batch_size ,1, num_of_points]


            if opt.use_High_Res_Component:
                netG.filter( render_low_pifu_tensor, nmlF=nmlF_low_tensor, nmlB = nmlB_low_tensor, current_depth_map = current_low_depth_map, human_parse_map=human_parse_map ) # forward-pass using only the low-resolution PiFU
                netG_output_map = netG.get_im_feat() # should have shape of [B, 256, H, W]


                error_high_pifu, res_high_pifu = highRes_netG.forward(images=render_pifu_tensor, points=samples_low_res_pifu_tensor, calibs=calib_tensor, labels=labels_low_res_pifu_tensor,  points_nml=None, labels_nml=None, nmlF = nmlF_tensor, nmlB = nmlB_tensor, current_depth_map = current_depth_map, netG_output_map=netG_output_map)
                if opt.update_low_res_pifu and currently_epoch_to_update_low_res_pifu:
                    optimizer_lowResFineTune.zero_grad()
                    error_high_pifu['Err(occ)'].backward()
                    curr_high_loss = error_high_pifu['Err(occ)'].item()
                    optimizer_lowResFineTune.step() 
                else:
                    optimizer_highRes.zero_grad()
                    error_high_pifu['Err(occ)'].backward()
                    curr_high_loss = error_high_pifu['Err(occ)'].item()
                    optimizer_highRes.step()

                print(
                'Name: {0} | Epoch: {1} | error_high_pifu: {2:.06f} | LR: {3:.06f} '.format(
                    opt.name, epoch, curr_high_loss, lr_highRes)
                )

                r = res_high_pifu

            else:
                error_low_res_pifu, res_low_res_pifu = netG.forward(images=render_pifu_tensor, points=samples_low_res_pifu_tensor, calibs=calib_tensor, labels=labels_low_res_pifu_tensor,  points_nml=None, labels_nml=None, nmlF = nmlF_tensor, nmlB = nmlB_tensor, current_depth_map = current_depth_map, human_parse_map=human_parse_map)
                optimizerG.zero_grad()
                error_low_res_pifu['Err(occ)'].backward()
                curr_low_res_loss = error_low_res_pifu['Err(occ)'].item()
                optimizerG.step()

                print(
                'Name: {0} | Epoch: {1} | error_low_res_pifu: {2:.06f} | LR: {3:.06f} '.format(
                    opt.name, epoch, curr_low_res_loss, lr_G)
                )

                r = res_low_res_pifu





                


        lr_G = adjust_learning_rate(optimizerG, epoch, lr_G, opt.schedule, opt.learning_rate_decay)
        if opt.use_High_Res_Component:
            if opt.update_low_res_pifu and currently_epoch_to_update_low_res_pifu:
                lr_highRes = adjust_learning_rate(optimizer_lowResFineTune, epoch, lr_highRes, opt.schedule, opt.learning_rate_decay)
            else:
                lr_highRes = adjust_learning_rate(optimizer_highRes, epoch, lr_highRes, opt.schedule, opt.learning_rate_decay)
 


        with torch.no_grad():

            if True:

                # save as pickle:
                with open( '%s/%s/netG_model_state_dict_epoch%s.pickle' % (opt.checkpoints_path, opt.name, str(epoch) ) , 'wb') as handle:
                    pickle.dump(netG.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)

                with open( '%s/%s/optimizerG_epoch%s.pickle' % (opt.checkpoints_path, opt.name, str(epoch)) , 'wb') as handle:
                    pickle.dump(optimizerG.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)

                if opt.use_High_Res_Component:

                    with open( '%s/%s/highRes_netG_model_state_dict_epoch%s.pickle' % (opt.checkpoints_path, opt.name, str(epoch) ) , 'wb') as handle:
                        pickle.dump(highRes_netG.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)

                    with open( '%s/%s/optimizer_highRes_epoch%s.pickle' % (opt.checkpoints_path, opt.name, str(epoch) ) , 'wb') as handle:
                        pickle.dump(optimizer_highRes.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)
                    
                    if opt.update_low_res_pifu:
                        with open( '%s/%s/optimizer_lowResFineTune_epoch%s.pickle' % (opt.checkpoints_path, opt.name,str(epoch) ) , 'wb') as handle:
                            pickle.dump(optimizer_lowResFineTune.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)


                    highRes_netG.eval()



                print('generate mesh (train) ...')
                train_dataset.is_train = False
                netG.eval()
                for gen_idx in tqdm(range(1)):

                    index_to_use = gen_test_counter % len(train_dataset)
                    gen_test_counter += 10 # 10 is the number of images for each class
                    train_data = train_dataset.get_item(index=index_to_use) 
                    # train_data["img"].shape  has shape of [1, 3, 512, 512]
                    save_path = '%s/%s/train_eval_epoch%d_%s.obj' % (
                        opt.results_path, opt.name, epoch, train_data['name'])


                    if opt.use_High_Res_Component:
                        gen_mesh(resolution=opt.resolution, net=[netG, highRes_netG] , device = device, data = train_data, save_path = save_path)
                    else:
                        gen_mesh(resolution=opt.resolution, net=netG , device = device, data = train_data, save_path = save_path)


                try:
                    # save visualization of model performance
                    save_path = '%s/%s/pred.ply' % (opt.results_path, opt.name)
                    r = r[0].cpu() # get only the first example in the batch (i.e. 1 CAD model or subject). [1, Num of sampled points]
                    points = samples_low_res_pifu_tensor[0].transpose(0, 1).cpu()    # note that similar to res[0], we only take sample_tensor[0] i.e. the first CAD model. Shape of [Num of sampled points, 3] after the transpose. 
                    save_samples_truncted_prob(save_path, points.detach().numpy(), r.detach().numpy())
                except:
                    print("Unable to save point cloud.")
                    

                train_dataset.is_train = True


            if opt.useValidationSet:
                import trimesh
                from evaluate_model import quick_get_chamfer_and_surface_dist
                num_samples_to_use = 5000

                print('Commencing validation..')
                print('generate mesh (validation) ...')

                netG.eval()
                if opt.use_High_Res_Component:
                    highRes_netG.eval()
                val_len = len(validation_dataset)
                num_of_val_subjects = val_len//10 # as each subject has 10 images
                val_mesh_paths = []
                index_to_use_list = []
                num_of_validation_subjects_to_use = 10
                #if debug_mode:
                #    num_of_validation_subjects_to_use = 2
                for gen_idx in tqdm(range(num_of_validation_subjects_to_use)):
                    print('[Validation] generating mesh #{0}'.format(gen_idx) )
                    index_to_use = np.random.randint(low=0, high=num_of_val_subjects)
                    while index_to_use in index_to_use_list:
                        print('repeated index_to_use is selected, re-sampling')
                        index_to_use = np.random.randint(low=0, high=num_of_val_subjects)
                    index_to_use_list.append(index_to_use)
                    val_data = validation_dataset.get_item(index=index_to_use*10)  # as each subject has 10 images

                    save_path = '%s/%s/val_eval_epoch%d_%s.obj' % (
                        opt.results_path, opt.name, epoch, val_data['name'])

                    val_mesh_paths.append(save_path)

                    if opt.use_High_Res_Component:
                        gen_mesh(resolution=opt.resolution, net=[netG, highRes_netG] , device = device, data = val_data, save_path = save_path)
                    else:
                        gen_mesh(resolution=opt.resolution, net=netG , device = device, data = val_data, save_path = save_path)

                total_chamfer_distance = []
                total_point_to_surface_distance = []
                for val_path in val_mesh_paths:
                    subject = val_path.split('_')[-1]
                    subject = subject.replace('.obj','')
                    GT_mesh = validation_dataset.mesh_dic[subject]
                    

                    try: 
                        print('Computing CD and P2S for {0}'.format( os.path.basename(val_path) ) )
                        source_mesh = trimesh.load(val_path)
                        chamfer_distance, point_to_surface_distance = quick_get_chamfer_and_surface_dist(src_mesh=source_mesh, tgt_mesh=GT_mesh, num_samples=num_samples_to_use )
                        total_chamfer_distance.append(chamfer_distance)
                        total_point_to_surface_distance.append(point_to_surface_distance)
                    except:
                        print('Unable to compute chamfer_distance and/or point_to_surface_distance!')
                
                if len(total_chamfer_distance) == 0:
                    average_chamfer_distance = 0
                else:
                    average_chamfer_distance = np.mean(total_chamfer_distance) 

                if len(total_point_to_surface_distance) == 0:
                    average_point_to_surface_distance = 0 
                else:
                    average_point_to_surface_distance = np.mean(total_point_to_surface_distance) 

                validation_epoch_cd_dist_list.append(average_chamfer_distance)
                validation_epoch_p2s_dist_list.append(average_point_to_surface_distance)

                
                print("[Validation] Overall Epoch {0}- Avg CD: {1}; Avg P2S: {2}".format(epoch, average_chamfer_distance, average_point_to_surface_distance ) )


                # Delete files that are created for validation
                for file_path in val_mesh_paths:
                    mesh_path = file_path
                    image_path = file_path.replace('.obj', '.png')
                    os.remove(mesh_path)
                    os.remove(image_path)






        plt.plot( np.arange(epoch+1) , np.array(validation_epoch_cd_dist_list) )
        plt.plot( np.arange(epoch+1) , np.array(validation_epoch_p2s_dist_list), '-.' )
        plt.xlabel('Epoch')
        plt.ylabel('Validation Error (CD + P2D)')
        plt.title('Epoch Against Validation Error (CD + P2D)')
        plt.savefig(validation_graph_path)






if __name__ == '__main__':
    train(opt)

