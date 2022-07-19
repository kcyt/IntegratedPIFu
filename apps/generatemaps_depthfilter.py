
import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import numpy as np
import cv2
import pickle

from PIL import Image

from lib.options import BaseOptions
from lib.model import RelativeDepthFilter
from lib.data import DepthDataset



parser = BaseOptions()
opt = parser.parse()
gen_test_counter = 0



generate_for_buff_dataset = False 


generate_refined_trained_depth_maps = True # if False, generate Coarse depth maps. If True, generate refined depth maps (Second-stage maps)

batch_size = 4




if generate_refined_trained_depth_maps:
    trained_depth_maps_path = "trained_depth_maps"
else:
    trained_depth_maps_path = "trained_coarse_depth_maps"


if generate_for_buff_dataset:
    print("Overwriting trained_depth_maps_path for Buff dataset")
    trained_depth_maps_path = "buff_dataset/buff_depth_maps"




def generate_maps(opt):
    global gen_test_counter
    global lr 

    if torch.cuda.is_available():
        # set cuda
        device = 'cuda:0'

    else:
        device = 'cpu'

    print("using device {}".format(device) )
    

    if generate_for_buff_dataset:
        from lib.data.BuffDataset import BuffDataset
        train_dataset = BuffDataset(opt)
        train_dataset.is_train = False
        train_data_loader = DataLoader(train_dataset, 
                                   batch_size=batch_size, shuffle=False,
                                   num_workers=opt.num_threads, pin_memory=opt.pin_memory)

        print('train loader size: ', len(train_data_loader))

        data_loader_list = [ ( 'first_data_loader' , train_data_loader) ]
    else:
        train_dataset = DepthDataset(opt, evaluation_mode=False)
        train_dataset.is_train = False

        test_dataset = DepthDataset(opt, evaluation_mode=True)
        test_dataset.is_train = False

        train_data_loader = DataLoader(train_dataset, 
                                       batch_size=batch_size, shuffle=False,
                                       num_workers=opt.num_threads, pin_memory=opt.pin_memory)


        test_data_loader = DataLoader(test_dataset, 
                                   batch_size=batch_size, shuffle=False,
                                   num_workers=opt.num_threads, pin_memory=opt.pin_memory)



        print('train loader size: ', len(train_data_loader))
        print('test loader size: ', len(test_data_loader))

        data_loader_list = [ ( 'first_data_loader' , train_data_loader), ( 'second_data_loader' , test_data_loader) ]


    
    
    depthfilter = RelativeDepthFilter(opt)



    # load model
    if generate_refined_trained_depth_maps:
        modeldepthfilter_path = "apps/checkpoints/Date_08_Jan_22_Time_02_03_43/depthfilter_model_state_dict.pickle" # Date_08_Jan_22_Time_02_03_43 is the folder to use.
    else:
        modeldepthfilter_path = "apps/checkpoints/Date_06_Jan_22_Time_02_37_32/depthfilter_model_state_dict.pickle" # Date_06_Jan_22_Time_02_37_32 is the folder to use.

    print('Resuming from ', modeldepthfilter_path)

    if device == 'cpu' :
        import io

        class CPU_Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == 'torch.storage' and name == '_load_from_bytes':
                    return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
                else:
                    return super().find_class(module, name)

        with open(modeldepthfilter_path, 'rb') as handle:
           net_state_dict = CPU_Unpickler(handle).load()


    else:
        with open(modeldepthfilter_path, 'rb') as handle:
           net_state_dict = pickle.load(handle)

    depthfilter.load_state_dict( net_state_dict , strict = True )
    
        
    depthfilter = depthfilter.to(device=device)
    depthfilter.eval()


    with torch.no_grad():
        for description, data_loader in data_loader_list:
            print( 'description: {0}'.format(description) )
            for idx, batch_data in enumerate(data_loader):
                print("batch {}".format(idx) )

                # retrieve the data
                subject_list = batch_data['name']
                render_filename_list = batch_data['render_path']  

                render_tensor = batch_data['original_high_res_render'].to(device=device)  # the renders. Shape of [Batch_size, Channels, Height, Width]

                if generate_refined_trained_depth_maps:
                    coarse_depth_map_tensor = batch_data['coarse_depth_map'].to(device=device)   # shape of [batch, 1,1024,1024]
                    render_tensor = torch.cat( [render_tensor, coarse_depth_map_tensor ], dim=1 )

                    if opt.use_normal_map_for_depth_training:
                        nmlF_high_res_tensor = batch_data['nmlF_high_res'].to(device=device)
                        render_tensor = torch.cat( [render_tensor, nmlF_high_res_tensor ], dim=1 )


                else:
                    center_indicator_tensor = batch_data['center_indicator'].to(device=device)   

                    render_tensor = torch.cat( [render_tensor, center_indicator_tensor ], dim=1 )

                    if opt.use_normal_map_for_depth_training:
                        nmlF_high_res_tensor = batch_data['nmlF_high_res'].to(device=device)
                        render_tensor = torch.cat( [render_tensor, nmlF_high_res_tensor ], dim=1 )





                depthfilter.filter( render_tensor ) # forward-pass  
                generated_depth_maps = depthfilter.generate_depth_map() # [B, C, H, W] where C == 1
                generated_depth_maps = generated_depth_maps.detach().cpu().numpy()

                
                for i in range(batch_size):


                    try:
                        subject = subject_list[i]
                        render_filename = render_filename_list[i]
                    except:
                        print('Last batch of data_loader reached!')
                        break

                    if generate_for_buff_dataset:
                        if generate_refined_trained_depth_maps:
                            save_depthmap_path = os.path.join(trained_depth_maps_path, "rendered_depthmap_" +  subject + ".npy"  )
                            save_depthmap_image_path = os.path.join(trained_depth_maps_path, "rendered_depthmap_" +  subject + ".png"  )
                        else:
                            save_depthmap_path = os.path.join(trained_depth_maps_path, "rendered_coarse_depthmap_" +  subject + ".npy"  )
                            save_depthmap_image_path = os.path.join(trained_depth_maps_path, "rendered_coarse_depthmap_" +  subject + ".png"  )

                    else:
                        if not os.path.exists( os.path.join(trained_depth_maps_path,subject) ):
                            os.makedirs(os.path.join(trained_depth_maps_path,subject) )

                        yaw = render_filename.split('_')[-1].split('.')[0]
                        save_depthmap_path = os.path.join(trained_depth_maps_path, subject, "rendered_depthmap_" + yaw + ".npy"  )
                        save_depthmap_image_path = os.path.join(trained_depth_maps_path, subject, "rendered_depthmap_" + yaw + ".png"  )





                    
                    generated_map = generated_depth_maps[i]
                    np.save(save_depthmap_path, generated_map) # generated_map has shape of [C,H,W]


                    # save as images
                    save_depthmap_image = (np.transpose( generated_map, (1, 2, 0)) ) * 255.0 /2
                    save_depthmap_image = save_depthmap_image.astype(np.uint8)[:,:,0] 
                    save_depthmap_image = Image.fromarray(save_depthmap_image, 'L')
                    save_depthmap_image.save(save_depthmap_image_path)







if __name__ == '__main__':

    generate_maps(opt)

