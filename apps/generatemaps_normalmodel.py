
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

from PIL import Image

from lib.options import BaseOptions
from lib.networks import define_G
from lib.data.NormalDataset import NormalDataset

import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn



parser = BaseOptions()
opt = parser.parse()
gen_test_counter = 0


generate_for_buff_dataset = False 


if generate_for_buff_dataset:
    trained_normal_maps_path = "buff_dataset/buff_normal_maps"
else:
    trained_normal_maps_path = "trained_normal_maps"


batch_size = 2



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
        train_dataset = NormalDataset(opt, evaluation_mode=False)
        train_dataset.is_train = False
        test_dataset = NormalDataset(opt, evaluation_mode=True)
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
    

    

    


    


    
    
    netF = define_G(3, 3, 64, "global", 4, 9, 1, 3, "instance")

    netB = define_G(3, 3, 64, "global", 4, 9, 1, 3, "instance")




    # load model

    F_modelnormal_path = "apps/checkpoints/Date_12_Nov_21_Time_19_17_20/netF_model_state_dict.pickle" # Date_12_Nov_21_Time_19_17_20 is the folder to use
    B_modelnormal_path = "apps/checkpoints/Date_12_Nov_21_Time_19_17_20/netB_model_state_dict.pickle"

    print('Resuming from ', F_modelnormal_path)
    print('Resuming from ', B_modelnormal_path)

    if device == 'cpu' :
        import io

        class CPU_Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == 'torch.storage' and name == '_load_from_bytes':
                    return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
                else:
                    return super().find_class(module, name)

        with open(F_modelnormal_path, 'rb') as handle:
           netF_state_dict = CPU_Unpickler(handle).load()

        with open(B_modelnormal_path, 'rb') as handle:
           netB_state_dict = CPU_Unpickler(handle).load()

    else:

        with open(F_modelnormal_path, 'rb') as handle:
           netF_state_dict = pickle.load(handle)

        with open(B_modelnormal_path, 'rb') as handle:
           netB_state_dict = pickle.load(handle)

    netF.load_state_dict( netF_state_dict , strict = True )
    netB.load_state_dict( netB_state_dict , strict = True )
        


    netF = netF.to(device=device)
    netB = netB.to(device=device)

    netF.eval()
    netB.eval()




    with torch.no_grad():
        for description, data_loader in data_loader_list:
            print( 'description: {0}'.format(description) )
            for idx, batch_data in enumerate(data_loader):
                print("batch {}".format(idx) )

                # retrieve the data
                subject_list = batch_data['name']
                render_filename_list = batch_data['render_path']  

                render_tensor = batch_data['original_high_res_render'].to(device=device)  # the renders. Shape of [Batch_size, Channels, Height, Width]

                res_netF = netF.forward(render_tensor)
                res_netB = netB.forward(render_tensor)

                res_netF = res_netF.detach().cpu().numpy()
                res_netB = res_netB.detach().cpu().numpy()



                for i in range(batch_size):

                    try:
                        subject = subject_list[i]
                        render_filename = render_filename_list[i]
                    except:
                        print('Last batch of data_loader reached!')
                        break


                    if generate_for_buff_dataset:
                        save_normalmapF_path = os.path.join(trained_normal_maps_path, "rendered_nmlF_" +  subject + ".npy"  )
                        save_normalmapB_path = os.path.join(trained_normal_maps_path, "rendered_nmlB_" +  subject + ".npy"  )

                        save_netF_normalmap_path = os.path.join(trained_normal_maps_path, "rendered_nmlF_" +  subject + ".png"  )
                        save_netB_normalmap_path = os.path.join(trained_normal_maps_path, "rendered_nmlB_" +  subject + ".png"  )

                    else:
                        if not os.path.exists( os.path.join(trained_normal_maps_path,subject) ):
                            os.makedirs(os.path.join(trained_normal_maps_path,subject) )

                        yaw = render_filename.split('_')[-1].split('.')[0]
                        save_normalmapF_path = os.path.join(trained_normal_maps_path, subject, "rendered_nmlF_" + yaw + ".npy"  )
                        save_normalmapB_path = os.path.join(trained_normal_maps_path, subject, "rendered_nmlB_" + yaw + ".npy"  )

                        save_netF_normalmap_path = os.path.join(trained_normal_maps_path, subject, "rendered_nmlF_" + yaw + ".png"  )
                        save_netB_normalmap_path = os.path.join(trained_normal_maps_path, subject, "rendered_nmlB_" + yaw + ".png"  )



                    generated_map = res_netF[i]
                    np.save(save_normalmapF_path, generated_map) # generated_map has shape of [C,H,W]

                    generated_map = res_netB[i]
                    np.save(save_normalmapB_path, generated_map) # generated_map has shape of [C,H,W]


                    # save as images
                    save_netF_normalmap = (np.transpose(res_netF[i], (1, 2, 0)) * 0.5 + 0.5) * 255.0
                    save_netF_normalmap = save_netF_normalmap.astype(np.uint8)
                    save_netF_normalmap = Image.fromarray(save_netF_normalmap)
                    save_netF_normalmap.save(save_netF_normalmap_path)

                    save_netB_normalmap = (np.transpose(res_netB[i], (1, 2, 0)) * 0.5 + 0.5) * 255.0
                    save_netB_normalmap = save_netB_normalmap.astype(np.uint8)
                    save_netB_normalmap = Image.fromarray(save_netB_normalmap)
                    save_netB_normalmap.save(save_netB_normalmap_path)






            



if __name__ == '__main__':
    generate_maps(opt)

