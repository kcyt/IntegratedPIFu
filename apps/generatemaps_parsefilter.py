
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
from lib.model import HumanParseFilter
from lib.data import HumanParseDataset



parser = BaseOptions()
opt = parser.parse()
gen_test_counter = 0

generate_for_buff_dataset = False 


num_classes = 7 # number of body parts categories
batch_size = 1 


if generate_for_buff_dataset:
    generated_results_path = "/mnt/lustre/kennard.chan/IntegratedPIFu/trained_buff_dataset/buff_parse_maps"
else:
    generated_results_path = "/mnt/lustre/kennard.chan/IntegratedPIFu/trained_parse_maps"





def generate_maps(opt):
    global gen_test_counter
    global lr 

    
    palette = get_palette(num_classes)

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
        train_dataset = HumanParseDataset(opt, evaluation_mode=False)
        train_dataset.is_train = False

        test_dataset = HumanParseDataset(opt, evaluation_mode=True)
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

    
    
    
    humanParsefilter = HumanParseFilter(opt)




    # load model
        
    modelparsefilter_path = "/mnt/lustre/kennard.chan/IntegratedPIFu/apps/checkpoints/Date_06_Jan_22_Time_00_49_24/humanParsefilter_model_state_dict.pickle"

    print('Resuming from ', modelparsefilter_path)


    if device == 'cpu' :
        import io

        class CPU_Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == 'torch.storage' and name == '_load_from_bytes':
                    return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
                else:
                    return super().find_class(module, name)

        with open(modelparsefilter_path, 'rb') as handle:
           net_state_dict = CPU_Unpickler(handle).load()


    else:
        with open(modelparsefilter_path, 'rb') as handle:
           net_state_dict = pickle.load(handle)

    humanParsefilter.load_state_dict( net_state_dict , strict = True )        



    humanParsefilter = humanParsefilter.to(device=device)     
    humanParsefilter.eval()

    with torch.no_grad():
        for description, data_loader in data_loader_list:
            print( 'description: {0}'.format(description) )
            for idx, batch_data in enumerate(data_loader):
                print("batch {}".format(idx) )

                # retrieve the data
                subject_list = batch_data['name']
                render_filename_list = batch_data['render_path']  

                render_tensor = batch_data['original_high_res_render'].to(device=device)  # the renders. Shape of [Batch_size, Channels, Height, Width]

                if opt.use_normal_map_for_parse_training:
                    nmlF_high_res_tensor = batch_data['nmlF_high_res'].to(device=device)
                    render_tensor = torch.cat( [render_tensor, nmlF_high_res_tensor ], dim=1 )


                humanParsefilter.filter( render_tensor ) # forward-pass  
                generated_parse_map = humanParsefilter.generate_parse_map() # [B,H,W] where 
                generated_parse_map = generated_parse_map.detach().cpu().numpy()  # [B,H,W]



                for i in range(batch_size):

                    try:
                        subject = subject_list[i]
                        render_filename = render_filename_list[i]
                    except:
                        print('Last batch of data_loader reached!')
                        break


                    if generate_for_buff_dataset:
                        save_parsemap_path = os.path.join(generated_results_path, "rendered_parse_" +  subject + ".npy"  )
                        save_parsemap_image_path = os.path.join(generated_results_path, "rendered_parse_" +  subject + ".png"  )

                    else:
                        if not os.path.exists( os.path.join(generated_results_path,subject) ):
                            os.makedirs(os.path.join(generated_results_path,subject) )

                        yaw = render_filename.split('_')[-1].split('.')[0]
                        save_parsemap_path = os.path.join(generated_results_path,  subject, "rendered_parse_" + yaw + ".npy"  )
                        save_parsemap_image_path = os.path.join(generated_results_path, subject, "rendered_parse_" + yaw + ".png"  )

                    
                    
                    generated_map = generated_parse_map[i] # [H,W]
                    
                    np.save(save_parsemap_path, generated_map) # [H,W]


                    # save as images
                    save_parsemap = Image.fromarray( np.asarray(generated_map, dtype=np.uint8) )
                    save_parsemap.putpalette(palette)
                    save_parsemap.save(save_parsemap_image_path)






def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette






   


if __name__ == '__main__':

    generate_maps(opt)

