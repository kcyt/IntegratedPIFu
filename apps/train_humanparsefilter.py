
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


seed = 10 
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


parser = BaseOptions()
opt = parser.parse()
gen_test_counter = 0


lr = 1e-3 
parse_schedule = [50] # epoch at which to reduce the lr
num_of_epoch = 70
batch_size = 4
num_classes = 7 # Number of body part categories
load_model = False # If True, need set the 'modelparsefilter_path' variable below

def adjust_learning_rate(optimizer, epoch, lr, schedule, learning_rate_decay):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in schedule:
        lr *= learning_rate_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr


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



def train(opt):
    global gen_test_counter
    global lr 

    if torch.cuda.is_available():
        # set cuda
        device = 'cuda:0'
    else:
        device = 'cpu'

    print("using device {}".format(device) )
    

    train_dataset = HumanParseDataset(opt, evaluation_mode=False)
    


    train_data_loader = DataLoader(train_dataset, 
                                   batch_size=batch_size, shuffle=not opt.serial_batches,
                                   num_workers=opt.num_threads, pin_memory=opt.pin_memory)


    print('train loader size: ', len(train_data_loader))


    
    
    humanParsefilter = HumanParseFilter(opt)



    if (not os.path.exists(opt.checkpoints_path) ):
        os.makedirs(opt.checkpoints_path)
    if (not os.path.exists(opt.results_path) ):
        os.makedirs(opt.results_path)
    if (not os.path.exists('%s/%s' % (opt.checkpoints_path, opt.name))  ):
        os.makedirs('%s/%s' % (opt.checkpoints_path, opt.name))
    if (not os.path.exists('%s/%s' % (opt.results_path, opt.name)) ):
        os.makedirs('%s/%s' % (opt.results_path, opt.name))


    palette = get_palette(num_classes)
     



    if load_model: 
        modelparsefilter_path = "apps/checkpoints/Date_06_Jan_22_Time_00_49_24/humanParsefilter_model_state_dict.pickle" # Date_06_Jan_22_Time_00_49_24 is folder to load

        print('Resuming from ', modelparsefilter_path)

        with open(modelparsefilter_path, 'rb') as handle:
           net_state_dict = pickle.load(handle)

        humanParsefilter.load_state_dict( net_state_dict , strict = True )
         
     

    opt_log = os.path.join(opt.results_path, opt.name, 'opt.txt')
    with open(opt_log, 'w') as outfile:
        outfile.write(json.dumps(vars(opt), indent=2))



    humanParsefilter = humanParsefilter.to(device=device)

 
    optimizer = torch.optim.RMSprop(humanParsefilter.parameters(), lr=lr, momentum=0, weight_decay=0)



    start_epoch = 0
    for epoch in range(start_epoch, num_of_epoch):

        print("start of epoch {}".format(epoch) )

        humanParsefilter.train()

        train_len = len(train_data_loader)
        for train_idx, train_data in enumerate(train_data_loader):
            print("batch {}".format(train_idx) )

            # retrieve the data
            render_tensor = train_data['original_high_res_render'].to(device=device)  # the renders. Shape of [Batch_size, Channels, Height, Width]
            human_parse_map_high_res_tensor = train_data['human_parse_map_high_res'].to(device=device)
            
            if opt.use_normal_map_for_parse_training:
                nmlF_high_res_tensor = train_data['nmlF_high_res'].to(device=device)
                render_tensor = torch.cat( [render_tensor, nmlF_high_res_tensor ], dim=1 )
            

            error = humanParsefilter.forward(images=render_tensor, groundtruth_parsemap=human_parse_map_high_res_tensor)
        
            optimizer.zero_grad()

            error['Err'].backward()
            curr_loss = error['Err'].item()

            optimizer.step()

            print(
            'Name: {0} | Epoch: {1} | error: {2:.06f} | LR: {3:.06f} '.format(
                opt.name, epoch, curr_loss, lr)
            )


                


        lr = adjust_learning_rate(optimizer, epoch, lr, schedule=parse_schedule, learning_rate_decay=0.05)



        with torch.no_grad():
            if True:

                # save models
                with open( '%s/%s/humanParsefilter_model_state_dict.pickle' % (opt.checkpoints_path, opt.name) , 'wb') as handle:
                    pickle.dump(humanParsefilter.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)

                with open( '%s/%s/optimizer.pickle' % (opt.checkpoints_path, opt.name) , 'wb') as handle:
                    pickle.dump(optimizer.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)

                print('generate parse map (train) ...')
                train_dataset.is_train = False
                humanParsefilter.eval()
                for gen_idx in tqdm(range(opt.num_gen_mesh_test)):

                    index_to_use = gen_test_counter % len(train_dataset)
                    gen_test_counter += 10 # 10 is the number of images for each class
                    train_data = train_dataset.get_item(index=index_to_use) 
                    save_path = '%s/%s/train_eval_epoch%d_%s.obj' % (
                        opt.results_path, opt.name, epoch, train_data['name'])


                    image_tensor = train_data['original_high_res_render'].to(device=device)  
                    image_tensor = torch.unsqueeze(image_tensor,0)
        

                    if opt.use_normal_map_for_parse_training:
                        nmlF_high_res_tensor = train_data['nmlF_high_res'].to(device=device)
                        nmlF_high_res_tensor = torch.unsqueeze(nmlF_high_res_tensor,0)
                        image_tensor = torch.cat( [image_tensor, nmlF_high_res_tensor ], dim=1 )



                    original_parse_map = train_data['human_parse_map_high_res'] # Shape of [C,H,W]
                    original_parse_map = torch.argmax(original_parse_map, dim=0).cpu().numpy() # Shape of [H,W] 


                    humanParsefilter.filter( image_tensor ) # forward-pass  
                    generated_parse_map = humanParsefilter.generate_parse_map() # [B,H,W]   
                    generated_parse_map = generated_parse_map.detach().cpu().numpy()[0,:,:]



                    save_img_path = save_path[:-4] + '.png'
                    save_parsemap_path = save_path[:-4] + 'generated_parsemap.png'

                    save_img = (np.transpose(image_tensor[0,:3,...].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5) * 255.0
                    save_img = save_img.astype(np.uint8)
                    save_img = Image.fromarray(save_img)
                    save_img.save(save_img_path)


                    save_parsemap = Image.fromarray( np.asarray(generated_parse_map, dtype=np.uint8) )
                    save_parsemap.putpalette(palette)
                    save_parsemap.save(save_parsemap_path)


                    original_parse_map = original_parse_map.astype(np.uint8)
                    original_parse_map = Image.fromarray(original_parse_map)
                    original_parse_map.putpalette(palette)
                    original_parse_map_save_path = '%s/%s/train_eval_epoch%d_%s_groundtruth.png' % (
                        opt.results_path, opt.name, epoch, train_data['name'])
                    original_parse_map.save(original_parse_map_save_path)


                train_dataset.is_train = True






if __name__ == '__main__':
    train(opt)

