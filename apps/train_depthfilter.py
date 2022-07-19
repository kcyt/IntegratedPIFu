
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
import matplotlib.pyplot as plt

from PIL import Image

from lib.options import BaseOptions
from lib.model import RelativeDepthFilter
from lib.data import DepthDataset


seed = 10 
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


parser = BaseOptions()
opt = parser.parse()
gen_test_counter = 0



lr = 1e-3 
depth_schedule = [50] # epoch at which to reduce the lr
num_of_epoch = 70
batch_size = 4
load_model = False # If True, need modify 'modeldepthfilter_path' variable


if opt.second_stage_depth:
    print("Changing lr!")
    lr = 5e-5  




def adjust_learning_rate(optimizer, epoch, lr, schedule, learning_rate_decay):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in schedule:
        lr *= learning_rate_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr






def train(opt):
    global gen_test_counter
    global lr 

    if torch.cuda.is_available():
        # set cuda
        device = 'cuda:0'

    else:
        device = 'cpu'

    print("using device {}".format(device) )
    

    train_dataset = DepthDataset(opt, evaluation_mode=False)
    



    train_data_loader = DataLoader(train_dataset, 
                                   batch_size=batch_size, shuffle=not opt.serial_batches,
                                   num_workers=opt.num_threads, pin_memory=opt.pin_memory)


    print('train loader size: ', len(train_data_loader))

    
    
    depthfilter = RelativeDepthFilter(opt)



    if (not os.path.exists(opt.checkpoints_path) ):
        os.makedirs(opt.checkpoints_path)
    if (not os.path.exists(opt.results_path) ):
        os.makedirs(opt.results_path)
    if (not os.path.exists('%s/%s' % (opt.checkpoints_path, opt.name))  ):
        os.makedirs('%s/%s' % (opt.checkpoints_path, opt.name))
    if (not os.path.exists('%s/%s' % (opt.results_path, opt.name)) ):
        os.makedirs('%s/%s' % (opt.results_path, opt.name))





    if load_model: # by default is false
        
        
        modeldepthfilter_path = "apps/checkpoints/Date_08_Jan_22_Time_02_03_43/depthfilter_model_state_dict.pickle" # Date_08_Jan_22_Time_02_03_43 is folder to load

        print('Resuming from ', modeldepthfilter_path)

        with open(modeldepthfilter_path, 'rb') as handle:
           net_state_dict = pickle.load(handle)

        depthfilter.load_state_dict( net_state_dict , strict = True )

                

    opt_log = os.path.join(opt.results_path, opt.name, 'opt.txt')
    with open(opt_log, 'w') as outfile:
        outfile.write(json.dumps(vars(opt), indent=2))



    depthfilter = depthfilter.to(device=device)
 
    optimizer = torch.optim.RMSprop(depthfilter.parameters(), lr=lr, momentum=0, weight_decay=0)



    start_epoch = 0
    for epoch in range(start_epoch, num_of_epoch):

        print("start of epoch {}".format(epoch) )

        depthfilter.train()

        train_len = len(train_data_loader)
        for train_idx, train_data in enumerate(train_data_loader):
            print("batch {}".format(train_idx) )

            # retrieve the data
            if opt.second_stage_depth:

                render_tensor = train_data['original_high_res_render'].to(device=device)  # the renders. Shape of [Batch_size, Channels, Height, Width]
                coarse_depth_map_tensor = train_data['coarse_depth_map'].to(device=device)   # shape of [batch, 1,1024,1024]
                depth_map_tensor = train_data['depth_map'].to(device=device)   # shape of [batch, 1,1024,1024]

                render_tensor = torch.cat( [render_tensor, coarse_depth_map_tensor ], dim=1 )

                if opt.use_normal_map_for_depth_training:
                    nmlF_high_res_tensor = train_data['nmlF_high_res'].to(device=device)
                    render_tensor = torch.cat( [render_tensor, nmlF_high_res_tensor ], dim=1 )



            else:
                render_tensor = train_data['original_high_res_render'].to(device=device)  # the renders. Shape of [Batch_size, Channels, Height, Width]
                depth_map_tensor = train_data['depth_map'].to(device=device)   # shape of [batch, 1,1024,1024]
                center_indicator_tensor = train_data['center_indicator'].to(device=device)  #  shape of [batch, 1,1024,1024]
                
                render_tensor = torch.cat( [render_tensor, center_indicator_tensor ], dim=1 )

                if opt.use_normal_map_for_depth_training:
                    nmlF_high_res_tensor = train_data['nmlF_high_res'].to(device=device)
                    render_tensor = torch.cat( [render_tensor, nmlF_high_res_tensor ], dim=1 )
                


            error = depthfilter.forward(images=render_tensor, groundtruth_depthmap=depth_map_tensor)
        
            optimizer.zero_grad()

            error['Err'].backward()
            curr_loss = error['Err'].item()

            optimizer.step()

            print(
            'Name: {0} | Epoch: {1} | error: {2:.06f} | LR: {3:.06f} '.format(
                opt.name, epoch, curr_loss, lr)
            )



                


        lr = adjust_learning_rate(optimizer, epoch, lr, schedule=depth_schedule, learning_rate_decay=0.05)



        with torch.no_grad():
            if True:


                # save models
                with open( '%s/%s/depthfilter_model_state_dict.pickle' % (opt.checkpoints_path, opt.name) , 'wb') as handle:
                    pickle.dump(depthfilter.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)

                print('generate depth map (train) ...')
                train_dataset.is_train = False
                depthfilter.eval()
                for gen_idx in tqdm(range(1)):

                    index_to_use = gen_test_counter % len(train_dataset)
                    gen_test_counter += 10 # 10 is the number of images for each class
                    train_data = train_dataset.get_item(index=index_to_use) 
                    save_path = '%s/%s/train_eval_epoch%d_%s.obj' % (
                        opt.results_path, opt.name, epoch, train_data['name'])



                    if opt.second_stage_depth:

                        image_tensor = train_data['original_high_res_render'].to(device=device)  # the renders. Shape of [Batch_size, Channels, Height, Width]
                        coarse_depth_map_tensor = train_data['coarse_depth_map'].to(device=device)   # shape of [batch, 1,1024,1024]
                        image_tensor = torch.unsqueeze(image_tensor,0) 
                        coarse_depth_map_tensor = torch.unsqueeze(coarse_depth_map_tensor,0)
                        original_depth_map = train_data['depth_map'].cpu().numpy()

                        image_tensor = torch.cat( [image_tensor, coarse_depth_map_tensor ], dim=1 )


                        if opt.use_normal_map_for_depth_training:
                            nmlF_high_res_tensor = train_data['nmlF_high_res'].to(device=device)
                            nmlF_high_res_tensor = torch.unsqueeze(nmlF_high_res_tensor,0)
                            image_tensor = torch.cat( [image_tensor, nmlF_high_res_tensor ], dim=1 )

                    else:
                        image_tensor = train_data['original_high_res_render'].to(device=device)   
                        image_tensor = torch.unsqueeze(image_tensor,0)
            
                        center_indicator_tensor = train_data['center_indicator'].to(device=device)   
                        center_indicator_tensor = torch.unsqueeze(center_indicator_tensor,0)

                        image_tensor = torch.cat( [image_tensor, center_indicator_tensor ], dim=1 )
                        
                        if opt.use_normal_map_for_depth_training:
                            nmlF_high_res_tensor = train_data['nmlF_high_res'].to(device=device)
                            nmlF_high_res_tensor = torch.unsqueeze(nmlF_high_res_tensor,0)
                            image_tensor = torch.cat( [image_tensor, nmlF_high_res_tensor ], dim=1 )


                        original_depth_map = train_data['depth_map'].cpu().numpy()



                    depthfilter.filter( image_tensor ) # forward-pass  
                    generated_depth_map = depthfilter.generate_depth_map() # [B, C, H, W] where B == C == 1
                    generated_depth_map = generated_depth_map.detach().cpu().numpy()[0,0:1,:,:]

                    if opt.second_stage_depth:
                        save_differencemap_path = save_path[:-4] + 'generated_differencemap.png'
                        save_differencemap = generated_depth_map - coarse_depth_map_tensor.detach().cpu().numpy()[0,0:1,:,:]
                        save_differencemap = (np.transpose( save_differencemap, (1, 2, 0)) ) * 255.0 /4 + 255.0 /2
                        save_differencemap = save_differencemap.astype(np.uint8)[:,:,0] 
                        save_differencemap = Image.fromarray(save_differencemap, 'L')
                        save_differencemap.save(save_differencemap_path)


                    save_img_path = save_path[:-4] + '.png'
                    save_depthmap_path = save_path[:-4] + 'generated_depthmap.png'

                    save_img = (np.transpose(image_tensor[0,:3,...].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5) * 255.0
                    save_img = save_img.astype(np.uint8)
                    save_img = Image.fromarray(save_img)
                    save_img.save(save_img_path)

                    save_depthmap = (np.transpose( generated_depth_map, (1, 2, 0)) ) * 255.0 /2
                    save_depthmap = save_depthmap.astype(np.uint8)[:,:,0] 
                    save_depthmap = Image.fromarray(save_depthmap, 'L')
                    save_depthmap.save(save_depthmap_path)

                    original_depth_map = (original_depth_map[0,:,:]  ) * 255.0 / 2
                    original_depth_map = original_depth_map.astype(np.uint8)
                    original_depth_map = Image.fromarray(original_depth_map, 'L')
                    original_depth_map_save_path = '%s/%s/train_eval_epoch%d_%s_groundtruth.png' % (
                        opt.results_path, opt.name, epoch, train_data['name'])
                    original_depth_map.save(original_depth_map_save_path)

                train_dataset.is_train = True
                




if __name__ == '__main__':
    train(opt)

