
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

seed = 10 
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

parser = BaseOptions()
opt = parser.parse()
gen_test_counter = 0

lr = 2e-4  
normal_schedule = [20] # epoch to reduce lr at
batch_size = 2
load_model = False  # If True, remember to set the 'F_modelnormal_path' and 'B_modelnormal_path'



def adjust_learning_rate(optimizer_list, epoch, lr, schedule, learning_rate_decay):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in schedule:
        lr *= learning_rate_decay
        for optimizer in optimizer_list:
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
    
    train_dataset = NormalDataset(opt, evaluation_mode=False)
    
    train_data_loader = DataLoader(train_dataset, 
                                   batch_size=batch_size, shuffle=not opt.serial_batches,
                                   num_workers=opt.num_threads, pin_memory=opt.pin_memory)


    print('train loader size: ', len(train_data_loader))


    smoothL1Loss = nn.SmoothL1Loss()


    
    netF = define_G(3, 3, 64, "global", 4, 9, 1, 3, "instance")

    netB = define_G(3, 3, 64, "global", 4, 9, 1, 3, "instance")




    if load_model: 

        
        F_modelnormal_path = "apps/checkpoints/Date_12_Nov_21_Time_01_38_54/netF_model_state_dict.pickle" # Date_12_Nov_21_Time_01_38_54 is folder to load
        B_modelnormal_path = "apps/checkpoints/Date_12_Nov_21_Time_01_38_54/netB_model_state_dict.pickle"

        print('Resuming from ', F_modelnormal_path)
        print('Resuming from ', B_modelnormal_path)

        with open(F_modelnormal_path, 'rb') as handle:
           netF_state_dict = pickle.load(handle)

        with open(B_modelnormal_path, 'rb') as handle:
           netB_state_dict = pickle.load(handle)

        netF.load_state_dict( netF_state_dict , strict = True )
        netB.load_state_dict( netB_state_dict , strict = True )
        
        



    if (not os.path.exists(opt.checkpoints_path) ):
        os.makedirs(opt.checkpoints_path)
    if (not os.path.exists(opt.results_path) ):
        os.makedirs(opt.results_path)
    if (not os.path.exists('%s/%s' % (opt.checkpoints_path, opt.name))  ):
        os.makedirs('%s/%s' % (opt.checkpoints_path, opt.name))
    if (not os.path.exists('%s/%s' % (opt.results_path, opt.name)) ):
        os.makedirs('%s/%s' % (opt.results_path, opt.name))



    opt_log = os.path.join(opt.results_path, opt.name, 'opt.txt')
    with open(opt_log, 'w') as outfile:
        outfile.write(json.dumps(vars(opt), indent=2))




    netF = netF.to(device=device)
    netB = netB.to(device=device)

    
 
    optimizer_netF = torch.optim.RMSprop(netF.parameters(), lr=lr, momentum=0, weight_decay=0)
    optimizer_netB = torch.optim.RMSprop(netB.parameters(), lr=lr, momentum=0, weight_decay=0)



    start_epoch = 0
    for epoch in range(start_epoch, opt.num_epoch):

        print("start of epoch {}".format(epoch) )

        netF.train()
        netB.train()

        train_len = len(train_data_loader)
        for train_idx, train_data in enumerate(train_data_loader):
            print("batch {}".format(train_idx) )

            # retrieve the data
            render_tensor = train_data['original_high_res_render'].to(device=device)  # the renders. Shape of [Batch_size, Channels, Height, Width]
            nmlF_high_res_tensor = train_data['nmlF_high_res'].to(device=device)   # shape of [batch, 3,1024,1024]
            nmlB_high_res_tensor = train_data['nmlB_high_res'].to(device=device)   # shape of [batch, 3,1024,1024]

        
            res_netF = netF.forward(render_tensor)
            res_netB = netB.forward(render_tensor)


            err_netF = smoothL1Loss(res_netF, nmlF_high_res_tensor) 
            err_netB = smoothL1Loss(res_netB, nmlB_high_res_tensor)
   


        
            optimizer_netF.zero_grad()
            err_netF.backward()
            curr_loss_netF = err_netF.item()
            optimizer_netF.step()

            optimizer_netB.zero_grad()
            err_netB.backward()
            curr_loss_netB = err_netB.item()
            optimizer_netB.step()

            print(
            'Name: {0} | Epoch: {1} | curr_loss_netF: {2:.06f} | curr_loss_netB: {3:.06f}  | LR: {4:.06f} '.format(
                opt.name, epoch, curr_loss_netF, curr_loss_netB, lr)
            )


                

        lr = adjust_learning_rate( [optimizer_netF, optimizer_netB] , epoch, lr, schedule=normal_schedule, learning_rate_decay=0.1)



        with torch.no_grad():
            if True:

                # save models
                with open( '%s/%s/netF_model_state_dict.pickle' % (opt.checkpoints_path, opt.name) , 'wb') as handle:
                    pickle.dump(netF.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)
                with open( '%s/%s/netB_model_state_dict.pickle' % (opt.checkpoints_path, opt.name) , 'wb') as handle:
                    pickle.dump(netB.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)

                print('generate normal map (train) ...')
                train_dataset.is_train = False
                netF.eval()
                netB.eval()
                for gen_idx in tqdm(range(1)):

                    index_to_use = gen_test_counter % len(train_dataset)
                    gen_test_counter += 10 # 10 is the number of images for each class
                    train_data = train_dataset.get_item(index=index_to_use) 
                    save_path = '%s/%s/train_eval_epoch%d_%s.obj' % (
                        opt.results_path, opt.name, epoch, train_data['name'])


                    image_tensor = train_data['original_high_res_render'].to(device=device) 
                    image_tensor = torch.unsqueeze(image_tensor,0)

                    original_nmlF_map = train_data['nmlF_high_res'].cpu().numpy()
                    original_nmlB_map = train_data['nmlB_high_res'].cpu().numpy()


                    res_netF = netF.forward(image_tensor)
                    res_netB = netB.forward(image_tensor)

                    res_netF = res_netF.detach().cpu().numpy()[0,:,:,:]
                    res_netB = res_netB.detach().cpu().numpy()[0,:,:,:]


                    save_netF_normalmap_path = save_path[:-4] + 'netF_normalmap.png'
                    save_netB_normalmap_path = save_path[:-4] + 'netB_normalmap.png'
                    numpy_save_netF_normalmap_path = save_path[:-4] + 'netF_normalmap.npy'
                    numpy_save_netB_normalmap_path = save_path[:-4] + 'netB_normalmap.npy'
                    GT_netF_normalmap_path = save_path[:-4] + 'netF_groundtruth.png'
                    GT_netB_normalmap_path = save_path[:-4] + 'netB_groundtruth.png'

                    np.save(numpy_save_netF_normalmap_path , res_netF)
                    np.save(numpy_save_netB_normalmap_path , res_netB)


                    save_netF_normalmap = (np.transpose(res_netF, (1, 2, 0)) * 0.5 + 0.5) * 255.0
                    save_netF_normalmap = save_netF_normalmap.astype(np.uint8)
                    save_netF_normalmap = Image.fromarray(save_netF_normalmap)
                    save_netF_normalmap.save(save_netF_normalmap_path)

                    save_netB_normalmap = (np.transpose(res_netB, (1, 2, 0)) * 0.5 + 0.5) * 255.0
                    save_netB_normalmap = save_netB_normalmap.astype(np.uint8)
                    save_netB_normalmap = Image.fromarray(save_netB_normalmap)
                    save_netB_normalmap.save(save_netB_normalmap_path)


                    GT_netF_normalmap = (np.transpose(original_nmlF_map, (1, 2, 0)) * 0.5 + 0.5) * 255.0
                    GT_netF_normalmap = GT_netF_normalmap.astype(np.uint8)
                    GT_netF_normalmap = Image.fromarray(GT_netF_normalmap)
                    GT_netF_normalmap.save(GT_netF_normalmap_path)

                    GT_netB_normalmap = (np.transpose(original_nmlB_map, (1, 2, 0)) * 0.5 + 0.5) * 255.0
                    GT_netB_normalmap = GT_netB_normalmap.astype(np.uint8)
                    GT_netB_normalmap = Image.fromarray(GT_netB_normalmap)
                    GT_netB_normalmap.save(GT_netB_normalmap_path)



                train_dataset.is_train = True






if __name__ == '__main__':
    train(opt)

