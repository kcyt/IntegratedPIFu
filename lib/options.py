
import argparse
import os
import time 

class BaseOptions():
    def __init__(self):
        self.initialized = False
        self.parser = None

    def initialize(self, parser):

        parser.add_argument('--useValidationSet', default=True)


        # start option check
        parser.add_argument('--useDOS', default=True, help='depth oriented sampling')
        parser.add_argument('--use_mask_for_rendering_high_res',default=True) 
        parser.add_argument('--use_mask_for_rendering_low_res',default=False) # Not used in the paper.


        parser.add_argument('--update_low_res_pifu', default=True)
        parser.add_argument('--epoch_interval_to_update_low_res_pifu', default=1)
        parser.add_argument('--epoch_to_start_update_low_res_pifu', default=10)
        parser.add_argument('--epoch_to_end_update_low_res_pifu', default=30)



        # high res component
        parser.add_argument('--use_High_Res_Component', default=True) 
        parser.add_argument('--High_Res_Component_sigma', default=2.0)

        parser.add_argument('--use_human_parse_maps', default=True) 
        parser.add_argument('--use_groundtruth_human_parse_maps', default=False)

        # depth usage in low res PIFU models 
        parser.add_argument('--use_depth_map', default=True)
        parser.add_argument('--depth_in_front', default=True) # Used in the paper.
        parser.add_argument('--useGTdepthmap',default=False)


        # depth filter training
        parser.add_argument('--use_normal_map_for_depth_training', default=True)
        parser.add_argument('--second_stage_depth', default=False)

        # parse filter training
        parser.add_argument('--use_normal_map_for_parse_training', default=True)

        
        parser.add_argument('--num_sample_inout', type=int, default=16000, help='# of sampling points')
        #parser.add_argument('--num_sample_inout', type=int, default=8000, help='# of sampling points')

        parser.add_argument('--sigma_low_resolution_pifu', type=float, default=3.5, help='sigma for sampling')
        parser.add_argument('--sigma_high_resolution_pifu', type=float, default=2.0, help='sigma for sampling') 

        parser.add_argument('--use_front_normal', default=True)
        parser.add_argument('--use_back_normal', default=False)


        parser.add_argument('--num_epoch', type=int, default=35, help='num epoch to train')


        parser.add_argument('--learning_rate_G', type=float, default=1e-3, help='adam learning rate for low res')
        parser.add_argument('--learning_rate_MR', type=float, default=1e-3, help='adam learning rate for high res')
        parser.add_argument('--learning_rate_low_res_finetune', type=float, default=1e-4)


        parser.add_argument('--resolution', type=int, default=256, help='# of grid in mesh reconstruction')

        parser.add_argument('--num_threads', default=2, type=int, help='# sthreads for loading data')








        # less commonly used options:
        parser.add_argument('--allow_highres_to_use_depth', default=False) # Not used in the paper.

        parser.add_argument('--ratio_of_way_inside_points',default=0.05) 
        parser.add_argument('--ratio_of_outside_points', default=0.05) 

        parser.add_argument('--no_first_down_sampling', default=False) # set to True to remove the downsampling

        parser.add_argument('--use_groundtruth_normal_maps', default=False) # for training the dif PIFU models

        
        parser.add_argument('--schedule', type=int, nargs='+', default=[90000, 90001, 90002],
                            help='Decrease learning rate at these epochs.')



        parser.add_argument('--linear_anneal_sigma', action='store_true', help='linear annealing of sigma')
        parser.add_argument('--mask_ratio', type=float, default=0.5, help='maximum sigma for sampling')
        parser.add_argument('--sampling_parts', action='store_true', help='Sampling on the fly')
        parser.add_argument('--hg_depth_high_res', type=int, default=2, help='# of stacked layer of hourglass')
        parser.add_argument('--hg_depth_low_res', type=int, default=2, help='# of stacked layer of hourglass')
        parser.add_argument('--mlp_norm', type=str, default='none', help='normalization for volume branch')
        parser.add_argument('--occ_loss_type', type=str, default='bce', help='bce | brock_bce | mse')
        parser.add_argument('--uniform_ratio', type=float, default=0.2, help='maximum sigma for sampling')


        parser.add_argument('--loadSize', type=int, default=1024, help='load size of input image')

        # Experiment related
        timestamp = time.strftime('Date_%d_%b_%y_Time_%H_%M_%S')
        parser.add_argument('--name', type=str, default=timestamp,
                           help='name of the experiment. It decides where to store samples and models')


        parser.add_argument('--batch_size', type=int, default=2, help='input batch size')


        parser.add_argument('--serial_batches', action='store_true',
                             help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--pin_memory', action='store_true', help='pin_memory')



        parser.add_argument('--z_size', type=float, default=200.0, help='z normalization factor')
        #parser.add_argument('--z_size', type=float, default=400.0, help='z normalization factor')


        
        parser.add_argument('--norm', type=str, default='group',
                             help='instance normalization or batch normalization or group normalization')

        # Image filter General
        parser.add_argument('--netG', type=str, default='hgpifu', help='piximp | fanimp | hghpifu')

        # hgimp specific
        parser.add_argument('--num_stack_low_res', type=int, default=4, help='# of hourglass')
        parser.add_argument('--num_stack_high_res', type=int, default=1, help='# of hourglass')


        parser.add_argument('--hg_down', type=str, default='ave_pool', help='ave pool || conv64 || conv128')
        parser.add_argument('--hg_dim_low_res', type=int, default=256, help='256 | 512')
        parser.add_argument('--hg_dim_high_res', type=int, default=16)


        # Classification General
        parser.add_argument('--mlp_dim_low_res', nargs='+', default=[257, 1024, 512, 256, 128, 1], type=int,
                             help='# of dimensions of mlp. no need to put the first channel')

        parser.add_argument('--mlp_res_layers_low_res', nargs='+', default=[2,3,4], type=int,
                             help='leyers that has skip connection. use 0 for no residual pass')

        parser.add_argument('--mlp_res_layers_high_res', nargs='+', default=[1,2], type=int,
                             help='leyers that has skip connection. use 0 for no residual pass')

        parser.add_argument('--merge_layer_low_res', type=int, default=2)


        parser.add_argument('--mlp_dim_high_res', nargs='+', default=[272, 512, 256, 128, 1], type=int,
                             help='# of dimensions of mlp.')

    


        parser.add_argument('--learning_rate_decay', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
        parser.add_argument('--no_finetune', action='store_true', help='fine tuning netG in training C')


        # path
        parser.add_argument('--checkpoints_path', type=str, default='./checkpoints', help='path to save checkpoints')
        parser.add_argument('--results_path', type=str, default='./results', help='path to save results ply')
        


        # for multi resolution
        parser.add_argument('--loadSizeBig', type=int, default=1024, help='load size of input image')
        parser.add_argument('--loadSizeLocal', type=int, default=512, help='load size of input image')
        parser.add_argument('--loadSizeGlobal', type=int, default=512, help='load size of input image')



        # for reconstruction
        parser.add_argument('--start_id', type=int, default=-1)
        parser.add_argument('--end_id', type=int, default=-1)



        # special tasks
        self.initialized = True
        return parser

    def gather_options(self, args=None):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
            self.parser = parser

        if args is None:
            return self.parser.parse_args()
        else:
            return self.parser.parse_args(args)

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def parse(self, args=None):
        opt = self.gather_options(args)

             
        return opt
