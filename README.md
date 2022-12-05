# IntegratedPIFu
Official Implementation of IntegratedPIFu (ECCV 2022)

## Update:
### a) To download the pretrained model (Both are required): 
    Low Resolution PIFu: https://drive.google.com/file/d/1Wn11fYQpiSiydzHD4tPe1GFnvfEuPIKx/view?usp=share_link
    HRI: https://drive.google.com/file/d/158dg4adoyMpwX7EnGuXch16TY0704Qv5/view?usp=share_link
    Frontal Normal Map generator: https://drive.google.com/file/d/10_6w4DKODuzYxC88UgwPp5jHb6SPg7_5/view?usp=share_link
    Rear Normal Map generator: https://drive.google.com/file/d/10FD3qNyGw6fajoBEsHMOLeehM_F63z4T/view?usp=share_link

This pretrained model is the best model (qauntitatively) in our paper, and does not require depth or human parsing maps. So please (refer to lib/options.py) set `--use_human_parse_maps` and  `--use_depth_map` to False. 

Also, set  `use_back_normal` to True (in our experiments, this does not affect the results by too much, but we happen to train this pre-trained model with this.). To generate the normal maps using the pretrained frontal and rear normal map generators, please download the pretrained normal generators' weights refer to apps/generatemaps_normalmodel.py and modify the file paths according to what you need (before running it).
   
The pretrained models can be used by setting the configuration in train_integratedPIFu.py i.e. Make sure to set load_model_weights and load_model_weights_for_high_res_too to True. Then, change `modelG_path = os.path.join( checkpoint_folder_to_load_low_res ,"netG_model_state_dict_epoch{0}.pickle".format(epoch_to_load_from_low_res) )`
into
`modelG_path = <path of the pretrained weights of Low Resolution PIFu>a`

And change `modelhighResG_path = os.path.join( checkpoint_folder_to_load_high_res, "highRes_netG_model_state_dict_epoch{0}.pickle".format(epoch_to_load_from_high_res) )`
into
`modelhighResG_path = <path of the pretrained weights of HRI>`



## Prerequisites:
### 1) Request permission to use THuman2.0 dataset (https://github.com/ytrock/THuman2.0-Dataset). 
After permission granted, download the dataset (THuman2.0_Release). Put the "THuman2.0_Release" folder inside the "rendering_script" folder. 

### 2) Rendering Images, Normal maps, and Depth maps
Run render_full_mesh.py, render_normal_map_of_full_mesh.py, and render_depth_map_of_full_mesh.py to generate RGB image, normal map, and depth map respectively. For example, to render the RGB image of subject '0501' at a yaw angle of '90' degrees, run: 
`blender blank.blend -b -P render_full_mesh.py -- 0501 90`
Replace render_full_mesh.py with render_normal_map_of_full_mesh.py or render_depth_map_of_full_mesh.py to render normal map and depth map respectively. 

### 3) Rendering Human Parsing maps
While we cannot really render human parsing maps, we can get pseudo-groundtruths using a pre-trained model. Go to https://github.com/kcyt/Self-Correction-Human-Parsing and follow the instructions to obtain human parsing maps from the rgb images rendered in the "rendering_script/buffer_fixed_full_mesh" folder. Put the results (a 'render_human_parse_results' folder) into "rendering_script". Each subject should have a separate subfolder (e.g. "rendering_script/render_human_parse_results/0510/rendered_parse_180.npy")

## To Run :
Run the script train_integratedPIFu.py. Configuration can be set in lib/options.py file.
