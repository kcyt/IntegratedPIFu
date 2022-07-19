# IntegratedPIFu
Official Implementation of IntegratedPIFu (ECCV 2022)

## Prerequisites:
### 1) Request permission to use THuman2.0 dataset (https://github.com/ytrock/THuman2.0-Dataset). 
After permission granted, download the dataset (THuman2.0_Release). Put the "THuman2.0_Release" folder inside the "rendering_script" folder. 

### 2) Rendering Images, Normal maps, and Depth maps
Run render_full_mesh.py, render_normal_map_of_full_mesh.py, and render_depth_map_of_full_mesh.py to generate RGB image, normal map, and depth map respectively. For example, to render the RGB image of subject '0501' at a yaw angle of '90' degrees, run: 
`blender blank.blend -b -P render_full_mesh.py -- 0501 90`
Replace render_full_mesh.py with render_normal_map_of_full_mesh.py or render_depth_map_of_full_mesh.py to render normal map and depth map respectively. 

### 3) Rendering Human Parsing maps
While we cannot really render human parsing maps, we can get pseudo-groundtruths using a pre-trained model. Go to https://github.com/GoGoDuck912/Self-Correction-Human-Parsing and you can run their google colab demo to obtain human parsing maps from the rgb images rendered in the "rendering_script/buffer_fixed_full_mesh" folder. Put the results in "rendering_script/render_human_parse_results". Each subject should have a separate subfolder (e.g. "rendering_script/render_human_parse_results/0510/rendered_parse_180.npy")

## To Run :
Run the script train_integratedPIFu.py. Configuration can be set in lib/options.py file.
