# IntegratedPIFu
Official Implementation of IntegratedPIFu (ECCV 2022)

## Prerequisites:
### 1) Request permission to use THuman2.0 dataset (https://github.com/ytrock/THuman2.0-Dataset). 
After permission granted, download the dataset (THuman2.0_Release). Put the "THuman2.0_Release" folder inside the "rendering_script" folder. 

### 2) Rendering Images, Normal maps, and Depth maps
Run render_full_mesh.py, render_normal_map_of_full_mesh.py, and render_depth_map_of_full_mesh.py to generate RGB image, normal map, and depth map respectively. For example, to render the RGB image of subject '0501' at a yaw angle of '90' degrees, run: 
`blender blank.blend -b -P render_full_mesh.py -- 0501 90`
Replace render_full_mesh.py with render_normal_map_of_full_mesh.py or render_depth_map_of_full_mesh.py to render normal map and depth map respectively. 

## To Run :
Run the script train_integratedPIFu.py. Configuration can be set in lib/options.py file.
