import os,sys,time
import bpy
import numpy as np
import shutil
from math import radians
from mathutils import Matrix, Vector


# running instructions (for subject '0501' at a yaw angle of '90' degree): 

#cd to this folder ('rendering_script')
#blender blank.blend -b -P render_normal_map_of_full_mesh.py -- 0501 90

use_gpu = True

curpath = os.path.abspath(os.path.dirname("."))
sys.path.insert(0,curpath)

argv = sys.argv 
argv = argv[argv.index("--") + 1:]

subject = argv[0]
angle = int(argv[1])





# converts angles that are more or equal to 360 degree
if angle>=360:
	angle = angle - 360




light_energy = 0.5e+02  

RESOLUTION = 1024  

BUFFER_PATH =  "buffer_normal_maps_of_full_mesh" # is the output folder
front_normal_map_filename = "rendered_nmlF"
back_normal_map_filename = "rendered_nmlB"

shape_file = os.path.join( 'THuman2.0_Release' ,subject , subject + ".obj")

save_folder_path = os.path.join(curpath, BUFFER_PATH, subject)
if not os.path.exists( save_folder_path ):
	os.makedirs(save_folder_path)


def load_obj_mesh(mesh_file, with_normal=False, with_texture=False):
    vertex_data = []
    norm_data = []
    uv_data = []

    face_data = []
    face_norm_data = []
    face_uv_data = []

    if isinstance(mesh_file, str):
        f = open(mesh_file, "r")
    else:
        f = mesh_file
    for line in f:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue

        if values[0] == 'v':
            v = list(map(float, values[1:4]))
            vertex_data.append(v)
        elif values[0] == 'vn':
            vn = list(map(float, values[1:4]))
            norm_data.append(vn)
        elif values[0] == 'vt':
            vt = list(map(float, values[1:3]))
            uv_data.append(vt)

        elif values[0] == 'f':
            # quad mesh
            if len(values) > 4:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)
                f = list(map(lambda x: int(x.split('/')[0]), [values[3], values[4], values[1]]))
                face_data.append(f)
            # tri mesh
            else:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)
            
            # deal with texture
            if len(values[1].split('/')) >= 2:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
                    face_uv_data.append(f)
                    f = list(map(lambda x: int(x.split('/')[1]), [values[3], values[4], values[1]]))
                    face_uv_data.append(f)
                # tri mesh
                elif len(values[1].split('/')[1]) != 0:
                    f = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
                    face_uv_data.append(f)
            # deal with normal
            if len(values[1].split('/')) == 3:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split('/')[2]), values[1:4]))
                    face_norm_data.append(f)
                    f = list(map(lambda x: int(x.split('/')[2]), [values[3], values[4], values[1]]))
                    face_norm_data.append(f)
                # tri mesh
                elif len(values[1].split('/')[2]) != 0:
                    f = list(map(lambda x: int(x.split('/')[2]), values[1:4]))
                    face_norm_data.append(f)

    vertices = np.array(vertex_data)
    faces = np.array(face_data) - 1

    if with_texture and with_normal:
        uvs = np.array(uv_data)
        face_uvs = np.array(face_uv_data) - 1
        norms = np.array(norm_data)
        if norms.shape[0] == 0:
            norms = compute_normal(vertices, faces)
            face_normals = faces
        else:
            norms = normalize_v3(norms)
            face_normals = np.array(face_norm_data) - 1
        return vertices, faces, norms, face_normals, uvs, face_uvs

    if with_texture:
        uvs = np.array(uv_data)
        face_uvs = np.array(face_uv_data) - 1
        return vertices, faces, uvs, face_uvs

    if with_normal:
        norms = np.array(norm_data)
        norms = normalize_v3(norms)
        face_normals = np.array(face_norm_data) - 1
        return vertices, faces, norms, face_normals

    return vertices, faces


def make_rotate(rx, ry, rz):
	# rx is rotation angle about the x-axis
	# ry is rotation angle about the y-axis
	# rz is rotation angle about the z-axis

	sinX = np.sin(rx)
	sinY = np.sin(ry)
	sinZ = np.sin(rz)

	cosX = np.cos(rx)
	cosY = np.cos(ry)
	cosZ = np.cos(rz)

	Rx = np.zeros((3,3))
	Rx[0, 0] = 1.0
	Rx[1, 1] = cosX
	Rx[1, 2] = -sinX
	Rx[2, 1] = sinX
	Rx[2, 2] = cosX

	Ry = np.zeros((3,3))
	Ry[0, 0] = cosY
	Ry[0, 2] = sinY
	Ry[1, 1] = 1.0
	Ry[2, 0] = -sinY
	Ry[2, 2] = cosY

	Rz = np.zeros((3,3))
	Rz[0, 0] = cosZ
	Rz[0, 1] = -sinZ
	Rz[1, 0] = sinZ
	Rz[1, 1] = cosZ
	Rz[2, 2] = 1.0

	R = np.matmul(np.matmul(Rz,Ry),Rx)
	return R



def setupBlender(angle_to_rotate):
	global overall_vertices

	scene = bpy.context.scene
	camera = bpy.data.objects["Camera"]

	
	bpy.context.scene.render.engine = 'CYCLES' 

	if use_gpu:
		# Set the device_type
		bpy.context.preferences.addons[
		    "cycles"
		].preferences.compute_device_type = "CUDA" # or "OPENCL"

		# Set the device and feature set
		bpy.context.scene.cycles.device = "GPU"

		# get_devices() to let Blender detects GPU device
		bpy.context.preferences.addons["cycles"].preferences.get_devices()
	


	# rotate the vertices first 
	R = make_rotate(0, radians(angle_to_rotate), 0)
	vertices = np.matmul(R, overall_vertices.T).T # shape of [num_of_pts, 3] 



	vmin = vertices.min(0)  
	vmax = vertices.max(0)  

	
	vcenter_vertical = (vmax[1] + vmin[1])/2

	vrange = vmax - vmin
	orthographic_scale_to_use = np.max(vrange) * 1.1  

	camera.data.clip_end = 50.0 # a large value, prevents the output from being clipped away by the camera
	camera.data.type = "ORTHO"

	# set x,y,z coordinates of the camera
	camera.location[0] = 0 
	camera.location[1] = 0
	camera.location[2] = 10.0 #108.0 

	small_x_range = vrange[0]*0.005  
	small_y_range = vrange[1]*0.005
	temp_bool_y = ( vertices[:,1] > (vcenter_vertical - small_y_range)  )  &   ( vertices[:,1] < (vcenter_vertical + small_y_range)  )
	horizontal_line = vertices[temp_bool_y,:] # a horizontal line of actual vertices
	vcenter_horizontal = np.median(horizontal_line, 0)[0]
	temp_bool_x = ( vertices[:,0] > (vcenter_horizontal - small_x_range)  )  &   ( vertices[:,0] < (vcenter_horizontal + small_x_range)  )
	vertical_line = vertices[temp_bool_x,:] # a vertical line of actual vertices
	temp_bool_x_and_y = temp_bool_x & temp_bool_y
	small_cube = vertices[temp_bool_x_and_y,:] # a small cube of vertices centered around the center
	pt_nearest_to_cam = small_cube.max(0)
	z_coor = pt_nearest_to_cam[2]

	mesh_center = np.array([vcenter_horizontal, vcenter_vertical, z_coor])

	# set the rotation of the camera (in radians)
	camera.rotation_euler[0] = 0.0/180*np.pi #87.5/180*np.pi
	camera.rotation_euler[1] = 0.0/180*np.pi #0.759/180*np.pi
	camera.rotation_euler[2] = 0.0/180*np.pi #0.0/180*np.pi #1.85/180*np.pi

	camera.data.ortho_scale = orthographic_scale_to_use

	# do the same for the Light
	light = bpy.data.objects["Light"]

	# set energy of light
	light.data.energy = light_energy 
	

	light.location[0] = (0 + vmin[0] - vcenter_horizontal) / 2  
	light.location[1] = (0 + vmax[1] - vcenter_vertical) / 2   
	light.location[2] = vmax[2] * 8  

	# start using composition in blender
	scene.render.use_compositing = True
	scene.use_nodes = True

	tree = bpy.context.scene.node_tree  
	rl = bpy.context.scene.node_tree.nodes["Render Layers"]  

	fo = tree.nodes.new("CompositorNodeOutputFile")  
	fo.base_path = save_folder_path  # set the save path
	fo.format.file_format = "OPEN_EXR"

	bpy.context.scene.view_layers["View Layer"].use_pass_normal = True

	tree.links.new(rl.outputs["Normal"],fo.inputs["Image"])




	
	# set the resolution of the image rendered
	scene.render.resolution_x = RESOLUTION
	scene.render.resolution_y = RESOLUTION

	# set background color
	world = bpy.data.worlds['World']
	world.use_nodes = True
	bg = world.node_tree.nodes['Background']
	r,g,b = 0,0,0
	bg.inputs[0].default_value[:3] = (r, g, b)


	
	return scene,camera,fo, mesh_center






# import the obj file into Blender
bpy.ops.import_scene.obj(filepath=shape_file)



mesh_name = ""
for _object in bpy.data.objects:
	if subject in _object.name:
		mesh_name = _object.name

mesh = bpy.data.objects[mesh_name] # get the just loaded mesh from blender

# overall_vertices is a global variable
overall_vertices, _ = load_obj_mesh(shape_file, with_normal=False, with_texture=False)






scene,camera,fo, mesh_center = setupBlender(angle_to_rotate=angle)



# start changing mesh's location/translation and rotation

rot_mat = Matrix.Rotation(radians(angle), 4, 'Y')    
backside_rot_mat = Matrix.Rotation(radians(180), 4, 'Y')

orig_loc, orig_rot, orig_scale = mesh.matrix_world.decompose()

vec = Vector(( -mesh_center[0], -mesh_center[1], -mesh_center[2]))
new_orig_loc_mat = Matrix.Translation(vec)

orig_scale_mat = Matrix.Scale(orig_scale[0],4,(1,0,0)) * Matrix.Scale(orig_scale[1],4,(0,1,0)) * Matrix.Scale(orig_scale[2],4,(0,0,1))

# assemble the new matrix
mesh.matrix_world =  new_orig_loc_mat @ rot_mat @ orig_scale_mat


bpy.ops.render.render(write_still=False)
os.rename( os.path.join(save_folder_path, "Image0001.exr") , os.path.join(save_folder_path, front_normal_map_filename + "_" + "{0:03d}".format( int(angle) ) + ".exr"  )  )



# start generating rear normal maps:
mesh.matrix_world =  backside_rot_mat @ new_orig_loc_mat @ rot_mat @ orig_scale_mat

bpy.ops.render.render(write_still=False)
os.rename( os.path.join(save_folder_path, "Image0001.exr") , os.path.join(save_folder_path, back_normal_map_filename + "_" + "{0:03d}".format( int(angle) ) + ".exr"  )  )







