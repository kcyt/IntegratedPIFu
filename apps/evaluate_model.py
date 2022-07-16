
import trimesh
import trimesh.proximity
import trimesh.sample
import numpy as np
import os

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

# Modify the two variables below as needed.
source_mesh_folder = "/mnt/lustre/kennard.chan/specialized_pifuhd/apps/results/Date_11_Jul_22_Time_01_12_07" # Depth_HighRes_Comp_DOS Thuman
use_Buff_dataset = True


print('source_mesh_folder:', source_mesh_folder)


num_samples_to_use = 10000




def run_test_mesh_already_prepared():

        if use_Buff_dataset:
            test_subject_list = np.loadtxt("/mnt/lustre/kennard.chan/render_Buff_with_blender/buff_subject_testing.txt", dtype=str).tolist()
        else:
            test_subject_list = np.loadtxt("/mnt/lustre/kennard.chan/getTestSet/test_set_list.txt", dtype=str).tolist()

        total_chamfer_distance = []
        total_point_to_surface_distance = []
        for subject in test_subject_list:

            if use_Buff_dataset:
                folder_name = subject.split('_')[0]
                attire_name = subject.split('_', 1)[1] 
                GT_mesh_path = os.path.join("/mnt/lustre/kennard.chan/render_Buff_with_blender/simple_buff_dataset/", folder_name, attire_name + '.ply' )
            else:
                GT_mesh_path = os.path.join("/mnt/lustre/kennard.chan/split_mesh/results", subject, '%s_clean.obj' % subject)

            source_mesh_path = os.path.join(source_mesh_folder, 'test_%s.obj' % subject)
            if not os.path.exists(source_mesh_path):
                source_mesh_path = os.path.join(source_mesh_folder, 'test_%s_1.obj' % subject)


            GT_mesh = trimesh.load(GT_mesh_path)
            source_mesh = trimesh.load(source_mesh_path)


            chamfer_distance = get_chamfer_dist(src_mesh=source_mesh, tgt_mesh=GT_mesh, num_samples=num_samples_to_use )
            point_to_surface_distance = get_surface_dist(src_mesh=source_mesh, tgt_mesh=GT_mesh, num_samples=num_samples_to_use )


            print("subject: ", subject)

            print("chamfer_distance: ", chamfer_distance)
            total_chamfer_distance.append(chamfer_distance)

            print("point_to_surface_distance: ", point_to_surface_distance)
            total_point_to_surface_distance.append(point_to_surface_distance)


        average_chamfer_distance = np.mean(total_chamfer_distance) 
        average_point_to_surface_distance = np.mean(total_point_to_surface_distance) 

        print("average_chamfer_distance:",average_chamfer_distance)
        print("average_point_to_surface_distance:",average_point_to_surface_distance)







def get_chamfer_dist(src_mesh, tgt_mesh,  num_samples=10000):
    # Chamfer
    src_surf_pts, _ = trimesh.sample.sample_surface(src_mesh, num_samples) # src_surf_pts  has shape of (num_of_pts, 3)
    tgt_surf_pts, _ = trimesh.sample.sample_surface(tgt_mesh, num_samples)

    _, src_tgt_dist, _ = trimesh.proximity.closest_point(tgt_mesh, src_surf_pts)
    _, tgt_src_dist, _ = trimesh.proximity.closest_point(src_mesh, tgt_surf_pts)

    src_tgt_dist[np.isnan(src_tgt_dist)] = 0
    tgt_src_dist[np.isnan(tgt_src_dist)] = 0

    src_tgt_dist = np.mean(np.square(src_tgt_dist))
    tgt_src_dist = np.mean(np.square(tgt_src_dist))

    chamfer_dist = (src_tgt_dist + tgt_src_dist) / 2

    return chamfer_dist




def get_surface_dist(src_mesh, tgt_mesh, num_samples=10000):
    # P2S
    src_surf_pts, _ = trimesh.sample.sample_surface(src_mesh, num_samples)

    _, src_tgt_dist, _ = trimesh.proximity.closest_point(tgt_mesh, src_surf_pts)

    src_tgt_dist[np.isnan(src_tgt_dist)] = 0

    src_tgt_dist = np.mean(np.square(src_tgt_dist))

    return src_tgt_dist




def quick_get_chamfer_and_surface_dist(src_mesh, tgt_mesh,  num_samples=10000):
    # Chamfer
    src_surf_pts, _ = trimesh.sample.sample_surface(src_mesh, num_samples) # src_surf_pts  has shape of (num_of_pts, 3)
    tgt_surf_pts, _ = trimesh.sample.sample_surface(tgt_mesh, num_samples)

    _, src_tgt_dist, _ = trimesh.proximity.closest_point(tgt_mesh, src_surf_pts)
    _, tgt_src_dist, _ = trimesh.proximity.closest_point(src_mesh, tgt_surf_pts)

    src_tgt_dist[np.isnan(src_tgt_dist)] = 0
    tgt_src_dist[np.isnan(tgt_src_dist)] = 0

    src_tgt_dist = np.mean(np.square(src_tgt_dist))
    tgt_src_dist = np.mean(np.square(tgt_src_dist))

    chamfer_dist = (src_tgt_dist + tgt_src_dist) / 2
    surface_dist = src_tgt_dist

    return chamfer_dist, surface_dist





if __name__ == "__main__":


    run_test_mesh_already_prepared()




