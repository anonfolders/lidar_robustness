import os
import sys

import numpy as np
import random

import argparse

from utils.file_utils import create_new_perturb_folder, get_validation_file_paths
from utils.pcd_utils import read_pc_bin, write_pc_bin, reshape_pcd
from utils.perturbations_utils import rand_uniform_perturbs_global
from utils.perturbations_utils import rand_uniform_perturbs_local
from utils.perturbations_utils import add_directional_perturbations_local

# for normal & laplace
from utils.perturbations_utils import random_perturbs, random_perturbs_local


from utils.perturbations_utils import reflect_perturbs_up, reflect_perturbs_down
from utils.perturbations_utils import local_fp, global_fp

from utils.config import VALIDATION_FRAMES, POINTS_IN_BBOXES_FILENAME
from utils.config import PerturbationMode as PM, LocalDirection as LD


data_path = 'data/'
original_pcd_path = 'data/baseline/velodyne/'


def get_empty_dirs(path):
    empty_dirs = []
    for root, dirs, _ in os.walk(path):
        for idx, dir in enumerate(dirs):
            dir_path = os.path.join(root, dir)
            if len(os.listdir(dir_path)) == 0:
                empty_dirs.append(dir_path)
    return empty_dirs


def is_point_in_bboxes(point, bbox_points):
    # point is np.array
    # bbox_points in np.array of np.array
    return np.any(np.all(point == bbox_points, axis=1))


dirs_dict = {
    '+x': LD.X_PLUS,
    '-x': LD.X_MINUS,
    '+y': LD.Y_PLUS,
    '-y': LD.Y_MINUS,
    '+z': LD.Z_PLUS,
    '-z': LD.Z_MINUS
}

types_dict = {
    'global_uniform': PM.GLOBAL_UNIFORM,
    'global_normal': PM.GLOBAL_NORMAL,
    'global_laplace': PM.GLOBAL_LAPLACE,

    'local_uniform': PM.LOCAL_UNIFORM,
    'local_normal': PM.LOCAL_NORMAL,
    'local_laplace': PM.LOCAL_LAPLACE,

    'direct_uniform':PM.DIRECT_UNIFORM,
    'direct_normal':PM.DIRECT_NORMAL,
    'direct_laplace':PM.DIRECT_LAPLACE,

    'reflect_down': PM.REFLECT_DOWNSAMPLE,
    'reflect_up': PM.REFLECT_UPSAMPLE,
    'reflect_other': PM.REFLECT_OTHER
}



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run options')
    parser.add_argument('--count', dest='rounds_count', type=int, help='Number of rounds to generate')
    parser.add_argument('--type', dest='perturbation_type', type=str, help='Type of perturbation to apply')
    parser.add_argument('--direction', dest='direction', type=str, help='Direction in case of local directional perturbation')
    args = parser.parse_args()

    pmode = None
    loc_dir = None
    if args.perturbation_type in types_dict:
        pmode = types_dict[args.perturbation_type]
        if pmode is PM.DIRECT_UNIFORM or pmode is PM.DIRECT_NORMAL or pmode is PM.DIRECT_LAPLACE:
            if args.direction in dirs_dict:
                loc_dir = dirs_dict[args.direction]
            else:
                print(f'Invalid direction select from {dirs_dict.keys()}')
                sys.exit()
    else:
        print(f'Invalid perturbation type, select from {types_dict.keys()}')
        sys.exit()


    # Step 1: create empty folders
    new_folder_paths = []

    # for _ in range(args.rounds_count):
    for _ in range(1):
        new_folder_paths.append(create_new_perturb_folder(data_path))

    # Step 2: get the lidar data & gt bboxes
    pcd_files = sorted(get_validation_file_paths(original_pcd_path, VALIDATION_FRAMES))

    all_points_in_bboxes = []
    with open(POINTS_IN_BBOXES_FILENAME, 'r') as fd:
        for line in fd.readlines():
            all_points_in_bboxes.append(line.strip().split())
    
    # Step 3: get validation frame names (ids only)
    frame_ids = []
    with open(VALIDATION_FRAMES, 'r') as fd:
        for lidx, line in enumerate(fd.readlines()):
            frame_id = line.strip()
            frame_ids.append(frame_id)
    frame_ids = sorted(frame_ids)

    # Step 4: populate new folder with perturbation
    # read lidar data for files

    for pf_idx, pf in enumerate(pcd_files):
        print('{}/{}'.format(pf_idx, len(pcd_files)))
        # load raw pcd
        pf_data = read_pc_bin(pf)

        # Step 1: Load projected_bboxes which contains the points in gt bboxes        
        fname = pf.split('/')[-1]
        frame_id = fname.split('.')[0]
        print(frame_id)

        new_folder_path = new_folder_paths[0]
        print(new_folder_path)

        if pmode is PM.GLOBAL_UNIFORM:
            pf_data_w_pertubations = rand_uniform_perturbs_global(pcd=pf_data)

        elif pmode is PM.GLOBAL_NORMAL:
            pf_data_w_pertubations = random_perturbs(pcd=pf_data, mode='normal')

        elif pmode is PM.GLOBAL_LAPLACE:
            pf_data_w_pertubations = random_perturbs(pcd=pf_data, mode='laplace')

        elif pmode is PM.GLOBAL_FALSE_POSITIVES:
            pf_data_w_pertubations = global_fp(pcd=bin_pcd, points_in_obs=points_in_obs)
        else:
            # get points in obstacles in this frame
            points_in_bboxes_in_frame = [point[2:] for point in all_points_in_bboxes if point[0] == frame_id]

            # if there are no obstacles, skip
            if len(points_in_bboxes_in_frame) == 0:
                continue

            # reshape original point cloud
            bin_pcd = reshape_pcd(pf_data)

            points_in_bboxes_in_frame = np.array(points_in_bboxes_in_frame).astype(np.float32)

            # Match points in bboxes with raw pcd to get an array of 0s and 1s 
            # where 1s shows that point is in bbox
            points_in_obs = []
            points_in_obs_up = []
            points_in_obs_down = []
            dropped_point = True
            for pp_idx, pcd_point in enumerate(bin_pcd):
                if is_point_in_bboxes(pcd_point, points_in_bboxes_in_frame):
                    points_in_obs.append([1, 1, 1, 0])
                    r = random.random()
                    if r <= 0.6:
                        points_in_obs_down.append([0, 0, 0, 0])
                    else:
                        points_in_obs_down.append([1, 1, 1, 0])

                    if r <= 0.67:
                        points_in_obs_up.append([1, 1, 1, 0])
                    else:
                        points_in_obs_up.append([0, 0, 0, 0])
                else:
                    points_in_obs.append([0, 0, 0, 0])
                    points_in_obs_up.append([0, 0, 0, 0])
                    points_in_obs_down.append([1, 1, 1, 0])

            # convert to np.array for multiplication
            points_in_obs = np.array(points_in_obs)
            if pmode is PM.LOCAL_UNIFORM:
                pf_data_w_pertubations = rand_uniform_perturbs_local(pcd=bin_pcd, points_in_obs=points_in_obs)

            elif pmode is PM.LOCAL_NORMAL:
                pf_data_w_pertubations = random_perturbs_local(pcd=bin_pcd, mode='normal', points_in_obs=points_in_obs)

            elif pmode is PM.LOCAL_LAPLACE:
                pf_data_w_pertubations = random_perturbs_local(pcd=bin_pcd, mode='laplace', points_in_obs=points_in_obs)

            elif pmode is PM.DIRECT_UNIFORM:
                pf_data_w_pertubations = add_directional_perturbations_local(pcd=bin_pcd, points_in_obs=points_in_obs, direction=loc_dir, mode='uniform')

            elif pmode is PM.DIRECT_NORMAL:
                pf_data_w_pertubations = add_directional_perturbations_local(pcd=bin_pcd, points_in_obs=points_in_obs, direction=loc_dir, mode='normal')

            elif pmode is PM.DIRECT_LAPLACE:
                pf_data_w_pertubations = add_directional_perturbations_local(pcd=bin_pcd, points_in_obs=points_in_obs, direction=loc_dir, mode='laplace')

            elif pmode is PM.LOCAL_FALSE_POSITIVES:
                pf_data_w_pertubations = local_fp(pcd=bin_pcd, points_in_obs=points_in_obs)

            elif pmode is PM.REFLECT_DOWNSAMPLE:
                pf_data_w_pertubations = reflect_perturbs_down(pcd=bin_pcd, points_in_obs=points_in_obs_down)

            elif pmode is PM.REFLECT_UPSAMPLE:
                pf_data_w_pertubations = reflect_perturbs_up(pcd=bin_pcd, points_in_obs=points_in_obs_up)

            elif pmode is PM.REFLECT_OTHER:
                pass

        fpath = os.path.join(new_folder_path, fname)
        write_pc_bin(pf_data_w_pertubations, fpath)
