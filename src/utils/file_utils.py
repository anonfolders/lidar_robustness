import os


def create_new_child_perturb_folders(root, name):
    new_dir_dat_path = os.path.join(root, name)
    os.makedirs(new_dir_dat_path)
    return new_dir_dat_path


def create_new_perturb_folder(path):
    # get only immediate folders
    root, dirs, _ = next(os.walk(path))
    # dirs = sorted(dirs)
    dirs = [d for d in dirs if d != 'baseline']
    # handles base case when empty
    new_dir_idx_start = 0
    if dirs:
        new_dir_idx_start = int(max(dirs)) + 1

    new_dir_name = str(new_dir_idx_start).zfill(6)
    new_dir_path = os.path.join(root, new_dir_name)

    # create output folders
    results_folders = ['results/00', 'results/01', 'results/02', 'results/03', 'results/04']
    for r in results_folders:
        create_new_child_perturb_folders(new_dir_path, r)

    # create input perturbations
    new_dir_dat_path = create_new_child_perturb_folders(new_dir_path, 'velodyne')
    print('Created {}'.format(new_dir_dat_path))
    return new_dir_dat_path


def get_validation_file_paths(path, validation_frames):
    validation_bin_names = []
    with open(validation_frames, 'r') as fd:
        for fid in fd.readlines():
            val_bin_name = '{}.bin'.format(fid.strip())
            validation_bin_names.append(val_bin_name)

    list_of_files = get_files_in_folder(path, validation_bin_names)
    return list_of_files


def get_files_in_folder(path, validation_bin_names):
    list_of_files = []
    for root, _, files in os.walk(path):
        for file in files:
            if file in validation_bin_names:
                list_of_files.append(os.path.join(root, file))
    return list_of_files


def get_files_in_folder_2(path):
    list_of_files = []
    list_of_files_with_path = []
    for root, _, files in os.walk(path):
        for file in files:
            list_of_files_with_path.append(os.path.join(root, file))
            list_of_files.append(file)
    return list_of_files, list_of_files_with_path
