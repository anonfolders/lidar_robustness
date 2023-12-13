import numpy as np


def get_min_max(pcd):
    # check min max each coordinates
    # print('size={}, x=[{}, {}], y=[{}, {}], z=[{}, {}]'.format(
    #     len(pcd),
    #     min(pcd[0::4]), max(pcd[0::4]),
    #     min(pcd[1::4]), max(pcd[1::4]),
    #     min(pcd[2::4]), max(pcd[2::4])
    # ))
    return min(pcd[0::4]), max(pcd[0::4]), min(pcd[1::4]), max(pcd[1::4]), min(pcd[2::4]), max(pcd[2::4])


def read_pc_bin(filename):
    # Load binary point cloud
    bin_pcd = np.fromfile(filename, dtype=np.float32)
    return bin_pcd


def write_pc_bin(pcd: np.ndarray, filename):
    print(f'Writing {filename}')
    pcd.tofile(filename)


def reshape_pcd(pcd):
    return pcd.reshape((-1, 4))


def save_pc_to_csv(pcd: np.ndarray, filename: str):
    tmp = reshape_pcd(pcd)
    tmp = np.delete(tmp, 3, axis=1)
    np.savetxt(filename, tmp, delimiter=',')
