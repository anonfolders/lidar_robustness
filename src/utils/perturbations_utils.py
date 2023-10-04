import numpy as np
from copy import deepcopy
import random

from utils.config import LocalDirection as LD
from utils.config import PerturbationMode as PM
# MAX_PERTURBATION_SIZE = 0.005 (meters)
# MAX_PERTURBATION_SIZE = 0.002
MAX_PERTURBATION_SIZE = 0.02


def add_random_perturbations_global(pcd):
    pcd_s = np.reshape(pcd, (-1, 4))
    random_pertubations = np.random.randint(low=-4, high=4 + 1, size=(pcd_s.shape[0], 3)) * 0.005
    pcd_s[:, :-1] = (pcd_s[:, :-1] + random_pertubations).astype(np.float32)
    return np.reshape(pcd_s, pcd.shape)


def add_random_perturbations_global_max(pcd, max_pertubation=MAX_PERTURBATION_SIZE):
    pcd_s = np.reshape(pcd, (-1, 4))
    random_pertubations = np.random.choice([-max_pertubation, max_pertubation], size=(pcd_s.shape[0], 3))
    pcd_s[:, :-1] = (pcd_s[:, :-1] + random_pertubations).astype(np.float32)
    return np.reshape(pcd_s, pcd.shape)


def rand_uniform_helper(pcd_s, mode):
    val_size = pcd_s.shape[0]

    val_range = MAX_PERTURBATION_SIZE
    val_range_square = val_range * val_range

    col_1 = np.random.uniform(low=-1 * val_range, high=val_range, size=val_size)
    col_1_square = col_1 * col_1

    range_col_2 = np.sqrt(val_range_square - col_1_square)

    col_2 = np.random.uniform(low=-1 * range_col_2, high=range_col_2, size=val_size)
    col_2_square = col_2 * col_2

    # compute the range of final col
    range_col_3 = np.sqrt(val_range_square - (col_1_square + col_2_square))

    if mode == 'max':
        # however, we want the largest possible deviation, thus
        col_0 = range_col_3 * np.random.choice([-1, 1], size=range_col_3.size)
        # print(col_1_square + col_2_square + col_0 * col_0)

        output = np.stack((col_0, col_1, col_2, np.zeros(col_0.size)), axis=1)
    # elif mode == 'rand':
    #     # if pure random
    #     col_3 = np.random.uniform(low=-1 * range_col_3, high=range_col_3, size=val_size)
    #     col_3_square = col_3 * col_3

    #     # print(col_1_square + col_2_square + col_3_square)
    #     output = np.stack((col_1, col_2, col_3, np.zeros(col_1.size)), axis=1)
    return output


def rand_uniform_perturbs_global(pcd, mode='max'):
    pcd_s = np.reshape(pcd, (-1, 4))

    output = rand_uniform_helper(pcd_s=pcd_s, mode=mode)

    pcd_s = (pcd_s + output).astype(np.float32)
    return np.reshape(pcd_s, pcd.shape)


def random_helper(pcd_s, mode):
    d_sq = MAX_PERTURBATION_SIZE ** 2

    if mode == 'normal':
        perturbs = np.random.normal(size=(pcd_s.shape[0], 3))
    elif mode == 'laplace':
        perturbs = np.random.laplace(size=(pcd_s.shape[0], 3))

    xyz_sum_sq = np.sum(perturbs ** 2, axis=1)

    ratio = np.sqrt(xyz_sum_sq / d_sq)
    # reshape ratio from row to colum
    ratio = np.reshape(ratio, (ratio.shape[0], 1))

    output = np.zeros(pcd_s.shape)
    # set all output columns except intensity to be perturbations (scaled)
    output[:,:-1] = perturbs / ratio

    # sum_of_squares = np.sum(output ** 2, axis=1)
    # has 1mm rounding error
    # print((sum_of_squares - d_sq <= 0.000001).sum())
    return output


def random_perturbs(pcd, mode):
    pcd_s = np.reshape(pcd, (-1, 4))

    perturbs = random_helper(pcd_s, mode=mode)

    pcd_s = (pcd_s + perturbs).astype(np.float32)
    return np.reshape(pcd_s, pcd.shape)


def random_perturbs_local(pcd, mode, points_in_obs):
    pcd_s = np.reshape(pcd, (-1, 4))

    perturbs = random_helper(pcd_s, mode=mode)

    # only take points unmasked
    perturbs = perturbs * points_in_obs

    pcd_s = (pcd_s + perturbs).astype(np.float32)
    return np.reshape(pcd_s, pcd.shape)


def rand_uniform_perturbs_local(pcd, points_in_obs, mode='max'):
    pcd_s = np.reshape(pcd, (-1, 4))

    output = rand_uniform_helper(pcd_s=pcd_s, mode=mode)

    # only take points unmasked
    output = output * points_in_obs

    pcd_s = (pcd_s + output).astype(np.float32)
    return np.reshape(pcd_s, pcd.shape)


def add_random_perturbations_local(pcd, points_in_obs):
    random_pertubations = np.random.randint(low=-4, high=4 + 1, size=pcd.shape) * 0.005
    random_pertubations_local = random_pertubations * points_in_obs

    pcd_after_perturbation = (pcd + random_pertubations_local).astype(np.float32)

    return pcd_after_perturbation


def add_random_perturbations_local_max(pcd, points_in_obs, max_pertubation=MAX_PERTURBATION_SIZE):
    random_pertubations = np.random.choice([-max_pertubation, max_pertubation], size=pcd.shape)
    random_pertubations = np.reshape(random_pertubations, (-1, 4))
    random_pertubations_local = random_pertubations * points_in_obs
    random_pertubations_local = random_pertubations_local.flatten()

    pcd_after_perturbation = (pcd + random_pertubations_local).astype(np.float32)

    return pcd_after_perturbation


def add_directional_perturbations_local(pcd, points_in_obs, direction, mode='uniform'):
    pcd_tmp = deepcopy(pcd)
    # generate shift
    if mode == 'uniform':
        directed_perturbations = np.random.randint(low=1, high=4 + 1, size=pcd_tmp.shape[0]) * 0.005
    elif mode == 'normal':
        directed_perturbations = np.random.normal(loc=0, scale=0.33, size=pcd_tmp.shape[0])
    elif mode == 'laplace':
        directed_perturbations = np.random.laplace(loc=0, scale=0.33, size=pcd_tmp.shape[0])

    # mask shift where there are no obstacles
    directed_perturbations = directed_perturbations * points_in_obs[:, 0]

    directed_perturbations = np.abs(np.array(directed_perturbations, dtype=np.float32))

    if direction is LD.X_PLUS:
        pcd_tmp[:, 0] += directed_perturbations
    elif direction is LD.X_MINUS:
        pcd_tmp[:, 0] -= directed_perturbations
    elif direction is LD.Y_PLUS:
        pcd_tmp[:, 1] += directed_perturbations
    elif direction is LD.Y_MINUS:
        pcd_tmp[:, 1] -= directed_perturbations
    elif direction is LD.Z_PLUS:
        pcd_tmp[:, 2] += directed_perturbations
    elif direction is LD.Z_MINUS:
        pcd_tmp[:, 2] -= directed_perturbations

    pcd_tmp = pcd_tmp.astype(np.float32)
    return pcd_tmp.flatten()


def add_directional_perturbations_local_max(pcd, points_in_obs, direction):
    # generate shift
    directed_perturbations = np.zeros(pcd.shape)
    
    if direction is LD.X_PLUS:
        directed_perturbations[:, 0] += MAX_PERTURBATION_SIZE
    elif direction is LD.X_MINUS:
        directed_perturbations[:, 0] -= MAX_PERTURBATION_SIZE
    elif direction is LD.Y_PLUS:
        directed_perturbations[:, 1] += MAX_PERTURBATION_SIZE
    elif direction is LD.Y_MINUS:
        directed_perturbations[:, 1] -= MAX_PERTURBATION_SIZE
    elif direction is LD.Z_PLUS:
        directed_perturbations[:, 2] += MAX_PERTURBATION_SIZE
    elif direction is LD.Z_MINUS:
        directed_perturbations[:, 2] -= MAX_PERTURBATION_SIZE

    # mask shift where there are no obstacles
    directed_perturbations_local = directed_perturbations * points_in_obs

    pcd_after_perturbation = (pcd + directed_perturbations_local).astype(np.float32)
    return pcd_after_perturbation.flatten()


def local_fp(pcd, points_in_obs):
    # remove one point from obstacles
    target_idx = np.random.choice(np.where(np.all(points_in_obs == [1, 1, 1, 0], axis=1))[0])
    pcd_after_perturbation = np.delete(pcd, [target_idx], axis=0)

    print('Reduce shape from {} to {}'.format(pcd.shape, pcd_after_perturbation.shape))
    return pcd_after_perturbation.flatten()


def global_fp(pcd):
    # remove one point from obstacles
    target_idx = np.random.choice(pcd.shape[0], 1, replace=False)
    pcd_after_perturbation = np.delete(pcd, [target_idx], axis=0)

    print('Reduce shape from {} to {}'.format(pcd.shape, pcd_after_perturbation.shape))
    return pcd_after_perturbation.flatten()


def reflect_perturbs_down(pcd, points_in_obs, color=PM.REFLECT_DOWNSAMPLE):
    # multiply pcd with points in obs
    # mask points to remove in obstacles
    pcd_after_perturbation = (pcd * points_in_obs).astype(np.float32)

    # create mask to remove [0, 0, 0, 0] rows
    row_mask = (pcd_after_perturbation != 0).any(axis=1)
    pcd_after_perturbation = pcd_after_perturbation[row_mask, :]

    print('Reduce shape from {} to {}'.format(pcd.shape, pcd_after_perturbation.shape))
    return pcd_after_perturbation.flatten()


def reflect_perturbs_up(pcd, points_in_obs):
    # generate new points
    # range -0.02 <= perturbation <= 0.02
    random_pertubations = np.random.randint(low=-4, high=4 + 1, size=pcd.shape) * 0.005
    new_pcd_with_perturbations = (pcd + random_pertubations).astype(np.float32)
    new_pcd_with_perturbations = np.reshape(new_pcd_with_perturbations, (-1, 4))
    new_pcd_with_perturbations = new_pcd_with_perturbations * points_in_obs
    # create mask to remove [0, 0, 0, 0] rows
    row_mask = (new_pcd_with_perturbations != 0).any(axis=1)
    new_pcd_with_perturbations = new_pcd_with_perturbations[row_mask, :]


    pcd_after_perturbation = np.append(pcd, new_pcd_with_perturbations).astype(np.float32)
    
    print('Increase shape from {} to {}'.format(pcd.shape, pcd_after_perturbation.shape))

    return pcd_after_perturbation


def add_reflective_perturbations(pcd, points_in_obs):
    # generate shift
    # range -0.02 <= perturbation <= 0.02
    random_pertubations = np.random.choice([-0.05, 0.05], size=pcd.shape)
    random_pertubations_local = random_pertubations * points_in_obs

    pcd_after_perturbation = np.append(pcd, random_pertubations_local).astype(np.float32)

    return pcd_after_perturbation
