import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import plotly.express as px
import pandas as pd
from utils.file_utils import get_files_in_folder_2
import sys
import os
import json

import argparse


pd.set_option('display.max_columns', None)

MDL_DAT_PTH = 'model_outputs/'

PPILAR = 'pointpillar/'
POINTRCNN = 'point_rcnn/'
VOXEL = 'voxel/'
TED = 'ted/'

# TODO: change me when change model
CURR_MODEL = PPILAR

# baselines
BL = 'bl_1/'

# SPECIAL FSE2024 TESTS
ROUND0 = 'original_model_pretrained_velodyne_orig1/'
ROUND1 = 'original_model_pretrained_velodyne_000002_new1/'
ROUND2 = 'retrained_model_velodyne_orig1/'
ROUND3 = 'retrained_model_velodyne_000002_new1/'

# GLBL_RND = '11_1/'
GLBL_RND_2CM_UNIFORM = '51_1/'
GLBL_RND_2CM_NORMAL = '54_1/'
GLBL_RND_2CM_LAPLACE = '55_1/'

# LCL_RND = '16_1/'
LCL_RND_2CM_UNIFORM = '52_1/'
LCL_RND_2CM_NORMAL = '56_1/'
LCL_RND_2CM_LAPLACE = '58_1/'

SHIFT_RND = '21_1/'

DEC_60 = '49_1/'
INC_80 = '50_1/'

FALSE_POSITIVE = '53_1/'

# TODO: needs to combine results with 45, 46
RFL_4CM = '48_3/'

CHSN_RND = INC_80

# FBL = os.path.join(MDL_DAT_PTH, CURR_MODEL, BL)
FBL = os.path.join(MDL_DAT_PTH, CURR_MODEL, ROUND2)

# F_TARGET = os.path.join(MDL_DAT_PTH, CURR_MODEL, CHSN_RND)
F_TARGET = os.path.join(MDL_DAT_PTH, CURR_MODEL, ROUND3)

# F48 = os.path.join(MDL_DAT_PTH, PPILAR, RFL_4CM)
# F49 = os.path.join(MDL_DAT_PTH, PPILAR, DEC_60)
# F50 = os.path.join(MDL_DAT_PTH, PPILAR, INC_80)




def process_line(line):
    cleaned_line = line.strip().split()
    if len(cleaned_line) % 2 != 0:
        print('Error {}'.format(line))
        return []
    
    obs_cnt = int(len(cleaned_line) / 2)

    data = []
    for oid, gtid in enumerate(cleaned_line[:obs_cnt]):
        data.append((int(gtid), float(cleaned_line[oid + obs_cnt])))

    return data


def read_data_from_file(file):
    print('Reading from file {}'.format(file))
    data = []
    with open(file, 'r') as fd:
        for lidx, line in enumerate(fd.readlines()):
            # reformat to map each gt_id to iou, threshold not applied, exclude 0.0
            ious = process_line(line)
            data.append(ious)
    return data


def get_detection_frames(frames_baseline):
    _, files = get_files_in_folder_2(frames_baseline)

    files = sorted([file for file in files if 'detections.txt' not in file])
    frame_ids = []
    for file in files:
        f_id = file.split('/')[-1].split('_')[-1].split('.')[0]
        frame_ids.append(f_id)

    return files, frame_ids


def read_detection_json_ouput(dst, ious):
    _, files_in_folder = get_files_in_folder_2(dst)
    files_in_folder.sort()
    files_in_folder = [f for f in files_in_folder if 'detections.txt' not in f]

    # if int(gtid) in data:
    #     data[int(gtid)] = max(float(cleaned_line[oid + obs_cnt]), data[int(gtid)])
    # else:
    #     if float(cleaned_line[oid + obs_cnt]) >= 0.0:
    #         data[int(gtid)] = float(cleaned_line[oid + obs_cnt])
    data = []
    for file, obs_ious in zip(files_in_folder, ious):
        frame_id = file.split('/')[-1].split('.')[0].split('_')[-1]
        # print('Opening file {} --> frame {}'.format(file, frame_id))
        with open(file, 'r') as fd:
            frame = json.loads(fd.read())
            for obs, iou in zip(frame, obs_ious):
                obs['frame'] = frame_id
                obs['gtid'] = iou[0]
                obs['iou'] = iou[1]
                data.append(obs)

    df = pd.json_normalize(data=data)
    df[['x', 'y', 'z']] = pd.DataFrame(df['location'].tolist())
    df[['length', 'width', 'height']] = pd.DataFrame(df['dimensions'].tolist())
    df['size'] = df['length'] * df['width'] * df['height']
    df.drop(['occluded', 'location', 'truncated', 'bbox', 'dimensions', 'length', 'width', 'height'], axis=1, inplace=True)

    # remove gtid == 0 obstacles with ious == 0, only select max when ious > 0
    df = df.loc[df['iou'] > 0.25]
    idx = df.groupby(['frame', 'gtid'])['iou'].transform(max) == df['iou']
    df_filtered = df[idx]

    # code to print count
    # x = df_filtered.groupby(['frame', 'gtid'])['iou'].count()
    # print(x.loc[x > 1])

    return df_filtered


def compare_lines(orig_line, new_line):
    output = {}
    for k, v in new_line.items():
        if k not in orig_line:
            output[k] = v
        else:
            output[k] = v - orig_line[k]
            
    for k, v in orig_line.items():
        if k not in new_line:
            output[k] = -v

    return output


def plot_diff(data11, data49):
    # format iou diff for plot
    df11 = pd.DataFrame(data11, columns=['iou_diff'])
    df11['round'] = '11'
    df49 = pd.DataFrame(data49, columns=['iou_diff'])
    df49['round'] = '49'

    df = pd.concat([df11, df49])

    fig = px.box(df, y='iou_diff', 
            #  hover_data=[], 
             color='round', 
             title='Shift of detection box when obs points are dropped 60%')
    fig.show()


def plot_raw(raw_databl, raw_data11, raw_data49):
    # format raw iou for plot
    df_raw_databl = pd.DataFrame(raw_databl, columns=['iou'])
    df_raw_databl['round'] = 'baseline'

    df_raw_data11 = pd.DataFrame(raw_data11, columns=['iou'])
    df_raw_data11['round'] = '11'

    df_raw_data49 = pd.DataFrame(raw_data49, columns=['iou'])
    df_raw_data49['round'] = '49'
    df = pd.concat([df_raw_databl, df_raw_data11, df_raw_data49])

    fig = px.box(df, y='iou', 
            #  hover_data=[], 
             color='round', 
             title='Raw ious')
    fig.show()


def compute_iou_diffs(orig, target):
    data = []
    for o, t in zip(orig, target):
        # compute iou diff
        com_res = compare_lines(o, t)
        # save iou diff
        data.extend(list(com_res.values()))

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run options')
    parser.add_argument('--baseline', dest='baseline', type=str, help='Directory of baseline results')
    parser.add_argument('--perturbed', dest='perturbed', type=str, help='Directory of baseline results')
    args = parser.parse_args()

    # baseline = FBL
    baseline = args.baseline
    raw_files, frame_ids = get_detection_frames(baseline)

    # list of dicts
    raw_databl = read_data_from_file(os.path.join(baseline, 'detections.txt'))
    df_bl = read_detection_json_ouput(dst=baseline, ious=raw_databl)

    # detected obstacles in baseline    
    df_bl_detected = df_bl.loc[((df_bl['name'] == 'Car') & (df_bl['iou'] >= 0.7)) | (((df_bl['name'] == 'Pedestrian') | (df_bl['name'] == 'Cyclist')) & (df_bl['iou'] >= 0.5))]
    # print(df_bl_detected['gtid'].count(), '/', df_bl['gtid'].count())

    # raw_data11 = read_data_from_file(os.path.join(F11, 'detections.txt'))
    # df_11 = read_detection_json_ouput(dst=F11, ious=raw_data11)

    # raw_data49 = read_data_from_file(os.path.join(F49, 'detections.txt'))
    # df_11 = read_detection_json_ouput(dst=F49, ious=raw_data49)

    # target = F_TARGET
    target = args.perturbed
    raw_data = read_data_from_file(os.path.join(target, 'detections.txt'))
    df_target = read_detection_json_ouput(dst=target, ious=raw_data)

    # data11 = compute_iou_diffs(raw_databl, raw_data11)
    # data49 = compute_iou_diffs(raw_databl, raw_data49)

    # print(sum(data11) / len(data11), sum(data49) / len(data49))
    # plot_diff(data11, data49)

    # detected obstacles in target
    df_target_detected = df_target.loc[((df_target['name'] == 'Car') & (df_target['iou'] >= 0.7)) | (((df_target['name'] == 'Pedestrian') | (df_target['name'] == 'Cyclist')) & (df_target['iou'] >= 0.5))]

    print('Detection bl --> perturb', 
          df_bl_detected['gtid'].count(), '--->', 
          df_target_detected['gtid'].count(), '---->', df_bl['gtid'].count())
   
    # compute deviation by
    # 1. group the obs by frame and id
    # 2. subtract
    df_bl.set_index(['frame', 'gtid'], inplace=True)
    df_target.set_index(['frame', 'gtid'], inplace=True)

    df_bl_detected.set_index(['frame', 'gtid'], inplace=True)
    df_target_detected.set_index(['frame', 'gtid'], inplace=True)


    print('diff obs by gt', df_bl_detected.index.symmetric_difference(df_target_detected.index).shape)
    # print(f'avg ious: {df_bl.iou.mean()} {df_target.iou.mean()}')

    df_bl['diff_iou'] = df_target['iou'].subtract(df_bl['iou'], fill_value=0.0)

    df_bl['diff_center_x'] = df_target['x'].subtract(df_bl['x'])
    df_bl['diff_center_z'] = df_target['z'].subtract(df_bl['z'])
    df_bl['diff_center_y'] = df_target['y'].subtract(df_bl['y'])
    
    df_bl['diff_size'] = df_target['size'].subtract(df_bl['size'])

    print('REMEMBER THAT Z IS DEPTH AND Y IS HEIGHT IN THIS CASE')
    # Next we compute deviations for large values
    max_distance_perturb = 0.02
    thres = max_distance_perturb * 5
    print(f'Using threshold values = {thres} m')
    for curr_col in ['diff_center_x', 'diff_center_z', 'diff_center_y', 'diff_size', 'diff_iou']:
        if curr_col in ['diff_center_x', 'diff_center_z', 'diff_center_y']:
            df_calc = df_bl.loc[(df_bl[curr_col] <= -thres) | (df_bl[curr_col] >= thres)]
            print(f'{curr_col} --> abs median:{round(df_calc[curr_col].abs().median(), 2)} abs max:{round(df_calc[curr_col].abs().max(), 2)}')
        else:
            df_calc = df_bl.loc[
                ((df_bl['diff_center_x'] <= -thres) | (df_bl['diff_center_x'] >= thres)) |
                ((df_bl['diff_center_z'] <= -thres) | (df_bl['diff_center_z'] >= thres)) |
                ((df_bl['diff_center_y'] <= -thres) | (df_bl['diff_center_y'] >= thres))
            ]
            print(f"diff_size --> abs median:{round(df_calc['diff_size'].abs().median(), 2)} max {round(df_calc['diff_size'].abs().max(), 2)}")
            print(f"diff_iou --> abs median:{round(df_calc['diff_iou'].abs().median(), 2)} max {round(df_calc['diff_iou'].abs().max(), 2)}")
            print(f'count large values {df_calc[curr_col].count()}/{df_bl[curr_col].count()}={round(df_calc[curr_col].count()/df_bl[curr_col].count() * 100, 2)}')
            break

    df_plot_iou = df_bl.loc[(df_bl['diff_iou'] >= 0.5) | (df_bl['diff_iou'] <= -0.5)]
    df_plot_x = df_bl.loc[(df_bl['diff_center_x'] >= 0.5) | (df_bl['diff_center_x'] <= -0.5)]
    df_plot_z = df_bl.loc[(df_bl['diff_center_z'] >= 0.5) | (df_bl['diff_center_z'] <= -0.5)]

    fig, axes = plt.subplots(1, 3, sharex=True, sharey=True)
    ego_x, ego_y = 0.0, 0.0

    pcm = axes[0].scatter(x=df_plot_x['x'], y=df_plot_x['z'], c=df_plot_x['diff_center_x'], cmap='seismic', vmin=-1.5, vmax=1.5)
    axes[0].set_title('Deviation in x direction')
    fig.colorbar(pcm, ax=axes[0], extend='max')
    axes[0].add_patch(Rectangle((ego_x - 1, ego_y - 4), 2, 8, alpha=1, facecolor='black'))

    # z COORDINATES IS Y IN GRAPH!!!
    pcm = axes[1].scatter(x=df_plot_z['x'], y=df_plot_z['z'], c=df_plot_z['diff_center_z'], cmap='seismic', vmin=-1.5, vmax=1.5)
    axes[1].set_title('Deviation in y direction')
    fig.colorbar(pcm, ax=axes[1], extend='max')
    axes[1].add_patch(Rectangle((ego_x - 1, ego_y - 4), 2, 8, alpha=1, facecolor='black'))

    pcm = axes[2].scatter(x=df_plot_iou['x'], y=df_plot_iou['z'], c=df_plot_iou['diff_iou'], cmap='seismic', vmin=-1, vmax=1)
    axes[2].set_title('Deviation in iou')
    fig.colorbar(pcm, ax=axes[2], extend='max')
    axes[2].add_patch(Rectangle((ego_x - 1, ego_y - 4), 2, 8, alpha=1, facecolor='black'))

    fig.supylabel('Distance y')
    fig.supxlabel('Distance x')
    plt.show()
