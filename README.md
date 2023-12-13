# Repository for *The Robustness of LiDAR 3D Obstacle Detection and Its Impacts on Autonomous Driving*

This folder contains the following sub folders

## Documentations (docs)

Contains the manuals/specifications for some LiDAR sensors:
- [Velodyne HDL-64E](docs/HDL64E.pdf)
- [Ouster OS2](docs/OUSTER-OS2-RevD-V2p0.pdf)

## Data

Contains data presented in the paper

## Source code (src)

Includes the code developed for the experiment. Found in `src/`.

## Instructions for RQ1

To replicate results for RQ1.

1. Download the Velodyne point cloud files KITTI's [3D Object Detection data set](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) (needs to register an account).

2. Copy the point cloud (`.bin`) files in `KITTI/training/velodyne/` (downloaded in step 1) to populate the `src/data/baseline/velodyne/` folder.

3. Run the following commands to generate perturbations which will generate one perturbation of specified type for each frame. The output is in `src/data/baseline/velodyne/<count>/velodyne/` where `<count>` would increase each run from `000000-999999`.
```bash
cd src/ 
python main.py --count 1 --type global_uniform
```
4. Download our [modified OpenPCDet](https://github.com/anonfolders/OpenPCDet-SORBET/) (OpenPCDet-SORBET)

5. Then, follow instructions from [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) to set up the OpenPCDet-SORBET library.
    - Follow installation steps in [INSTALL](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/INSTALL.md) except that for step `a.`, we have already downloaded OpenPCDet-SORBET in step 4.
    - Follow [GETTING STARTED](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md) to set up the KITTI data set.

    - The following models (from the [model zoo](https://github.com/open-mmlab/OpenPCDet/tree/master#model-zoo)) are used in our paper which can be just copied to the `OpenPCDet-SORBET/checkpoints` for evaluation:

| Architecture | Year | Model |
|---|---|---|
| Point Pillar | 2019 | [link](data/RQ1/models/pointpillar_7728.pth) |
| Point RCNN | 2019 | [link](data/RQ1/models/pointrcnn_7870.pth) |
| Voxel RCNN | 2020 | [link](data/RQ1/models/voxel_rcnn_car_84.54.pth) |

6. In `OpenPCDet-SORBET/data/kitti/training/`, rename `OpenPCDet-SORBET/data/kitti/training/velodyne` to `OpenPCDet-SORBET/data/kitti/training/velodyne_orig`

6. Organize the data as follows: copy the chosen `<count>/velodyne/`folder to `data/kitti/training/` after finish setting up the KITTI data set as described in [OpenPCDet GETTING STARTED](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md) and rename it to `OpenPCDet-SORBET/data/kitti/training/velodyne_<count>`

7. In this step, we would run the corresponding model twice. One time on the orignal data to obtain the baseline, and one time on the perturbed data. **In `OpenPCDet-SORBET/`** runs:
```bash
cd tools/
# for the baseline
bash run_model_test.sh ../data/kitti/training/velodyne_<count> <model_config> <model>
# for the perturbed data
bash run_model_test.sh <dir_to_velodyne> <model_config> <model>
```
where
- `<dir_to_velodyne>`: is the velodyne data generated created in step 6 `OpenPCDet-SORBET/data/kitti/training/velodyne_<count>`
- `<model_config>`: config of each model architecture, can be found in `OpenPCDet-SORBET/tools/cfgs/kitti_models/`
- `<model>`: the models located in `OpenPCDet-SORBET/checkpoints` as described in step 5

Sample command:
```bash
bash run_model_test.sh ../data/kitti/training/velodyne_000001 cfgs/kitti_models/pointpillar.yaml ../checkpoints/pointpillar_7728.pth
```
Once done, locate the output in `OpenPCDet-SORBET/tools/SORBET_output/model_output`. A sample output called `231213-143349` is included for reference.

8. Move the output, e.g., `231213-143349` to `lidar_robustness\data\RQ1\model_outputs`.

9. To visualize the results, run
```bash
python visualize_model_detection_output.py --baseline <baseline_results> --perturbed <perturbed_results>
```
where

- `<baseline_results>` is the detection results of the model on the original KITTI data set

- `<perturbed_results>` is the detection results of the model on the perturbed KITTI data set

<!-- [TODO] Instructions to set up and run individual models in Apollo.

1. Check out Apollo source code from (https://github.com/ApolloAuto/apollo).

2. In folder [TODO], add the following code to `setup` -->

## Instructions for RQ2

1. Setup [Trajectron++](https://github.com/StanfordASL/Trajectron-plus-plus)

2. Copy [notebook](src/RQ2/run_rq2_experiment.ipynb) to `Trajectron-plus-plus/experiments/nuScenes/`

3. Run notebook cell-by-cell

