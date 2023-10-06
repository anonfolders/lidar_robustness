# Repository for *The Robustness of LiDAR 3D Obstacle Detection and Its Impacts on Autonomous Driving*

This folder contains the following sub folders

## Documentations (docs)

Contains the manuals/specifications for some LiDAR sensors:
- Velodyne HDL-64E
- Ouster OS2

## Data

Contains data presented in the paper

## Source code (src)

Includes the code developed for the experiment.
- [TODO] clean personal details before upload

## Instructions for RQ1

To replicate results for RQ1.

1. Download the Velodyne point cloud files KITTI's 3D Object Detection data set (https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).

2. Copy the contents of the files to `data/baseline/velodyne/` populate the `src/data/` folder.

3. Run the following commands to generate perturbations which will
```bash
cd src/ 
python main.py --count 1 --type global_uniform
```

4. Then, follow instructions from OpenPCDet to download and set up the pre-trained models (https://github.com/open-mmlab/OpenPCDet).

5. [TODO] Add the following lines to the openpcdet evaluation code to save the raw 3D detection information.

6. Run chosen model on files generated in `src/data/`.
[TODO] provide examples.

7. [TODO] Run following code to visualize the results.

[TODO] Instructions to set up and run individual models in Apollo.

1. Check out Apollo source code from (https://github.com/ApolloAuto/apollo).

## Instructions for RQ2