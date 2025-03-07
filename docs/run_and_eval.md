# BEVHeight Experiments
## Train and Test
Train BEVHeight with 8 GPUs
```
python [EXP_PATH] --amp_backend native -b 8 --gpus 8
```
Eval BEVHeight with 8 GPUs
```
python [EXP_PATH] --ckpt_path [CKPT_PATH] -e -b 8 --gpus 8
```

# SGV3D Experiments

## 1. Train & Eval Teacher Model.
```
python exps/sgv3d/bsm_bev_height_lss_r50_864_1536_128x128.py --amp_backend native -b 2 --gpus 8
python exps/sgv3d/bsm_bev_height_lss_r50_864_1536_128x128.py  --ckpt outputs/bsm_bev_height_lss_r50_864_1536_128x128/checkpoints/ -e -b 2 --gpus 8
```
## 2. Generate Pseudo Labels for Unlabeled Data.

```
python exps/sgv3d/bsm_bev_height_lss_r50_864_1536_128x128.py  --ckpt outputs/bsm_bev_height_lss_r50_864_1536_128x128/checkpoints/last.ckpt. -e -b 2 --gpus 8 --val_info_path data/rope3d-kitti/rope3d_12hz_infos_unlabeled_data.pkl
```

## 3. Perform Semi-supervised Data Generation Pipeline (SSDG).   TODO
```
python scripts/data_preprocess/recombine_strategy.py --src-root data/rope3d-kitti --dest-root data/rope3d-kitti-gen
```

## 4. Generate `**_12hz_infos_*.pkl`.
```
python scripts/gen_info_rope3d_kitti.py --data-root data/rope3d-kitti-gen
```

The directory will be as follows.
```
SGV3D
├── data
│   ├── dair-v2x-i
│   │   ├── velodyne
│   │   ├── image
│   │   ├── calib
│   │   ├── label
|   |   └── data_info.json
|   └── dair-v2x-i-kitti
|   |   ├── training
|   |   |   ├── calib
|   |   |   ├── label_2
|   |   |   └── images_2
|   |   └── ImageSets
|   |        ├── train.txt
|   |        └── val.txt
|   ├── rope3d
|   |   ├── training
|   |   ├── validation
|   |   ├── training-image_2a
|   |   ├── training-image_2b
|   |   ├── training-image_2c
|   |   ├── training-image_2d
|   |   └── validation-image_2
|   |   ├── ImageSets
|   |   |    ├── train_dair.txt     # DAIR-V2X-I train split in Rope3D
|   |   |    ├── val_dair.txt       # DAIR-V2X-I val split in Rope3D
|   |   |    ├── train.txt          # Rope3D train split
|   |   |    └── val.txt            # Rope3D val split
|   |   ├── dair_12hz_infos_train.pkl
|   |   ├── dair_12hz_infos_val.pkl
|   |   ├── dair_12hz_infos_unlabeled_data.pkl
|   ├── rope3d-kitti
|   |   ├── training
|   |   |   ├── calib
|   |   |   ├── denorm
|   |   |   ├── label_2
|   |   |   ├── mask_image          # multi-class foreground mask
|   |   |   └── images_2
|   |   ├── ImageSets
|   |   |    ├── train_dair.txt     # DAIR-V2X-I train split in Rope3D
|   |   |    ├── val_dair.txt       # DAIR-V2X-I val split in Rope3D
|   |   |    ├── train.txt          # Rope3D val split
|   |   |    ├── val.txt            # Rope3D val split
|   |   |    └── unlabeled_data.txt # unlabeled data for DAIR-V2X-I from Rope3D
|   |   └── map_token2id.json
|   |   ├── rope3d_12hz_infos_train.pkl
|   |   ├── rope3d_12hz_infos_val.pkl
|   |   ├── rope3d_12hz_infos_unlabeled_data.pkl
├── ├── rope3d-kitti-gen
|   |   ├── training
|   |   |   ├── calib
|   |   |   ├── denorm
|   |   |   ├── label_2
|   |   |   ├── label_eval
|   |   |   ├── mask_image          # multi-class foreground mask
|   |   |   └── images_2
|   |   ├── ImageSets
|   |   |    ├── train_dair.txt
|   |   |    ├── val_dair.txt
|   |   |    └── train_ssdg.txt
|   |   ├── map_token2id.json
|   |   ├── rope3d_12hz_infos_train_dair.pkl
|   |   ├── rope3d_12hz_infos_val_dair.pkl
|   |   ├── rope3d_12hz_infos_train_ssdg.pkl
```

## 5. Train & Eval Student Model.
```
python exps/sgv3d/bsm_bev_height_lss_r50_864_1536_128x128.py --amp_backend native -b 2 --gpus 8 --data_root data/rope3d-kitti-gen
python exps/sgv3d/bsm_bev_height_lss_r50_864_1536_128x128.py  --ckpt outputs/bsm_bev_height_lss_r50_864_1536_128x128/checkpoints/ -e -b 2 --gpus 8 --data_root data/rope3d-kitti-gen
```

## Go To Step. 2