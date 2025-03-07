# DAIR-V2X-I  Rope3D
Download DAIR-V2X-I or Rope3D dataset from official [website](https://thudair.baai.ac.cn/index).

## 1. Symlink the dataset root to `./data/`.
```
ln -s [dair-v2x/single-infrastructure-side root] ./data/dair-v2x
ln -s [rope3d root] ./data/rope3d
```

## 2. Convert DAIR-V2X-I or Rope3D to KITTI format.
```
python scripts/data_converter/dair2kitti.py --source-root data/dair-v2x-i --target-root data/dair-v2x-i-kitti
python scripts/data_converter/rope2kitti.py --source-root data/rope3d --target-root data/rope3d-kitti
```

## 3. Generate unlabeled_data.txt for DAIR-V2X-I from Rope3D.
```
python scripts/data_preprocess/gen_unlabeled_split.py
```

## 4. Generate Foreground Mask
```
python scripts/data_preprocess/recombine_strategy.py --src-root data/rope3d-kitti --dest-root data/rope3d-kitti
```

## 5. Generate `**_12hz_infos_*.pkl`
```
python scripts/gen_info_dair.py
python scripts/gen_info_rope3d.py
python scripts/gen_info_rope3d_kitti.py --data-root data/rope3d-kitti
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
|   |   ├── rope3d_12hz_infos_train_ssdg.pkl
```

## 6. Visualize the dataset in KITTI format
```
python scripts/data_converter/visual_tools.py --data_root data/rope3d-kitti --demo_dir ./demo
```