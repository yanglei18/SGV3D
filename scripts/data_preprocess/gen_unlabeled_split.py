import os
import json
import argparse
import csv

import numpy as np
from tqdm import tqdm

def parse_option():
    parser = argparse.ArgumentParser('SGV3D data preprocess', add_help=False)
    parser.add_argument('--src-root', type=str, default="data/rope3d-kitti", help='root path to src rope3d-kitti dataset')
    parser.add_argument('--dest-root', type=str, default="data/rope3d-kitti-gen", help='root path to result rope3d-kitti dataset')
    parser.add_argument('--vis', type=bool, default=False, help='flag to control visualization')
    args = parser.parse_args()
    return args

def read_split(split_txt):
    with open(split_txt, "r") as file:
        lines = file.readlines()
    split_list = list()
    for line in lines:
        split_list.append(line.rstrip('\n'))
    return split_list

def write_split(split_list, split_txt):
    wf = open(split_txt, "w")
    for line in split_list:
        line_string = line + "\n"
        wf.write(line_string)
    wf.close()

def is_exists(data_root, subset, frame_id):
    img_file = os.path.join(data_root, subset, "mask_image", frame_id + ".jpg")
    return True if os.path.exists(img_file) else False

def load_calib_dair(calib_file):
    with open(os.path.join(calib_file), 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=' ')
        for line, row in enumerate(reader):
            if row[0] == 'P2:':
                P2 = np.array([float(i) for i in row[1:]], dtype=np.float32).reshape(3, 4)
                continue
            elif row[0] == 'Tr_velo_to_cam:':
                Tr_ego2cam = np.array([float(i) for i in row[1:]], dtype=np.float32).reshape(3, 4)
    Tr_ego2cam = np.concatenate((Tr_ego2cam, np.array([[0, 0, 0, 1]])), axis=0)
    return Tr_ego2cam, P2

def read_calib(data_root, subset, frame_id, is_pred=False, img_dest_path=None):
    calib_file = os.path.join(data_root, subset, "calib", frame_id + ".txt")
    Tr_ego2cam, P2 = load_calib_dair(calib_file) # 得到相机外参Tr_cam2ego
    Tr_cam2ego = np.linalg.inv(Tr_ego2cam)
    Tr_ego2cam = np.linalg.inv(Tr_cam2ego)
    return {"Tr_ego2cam": Tr_ego2cam, "P2": P2} 

def gen_scene_P2(src_root, frame_ids):
    P2_list = []
    for frame_id in frame_ids:
        sample_info = read_calib(src_root, "training", frame_id)
        if sample_info['P2'][0,0] not in P2_list:
            P2_list.append(sample_info['P2'][0,0])
    return P2_list

if __name__ == "__main__":
    args = parse_option()
    split_root = 'data/rope3d-kitti/ImageSets'
    src_root, dest_root, vis = args.src_root, args.dest_root, args.vis
    token2sample_txt = os.path.join(split_root, "../map_token2id.json")
    train_frame_ids = read_split(os.path.join(split_root, "train_dair.txt"))
    val_frame_ids = read_split(os.path.join(split_root, "val_dair.txt"))
    print(len(train_frame_ids), len(val_frame_ids))
    raw_frame_ids = read_split(os.path.join(src_root, "ImageSets/train.txt")) +  read_split(os.path.join(src_root, "ImageSets/val.txt"))
    
    train_P2_list = gen_scene_P2(src_root, train_frame_ids)
    val_P2_list = gen_scene_P2(src_root, val_frame_ids)
    trainval_P2_list = train_P2_list + val_P2_list

    unlabeled_data_list = list()
    for frame_id in tqdm(raw_frame_ids):
        sample_info= read_calib(src_root, "training", frame_id)
        if sample_info['P2'][0,0] in trainval_P2_list: continue
        unlabeled_data_list.append(frame_id)
    print("unlabeled_data_list: ", len(unlabeled_data_list))
    write_split(unlabeled_data_list, os.path.join(src_root, "ImageSets", "unlabeled_data.txt"))
