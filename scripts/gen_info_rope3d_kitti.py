import os
import math
import json
import random
import argparse
import mmcv

import numpy as np
from pyquaternion import Quaternion
from tqdm import tqdm

from scripts.data_converter.visual_utils import *
from scripts.data_preprocess.recombine_utils import get_denorm, compute_box_3d_camera_v2
from scripts.data_converter.rope2kitti import get_cam2velo

name2nuscenceclass = {
    "car": "vehicle.car",
    "van": "vehicle.car",
    "truck": "vehicle.truck",
    "bus": "vehicle.bus.rigid",
    "cyclist": "vehicle.bicycle",
    "bicycle": "vehicle.bicycle",
    "tricyclist": "vehicle.bicycle",
    "motorcycle": "vehicle.bicycle",
    "motorcyclist": "vehicle.bicycle",
    "barrowlist": "vehicle.bicycle",
    "barrow": "vehicle.bicycle",
    "pedestrian": "human.pedestrian.adult",
    "traffic_cone": "movable_object.trafficcone",
}

def alpha2roty(alpha, pos):
    ry = alpha + np.arctan2(pos[0], pos[2])
    if ry > np.pi:
        ry -= 2 * np.pi
    if ry < -np.pi:
        ry += 2 * np.pi
    return ry

def clip2pi(ry):
    if ry > 2 * np.pi:
        ry -= 2 * np.pi
    if ry < - 2* np.pi:
        ry += 2 * np.pi
    return ry

def get_bbox(dim, location, rotation_y, denorm, P2):
    corners_3d = compute_box_3d_camera_v2(dim, location, rotation_y, denorm)
    box_2d = project_to_image(corners_3d, P2)
    xmin, ymin, xmax, ymax = np.min(box_2d[:, 0]), np.min(box_2d[:, 1]), np.max(box_2d[:, 0]), np.max(box_2d[:, 1])
    xmin, ymin = max(0, xmin), max(0, ymin)
    xmax, ymax = min(1919, xmax), min(1079, ymax)
    return [xmin, ymin, xmax, ymax]

def get_annos(label_path, Tr_cam2lidar, P2):
    if not os.path.exists(label_path):
        return None, None
    fieldnames = ['type', 'truncated', 'occluded', 'alpha', 'xmin', 'ymin', 'xmax', 'ymax', 'dh', 'dw',
                      'dl', 'lx', 'ly', 'lz', 'ry']
    gt_names = []
    gt_boxes, gt_2dboxes = [], []
    with open(label_path, 'r') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=' ', fieldnames=fieldnames)
        for line, row in enumerate(reader):
            if row["type"].lower() in name2nuscenceclass.keys():
                alpha = float(row["alpha"])
                pos = np.array((float(row['lx']), float(row['ly']), float(row['lz'])), dtype=np.float32)
                ry = float(row["ry"])
                if alpha > np.pi:
                    alpha -= 2 * np.pi
                    ry = alpha2roty(alpha, pos)
                alpha = clip2pi(alpha)
                ry = clip2pi(ry)
                rotation =  0.5 * np.pi - ry
                
                dim = [float(row['dl']), float(row['dw']), float(row['dh'])]
                if sum(dim) == 0:
                    continue
                loc_cam = np.array([float(row['lx']), float(row['ly']), float(row['lz']), 1.0]).reshape(4, 1)
                
                loc_lidar = np.matmul(Tr_cam2lidar, loc_cam).squeeze(-1)[:3]
                loc_lidar[2] += 0.5 * float(row['dh'])
 
                x, y, z = loc_lidar[0], loc_lidar[1], loc_lidar[2]
                h, w, l = float(row['dh']), float(row['dw']), float(row['dl']) 
                
                Tr_lidar2cam = np.linalg.inv(Tr_cam2lidar)
                denorm = get_denorm(Tr_lidar2cam)
                xmin, ymin, xmax, ymax = float(row['xmin']), float(row['ymin']), float(row['xmax']), float(row['ymax'])
                bbox = get_bbox(np.array([float(row['dh']), float(row['dw']), float(row['dl'])]), np.array([float(row['lx']), float(row['ly']), float(row['lz'])]), ry, denorm, P2)
                xmin, ymin, xmax, ymax = bbox
                lidar_yaw = rotation
                gt_names.append(row["type"].lower())
                gt_boxes.append([x, y, z, l, w, h, lidar_yaw])
                gt_2dboxes.append([xmin, ymin, xmax, ymax])
    gt_boxes = np.array(gt_boxes)
    gt_2dboxes = np.array(gt_2dboxes)

    return gt_names, gt_boxes, gt_2dboxes

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

def load_data(dair_root, sample_id, load_gt):
    label_file = os.path.join(dair_root, "training/label_2", sample_id + ".txt")
    calib_file = os.path.join(dair_root, "training/calib", sample_id + ".txt")
    Tr_velo2cam, P2 = load_calib_dair(calib_file)
    P = P2[:3, :3]
    r_velo2cam, t_velo2cam = Tr_velo2cam[:3,:3], Tr_velo2cam[:3,3]
    Tr_cam2velo = np.linalg.inv(Tr_velo2cam)
    if load_gt:
        gt_names, gt_boxes, gt_2dboxes = get_annos(label_file, Tr_cam2velo, P2)
    else:
        gt_names, gt_boxes, gt_2dboxes = None, None, None

    return r_velo2cam, t_velo2cam, P, gt_names, gt_boxes, gt_2dboxes

def cam2velo(r_velo2cam, t_velo2cam):
    Tr_velo2cam = np.eye(4)
    Tr_velo2cam[:3, :3] = r_velo2cam
    Tr_velo2cam[:3 ,3] = t_velo2cam.flatten()
    Tr_cam2velo = np.linalg.inv(Tr_velo2cam)
    r_cam2velo = Tr_cam2velo[:3, :3]
    t_cam2velo = Tr_cam2velo[:3, 3]
    return r_cam2velo, t_cam2velo
    
def load_denorm(denorm_file):
    with open(denorm_file, 'r') as f:
        lines = f.readlines()
    denorm = np.array([float(item) for item in lines[0].split(' ')])
    return denorm

def generate_info_rope3d_kitti(dair_root, split, load_gt=True):   
    if split == 'unlabeled_data':
        split_list = [x.strip() for x in open(os.path.join(dair_root, "ImageSets", "unlabeled_data.txt")).readlines()]
        split_list = list(set(split_list))
    else:
        split_list = [x.strip() for x in open(os.path.join(dair_root, "ImageSets", split + ".txt")).readlines()]

    with open(os.path.join(dair_root, "map_token2id.json"), 'r') as fp:
        token2sample = json.load(fp) 

    infos = list()
    for sample_id in tqdm(split_list):
        sample_id = token2sample[sample_id] if not sample_id.isdigit() else sample_id
        r_velo2cam, t_velo2cam, camera_intrinsic, gt_names, gt_boxes, gt_2dboxes = load_data(dair_root, sample_id, load_gt=load_gt)
        token = "training/image_2/" + sample_id + ".jpg"
        info = dict()
        cam_info = dict()
        info['sample_token'] = token
        info['timestamp'] = 1000000
        info['scene_token'] = token
        cam_names = ['CAM_FRONT']
        lidar_names = ['LIDAR_TOP']
        cam_infos, lidar_infos = dict(), dict()
        for cam_name in cam_names:
            cam_info = dict()
            cam_info['sample_token'] = token
            cam_info['timestamp'] = 1000000
            cam_info['is_key_frame'] = True
            cam_info['height'] = 1080
            cam_info['width'] = 1920
            cam_info['filename'] = token
            ego_pose = {"translation": [0.0, 0.0, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0], "token": token, "timestamp": 1000000}
            cam_info['ego_pose'] = ego_pose
            denorm_file = os.path.join(dair_root, "training/denorm", sample_id + ".txt")
            denorm = load_denorm(denorm_file)

            r_cam2velo, t_cam2velo = cam2velo(r_velo2cam, t_velo2cam)
            calibrated_sensor = {"token": token, "sensor_token": token, "translation": t_cam2velo.flatten(), "rotation_matrix": r_cam2velo, "camera_intrinsic": camera_intrinsic}
            cam_info['calibrated_sensor'] = calibrated_sensor
            cam_info['denorm'] = denorm
            cam_infos[cam_name] = cam_info                  
        for lidar_name in lidar_names:
            lidar_info = dict()
            lidar_info['sample_token'] = token
            ego_pose = {"translation": [0.0, 0.0, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0], "token": token, "timestamp": 1000000}
            lidar_info['ego_pose'] = ego_pose
            lidar_info['timestamp'] = 1000000
            lidar_info['filename'] = None
            lidar_info['calibrated_sensor'] = calibrated_sensor
            lidar_infos[lidar_name] = lidar_info            
        info['cam_infos'] = cam_infos
        info['lidar_infos'] = lidar_infos
        info['sweeps'] = list()
        
        # demo(img_pth, gt_boxes, r_velo2cam, t_velo2cam, camera_intrinsic)   
        ann_infos = list()
        if gt_boxes is not None:
            for idx in range(gt_boxes.shape[0]):
                category_name = gt_names[idx].lower()
                if category_name not in name2nuscenceclass.keys(): continue
                gt_box = gt_boxes[idx]
                gt_2dbox = gt_2dboxes[idx]
                xmin, ymin, xmax, ymax = gt_2dbox
            
                '''
                if xmax <= xmin or ymax <= ymin or xmax - xmin <= 1 or ymax - ymin <=1: 
                    print(xmin, ymin, xmax, ymax)
                    continue
                '''
                lwh = gt_box[3:6]
                loc = gt_box[:3]    # need to certify
                yaw_lidar = gt_box[6]
                rot_mat = np.array([[math.cos(yaw_lidar), -math.sin(yaw_lidar), 0], 
                                    [math.sin(yaw_lidar), math.cos(yaw_lidar), 0], 
                                    [0, 0, 1]])    
                rotation = Quaternion(matrix=rot_mat)
                ann_info = dict()
                ann_info["category_name"] = name2nuscenceclass[category_name]
                ann_info["translation"] = loc
                ann_info["rotation"] = rotation
                ann_info["yaw_lidar"] = yaw_lidar
                ann_info["size"] = lwh
                ann_info["prev"] = ""
                ann_info["next"] = ""
                ann_info["sample_token"] = token
                ann_info["instance_token"] = token
                ann_info["token"] = token
                ann_info["visibility_token"] = "0"
                ann_info["num_lidar_pts"] = 3
                ann_info["num_radar_pts"] = 0            
                ann_info['velocity'] = np.zeros(3)
                ann_infos.append(ann_info)
        info['ann_infos'] = ann_infos
        infos.append(info)
        
    return infos

def parse_args():
    parser = argparse.ArgumentParser(description='Generate ROPE3D KITTI format info files')
    parser.add_argument('--data-root', type=str, default='data/rope3d-kitti', help='Root path to ROPE3D KITTI format dataset')
    return parser.parse_args()

def main():
    args = parse_args()
    data_root = args.data_root
    
    print("process train split")
    train_infos = generate_info_rope3d_kitti(data_root, split='train_dair')
    mmcv.dump(train_infos, os.path.join(data_root, 'rope3d_12hz_infos_train_dair.pkl'))

    print("process val split")
    val_infos = generate_info_rope3d_kitti(data_root, split='val_dair')
    mmcv.dump(val_infos, os.path.join(data_root, 'rope3d_12hz_infos_val_dair.pkl'))
    
    print("process ssdg split")
    print(os.path.join(data_root, "ImageSets", "train_ssdg.txt"))
    if os.path.exists(os.path.join(data_root, "ImageSets", "train_ssdg.txt")):
        print("process train_ssdg split")
        train_ssdg_infos = generate_info_rope3d_kitti(data_root, split='train_ssdg')
        print("train_ssdg_infos: ", len(train_ssdg_infos))
        mmcv.dump(train_ssdg_infos, os.path.join(data_root, 'rope3d_12hz_infos_train_ssdg.pkl'))
    
    print("process unlabeled split")
    if os.path.exists(os.path.join(data_root, "ImageSets", "unlabeled_data.txt")) and "gen" not in data_root:
        unlabeled_infos = generate_info_rope3d_kitti(data_root, split='unlabeled_data', load_gt=False)
        mmcv.dump(unlabeled_infos, os.path.join(data_root, 'rope3d_12hz_infos_unlabeled_data.pkl'))

if __name__ == '__main__':
    main()