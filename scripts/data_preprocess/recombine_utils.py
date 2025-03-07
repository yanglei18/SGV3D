import os
import math
import cv2

import numpy as np
import torch
from PIL import Image
import csv
import json
import random

from scripts.gen_info_dair import *

color_map = {"car":(0, 255, 0), "bus":(0, 255, 255), "van":(255, 255, 0), "truck":(255, 255, 0), "pedestrian":(255, 255, 0),
"cyclist": (255, 255, 0), "bicycle": (255, 255, 0), "tricyclist": (255, 255, 0), "motorcycle": (255, 255, 0), "motorcyclist": (255, 255, 0)}

mask_classes = 3
class2id = {"car": 6, "van": 5, "bus": 4, "truck": 3, "pedestrian": 2, "cyclist": 1, "bicycle": 1, "tricyclist": 1, "motorcycle": 1, "motorcyclist": 1}

class Robutness(object):
    def __init__(self) -> None:
        self.ratio_range = [1.0, 0.20]
        self.roll_range = [0.0, 2.00]
        self.pitch_range = [0.0, 0.67]
    
    def rad2degree(self, radian):
        return radian * 180 / np.pi
    
    def degree2rad(self, degree):
        return degree * np.pi / 180

    def get_M(self, R, K, R_r, K_r):
        R_inv = np.linalg.inv(R)
        K_inv = np.linalg.inv(K)
        M = np.matmul(K_r, R_r)
        M = np.matmul(M, R_inv)
        M = np.matmul(M, K_inv)
        return M

    def alpha2roty(self, alpha, pos):
        ry = alpha + np.arctan2(pos[0], pos[2])
        if ry > np.pi:
            ry -= 2 * np.pi
        if ry < -np.pi:
            ry += 2 * np.pi
        return ry

    def intrin_extrin_recify(self, intrin_mat, sweepego2sweepsensor, ratio_x, ratio_y, roll, pitch):
        # rectify intrin_mat
        intrin_mat_rectify = intrin_mat.copy()
        intrin_mat_rectify[0,0] = intrin_mat[0,0] * ratio_x
        intrin_mat_rectify[1,1] = intrin_mat[1,1] * ratio_y
        
        # rectify sweepego2sweepsensor by roll
        roll_rad = degree2rad(roll)
        rectify_roll = np.array([[math.cos(roll_rad), -math.sin(roll_rad), 0, 0], 
                                    [math.sin(roll_rad), math.cos(roll_rad), 0, 0], 
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])
        sweepego2sweepsensor_rectify_roll = np.matmul(rectify_roll, sweepego2sweepsensor)
        
        # rectify sweepego2sweepsensor by pitch
        pitch_rad = degree2rad(pitch)
        rectify_pitch = np.array([[1, 0, 0, 0],
                                    [0,math.cos(pitch_rad), -math.sin(pitch_rad), 0], 
                                    [0,math.sin(pitch_rad), math.cos(pitch_rad), 0],
                                    [0, 0, 0, 1]])
        sweepego2sweepsensor_rectify_pitch = np.matmul(rectify_pitch, sweepego2sweepsensor_rectify_roll)
        M = self.get_M(sweepego2sweepsensor_rectify_roll[:3,:3], intrin_mat_rectify[:3,:3], sweepego2sweepsensor_rectify_pitch[:3,:3], intrin_mat_rectify[:3,:3])
        center = intrin_mat_rectify[:2, 2]  # w, h
        center_ref = np.array([center[0], center[1], 1.0])
        center_ref = np.matmul(M, center_ref.T)[:2]
        transform_pitch = int(center_ref[1] - center[1])
        intrin_mat_rectify, sweepego2sweepsensor_rectify = torch.Tensor(intrin_mat_rectify), torch.Tensor(sweepego2sweepsensor_rectify_pitch)
        return intrin_mat_rectify, sweepego2sweepsensor_rectify, ratio_x, ratio_y, roll, transform_pitch

    def sample_intrin_extrin_rectify(self, roll_src, pitch_src, fx_src, roll_dest, pitch_dest, fx_dest, intrin_mat, sweepego2sweepsensor):
        # rectify intrin_mat
        ratio = float(fx_dest / fx_src)
        intrin_mat_rectify = intrin_mat.copy()
        intrin_mat_rectify[:2,:2] = intrin_mat[:2,:2] * ratio
        
        # rectify sweepego2sweepsensor by roll
        roll = roll_dest - roll_src
        roll_rad = degree2rad(roll)
        rectify_roll = np.array([[math.cos(roll_rad), -math.sin(roll_rad), 0, 0], 
                                    [math.sin(roll_rad), math.cos(roll_rad), 0, 0], 
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])
        sweepego2sweepsensor_rectify_roll = np.matmul(rectify_roll, sweepego2sweepsensor)
        
        # rectify sweepego2sweepsensor by pitch
        pitch = np.random.normal(self.pitch_range[0], self.pitch_range[1])
        pitch = -1 * (pitch_dest - pitch_src)

        pitch_rad = degree2rad(pitch)
        rectify_pitch = np.array([[1, 0, 0, 0],
                                    [0,math.cos(pitch_rad), -math.sin(pitch_rad), 0], 
                                    [0,math.sin(pitch_rad), math.cos(pitch_rad), 0],
                                    [0, 0, 0, 1]])
        sweepego2sweepsensor_rectify_pitch = np.matmul(rectify_pitch, sweepego2sweepsensor_rectify_roll)
        M = self.get_M(sweepego2sweepsensor_rectify_roll[:3,:3], intrin_mat_rectify[:3,:3], sweepego2sweepsensor_rectify_pitch[:3,:3], intrin_mat_rectify[:3,:3])
        center = intrin_mat_rectify[:2, 2]  # w, h
        center_ref = np.array([center[0], center[1], 1.0])
        center_ref = np.matmul(M, center_ref.T)[:2]
        transform_pitch = int(center_ref[1] - center[1])
        return intrin_mat_rectify, sweepego2sweepsensor_rectify_pitch, ratio, roll, transform_pitch

    def img_intrin_extrin_transform(self, img, ratio, roll, transform_pitch, intrin_mat):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        center = intrin_mat[:2, 2].astype(np.int32) 
        center = (int(center[0]), int(center[1]))

        width, height = img.size[0], img.size[1]
        new_W, new_H = (int(width * ratio), int(height * ratio))
        img = img.resize((new_W, new_H), Image.ANTIALIAS)
        
        h_min = int(center[1] * abs(1.0 - ratio))
        w_min = int(center[0] * abs(1.0 - ratio))
        if ratio <= 1.0:
            image = Image.new(mode='RGB', size=(width, height))
            image.paste(img, (w_min, h_min,  w_min + new_W, h_min + new_H))
        else:
            image = img.crop((w_min, h_min,  w_min + width, h_min + height))
        img = image.rotate(-roll, expand=0, center=center, translate=(0, transform_pitch), fillcolor=(0,0,0), resample=Image.BICUBIC)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return img
    
    def unify_extrinsic_params(self, img, P2, Tr_ego2cam, roll_src, pitch_src, fx_src, roll_dest, pitch_dest, fx_dest):
        P2_rectify, Tr_ego2cam_rectify, ratio, roll, transform_pitch = self.sample_intrin_extrin_rectify(roll_src, pitch_src, fx_src, roll_dest, pitch_dest, fx_dest, P2, Tr_ego2cam)
        img = self.img_intrin_extrin_transform(img, ratio, roll, transform_pitch, P2_rectify)
        return img, P2_rectify, Tr_ego2cam_rectify

    def transform_with_M_bilinear(self, image, M):
        u = range(image.shape[1])
        v = range(image.shape[0])
        xu, yv = np.meshgrid(u, v)
        uv = np.concatenate((xu[:,:,np.newaxis], yv[:,:,np.newaxis]), axis=2)
        uvd = np.concatenate((uv, np.ones((uv.shape[0], uv.shape[1], 1))), axis=-1) * 10
        uvd = uvd.reshape(-1, 3)

        M = np.linalg.inv(M)
        uvd_new = np.matmul(M, uvd.T).T
        uv_new = uvd_new[:,:2] / (uvd_new[:,2][:, np.newaxis])
        uv_new_mask = uv_new.copy()
        uv_new_mask = uv_new_mask.reshape(image.shape[0], image.shape[1], 2)
        
        uv_new[:,0] = np.clip(uv_new[:,0], 0, image.shape[1]-2)
        uv_new[:,1] = np.clip(uv_new[:,1], 0, image.shape[0]-2)
        uv_new = uv_new.reshape(image.shape[0], image.shape[1], 2)
        
        image_new = np.zeros_like(image)
        corr_x, corr_y = uv_new[:,:,1], uv_new[:,:,0]
        point1 = np.concatenate((np.floor(corr_x)[:,:,np.newaxis].astype(np.int32), np.floor(corr_y)[:,:,np.newaxis].astype(np.int32)), axis=2)
        point2 = np.concatenate((point1[:,:,0][:,:,np.newaxis], (point1[:,:,1]+1)[:,:,np.newaxis]), axis=2)
        point3 = np.concatenate(((point1[:,:,0]+1)[:,:,np.newaxis], point1[:,:,1][:,:,np.newaxis]), axis=2)
        point4 = np.concatenate(((point1[:,:,0]+1)[:,:,np.newaxis], (point1[:,:,1]+1)[:,:,np.newaxis]), axis=2)

        fr1 = (point2[:,:,1]-corr_y)[:,:,np.newaxis] * image[point1[:,:,0], point1[:,:,1], :] + (corr_y-point1[:,:,1])[:,:,np.newaxis] * image[point2[:,:,0], point2[:,:,1], :]
        fr2 = (point2[:,:,1]-corr_y)[:,:,np.newaxis] * image[point3[:,:,0], point3[:,:,1], :] + (corr_y-point1[:,:,1])[:,:,np.newaxis] * image[point4[:,:,0], point4[:,:,1], :]
        image_new = (point3[:,:,0] - corr_x)[:,:,np.newaxis] * fr1 + (corr_x - point1[:,:,0])[:,:,np.newaxis] * fr2
        
        mask_1 = np.logical_or(uv_new_mask[:,:,0] < 0, uv_new_mask[:,:,0] > image.shape[1] -2)
        mask_2 = np.logical_or(uv_new_mask[:,:,1] < 0, uv_new_mask[:,:,1] > image.shape[0] -2)
        mask = np.logical_or(mask_1, mask_2)
        image_new[mask] = [0,0,0]
        image_new = image_new.astype(np.float32)
        return image_new

def area(boxes, add1=False):
    if add1:
        return (boxes[:, 2] - boxes[:, 0] + 1.0) * (
            boxes[:, 3] - boxes[:, 1] + 1.0)
    else:
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def intersection(boxes1, boxes2, add1=False):
    [y_min1, x_min1, y_max1, x_max1] = np.split(boxes1, 4, axis=1)
    [y_min2, x_min2, y_max2, x_max2] = np.split(boxes2, 4, axis=1)

    all_pairs_min_ymax = np.minimum(y_max1, np.transpose(y_max2))
    all_pairs_max_ymin = np.maximum(y_min1, np.transpose(y_min2))
    if add1:
        all_pairs_min_ymax += 1.0
    intersect_heights = np.maximum(
        np.zeros(all_pairs_max_ymin.shape),
        all_pairs_min_ymax - all_pairs_max_ymin)

    all_pairs_min_xmax = np.minimum(x_max1, np.transpose(x_max2))
    all_pairs_max_xmin = np.maximum(x_min1, np.transpose(x_min2))
    if add1:
        all_pairs_min_xmax += 1.0
    intersect_widths = np.maximum(
        np.zeros(all_pairs_max_xmin.shape),
        all_pairs_min_xmax - all_pairs_max_xmin)
    return intersect_heights * intersect_widths

def iou(boxes1, boxes2, add1=False):
    intersect = intersection(boxes1, boxes2, add1)
    area1 = area(boxes1, add1)
    area2 = area(boxes2, add1)
    union = np.expand_dims(
        area1, axis=1) + np.expand_dims(
            area2, axis=0) - intersect
    return intersect / (union + 10e-9)

def project_to_image(pts_3d, P):
  pts_3d_homo = np.concatenate(
    [pts_3d, np.ones((pts_3d.shape[0], 1), dtype=np.float32)], axis=1)
  pts_2d = np.dot(P, pts_3d_homo.transpose(1, 0)).transpose(1, 0)
  pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]
  return pts_2d

def draw_box_3d(image, corners, c=(0, 255, 0)):
  face_idx = [[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]]
  for ind_f in [3, 2, 1, 0]:
    f = face_idx[ind_f]
    for j in [0, 1, 2, 3]:
      cv2.line(image, (int(corners[f[j], 0]), int(corners[f[j], 1])),
               (int(corners[f[(j+1)%4], 0]), int(corners[f[(j+1)%4], 1])), c, 2, lineType=cv2.LINE_AA)
    if ind_f == 0:
      cv2.line(image, (int(corners[f[0], 0]), int(corners[f[0], 1])),
               (int(corners[f[2], 0]), int(corners[f[2], 1])), c, 1, lineType=cv2.LINE_AA)
      cv2.line(image, (int(corners[f[1], 0]), int(corners[f[1], 1])),
               (int(corners[f[3], 0]), int(corners[f[3], 1])), c, 1, lineType=cv2.LINE_AA)
  return image

def load_calib(calib_file):
    with open(os.path.join(calib_file), 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=' ')
        for line, row in enumerate(reader):
            if row[0] == 'P2:':
                P2 = np.array([float(i) for i in row[1:]], dtype=np.float32).reshape(3, 4)
    return P2

def load_calib_v2(calib_file):
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

def load_denorm(denorm_file):
    with open(denorm_file, 'r') as f:
        lines = f.readlines()
    denorm = np.array([float(item) for item in lines[0].split(' ')])
    return denorm

def get_denorm(Tr_velo_to_cam):
    ground_points_lidar = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    ground_points_lidar = np.concatenate((ground_points_lidar, np.ones((ground_points_lidar.shape[0], 1))), axis=1)
    ground_points_cam = np.matmul(Tr_velo_to_cam, ground_points_lidar.T).T
    denorm = -1 * equation_plane(ground_points_cam)
    return denorm

def equation_plane(points): 
    x1, y1, z1 = points[0, 0], points[0, 1], points[0, 2]
    x2, y2, z2 = points[1, 0], points[1, 1], points[1, 2]
    x3, y3, z3 = points[2, 0], points[2, 1], points[2, 2]
    a1 = x2 - x1
    b1 = y2 - y1
    c1 = z2 - z1
    a2 = x3 - x1
    b2 = y3 - y1
    c2 = z3 - z1
    a = b1 * c2 - b2 * c1
    b = a2 * c1 - a1 * c2
    c = a1 * b2 - b1 * a2
    d = (- a * x1 - b * y1 - c * z1)
    return np.array([a, b, c, d])

def rad2degree(radian):
            return radian * 180 / np.pi
        
def degree2rad(degree):
    return degree * np.pi / 180 #将度数转换为弧度

def parse_roll_pitch(Tr_ego2cam):
    ground_points_lidar = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    ground_points_lidar = np.concatenate((ground_points_lidar, np.ones((ground_points_lidar.shape[0], 1))), axis=1)
    ground_points_cam = np.matmul(Tr_ego2cam, ground_points_lidar.T).T
    denorm = equation_plane(ground_points_cam)
    
    origin_vector = np.array([0, 1.0, 0])
    target_vector_xy = np.array([denorm[0], denorm[1], 0.0])
    target_vector_yz = np.array([0.0, denorm[1], denorm[2]])
    target_vector_xy = target_vector_xy / np.sqrt(target_vector_xy[0]**2 + target_vector_xy[1]**2 + target_vector_xy[2]**2)       
    target_vector_yz = target_vector_yz / np.sqrt(target_vector_yz[0]**2 + target_vector_yz[1]**2 + target_vector_yz[2]**2)       
    roll = math.acos(np.inner(origin_vector, target_vector_xy))
    pitch = math.acos(np.inner(origin_vector, target_vector_yz))
    roll = -1 * rad2degree(roll) if target_vector_xy[0] > 0 else rad2degree(roll)
    pitch = -1 * rad2degree(pitch) if target_vector_yz[1] > 0 else rad2degree(pitch)
    return roll, pitch

def parse_focal_length(intrin_mat):
    fx, fy = intrin_mat[0,0], intrin_mat[1,1]
    return fx, fy

def parse_height_from_denorm(denorm):
    ref_height = np.abs(denorm[3]) / np.sqrt(denorm[0]**2 + denorm[1]**2 + denorm[2]**2)
    return round(ref_height.astype(np.float32), 6)

def parse_height_from_Tr(Tr_ego2cam):
    denorm = get_denorm(Tr_ego2cam)
    return parse_height_from_denorm(denorm), denorm

# denorm --> Tr_cam2ego (Tr_cam2ego also is Tr_cam2ego)
def get_cam2ego(denorm):
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
    Rz = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    
    origin_vector = np.array([0, 1, 0])
    target_vector = -1 * np.array([denorm[0], denorm[1], denorm[2]])
    target_vector_norm = target_vector / np.sqrt(target_vector[0]**2 + target_vector[1]**2 + target_vector[2]**2)       
    sita = math.acos(np.inner(target_vector_norm, origin_vector))
    n_vector = np.cross(target_vector_norm, origin_vector) 
    n_vector = n_vector / np.sqrt(n_vector[0]**2 + n_vector[1]**2 + n_vector[2]**2)
    n_vector = n_vector.astype(np.float32)
    cam2ego, _ = cv2.Rodrigues(n_vector * sita)
    cam2ego = cam2ego.astype(np.float32)
    cam2ego = np.matmul(Rx, cam2ego)
    cam2ego = np.matmul(Rz, cam2ego)
    
    Ax, By, Cz, D = denorm[0], denorm[1], denorm[2], denorm[3]
    mod_area = np.sqrt(np.sum(np.square([Ax, By, Cz])))
    d = abs(D) / mod_area
    Tr_cam2ego = np.eye(4)
    Tr_cam2ego[:3, :3] = cam2ego
    Tr_cam2ego[:3, 3] = [0, 0, d]
    
    return Tr_cam2ego

# load label_2
def load_annos(label_path, is_pred=False):
    if is_pred:
        fieldnames = ['type', 'truncated', 'occluded', 'alpha', 'xmin', 'ymin', 'xmax', 'ymax', 'dh', 'dw',
                        'dl', 'lx', 'ly', 'lz', 'ry', 'score']
    else:
        fieldnames = ['type', 'truncated', 'occluded', 'alpha', 'xmin', 'ymin', 'xmax', 'ymax', 'dh', 'dw',
                        'dl', 'lx', 'ly', 'lz', 'ry']
    annos = []
    with open(label_path, 'r') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=' ', fieldnames=fieldnames)
        for line, row in enumerate(reader):
            loc = np.array((float(row['lx']), float(row['ly']), float(row['lz'])), dtype=np.float32)
            if math.sqrt(loc[0]**2 + loc[1]**2 + loc[2]**2) > 140: continue 
            
            ry = float(row["ry"])            
            name = row["type"]
            if name.lower() not in color_map.keys(): continue
            dim = [float(row['dh']), float(row['dw']), float(row['dl'])]
            box2d = [float(row['xmin']), float(row['ymin']), float(row['xmax']), float(row['ymax'])]   #2D检测框位置
            truncated_state = float(row["truncated"])
            occluded_state = float(row["occluded"])
            if sum(dim) == 0: continue
            
            score = float(row['score']) if is_pred else 1.0
            if score < 0.70: continue
            anno = {"dim": dim, "loc": loc, "rotation": ry, "name": name, "box2d": box2d, "truncated_state": truncated_state, "occluded_state": occluded_state, "alpha": float(row["alpha"])}
            annos.append(anno)
    return annos

def draw_3d_box_on_image_cam(image, annos, P2, denorm, c=(0, 255, 0)): 
    for line in annos:
        object_type = line['name']
        
        if object_type not in color_map.keys(): 
            continue
        dim = np.array(line['dim']).astype(float)
        location = np.array(line['loc']).astype(float)
        rotation_y = line['rotation']
        box_3d = compute_box_3d_camera_v2(dim, location, rotation_y, denorm)
        box_2d = project_to_image(box_3d, P2)
        image = draw_box_3d(image, box_2d, c=color_map[object_type])
    return image

def draw_3d_box_on_ego(image, annos, P2, Tr_ego2cam, c=None): 
    for line in annos:
        object_type = line['name']
        if object_type.lower() not in color_map.keys(): continue
        box_3d = line['corners_3d']
        box_3d = np.concatenate((box_3d, np.ones((1, box_3d.shape[1]))), axis=0)
        box_3d = np.matmul(Tr_ego2cam, box_3d).T[:, :3]

        box_2d = project_to_image(box_3d, P2)
        color = c if c is not None else color_map[object_type]
        image = draw_box_3d(image, box_2d, c=color)
        '''
        if "bbox" in line.keys():
            bbox = line["bbox"]
            cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, thickness=1, lineType=cv2.LINE_8, shift=0)
        '''
    return image
 
def compute_box_3d_camera_v2(dim, location, rotation_y, denorm):
    c, s = np.cos(rotation_y), np.sin(rotation_y)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    l, w, h = dim[2], dim[1], dim[0]
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [0,0,0,0,-h,-h,-h,-h]
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]

    corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
    corners_3d = np.dot(R, corners) 

    denorm = denorm[:3]
    denorm_norm = denorm / np.sqrt(denorm[0]**2 + denorm[1]**2 + denorm[2]**2)
    ori_denorm = np.array([0.0, -1.0, 0.0])
    theta = -1 * math.acos(np.dot(denorm_norm, ori_denorm))
    n_vector = np.cross(denorm, ori_denorm)
    n_vector_norm = n_vector / np.sqrt(n_vector[0]**2 + n_vector[1]**2 + n_vector[2]**2)
    rotation_matrix, j = cv2.Rodrigues(theta * n_vector_norm)
    corners_3d = np.dot(rotation_matrix, corners_3d)
    corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(3, 1)
    return corners_3d.transpose(1, 0)

def annos_cam2ego(annos, Tr_cam2ego, denorm):
    annos_ego = []
    for line in annos:
        object_type = line['name']
        # if object_type not in color_map.keys(): continue
        dim = np.array(line['dim']).astype(float)
        location = np.array(line['loc']).astype(float)
        rotation_y = line['rotation']
        box2d = line['box2d']
        corners_3d = compute_box_3d_camera_v2(dim, location, rotation_y, denorm).T
        corners_3d = np.concatenate((corners_3d, np.ones((1, corners_3d.shape[1]))), axis=0)
        corners_3d_ego = np.matmul(Tr_cam2ego, corners_3d)[:3, :]

        loc_lidar = np.mean(corners_3d_ego, axis=-1)
        x0, y0 = corners_3d_ego[0, 0], corners_3d_ego[1, 0]
        x3, y3 = corners_3d_ego[0, 3], corners_3d_ego[1, 3]
        dx, dy = x0 - x3, y0 - y3
        rotation = math.atan2(dy, dx)

        anno = {"dim": dim, "loc": loc_lidar, "rotation": rotation, "name": object_type, "box2d": box2d, "corners_3d": corners_3d_ego, "truncated_state": line['truncated_state'], "occluded_state": line["occluded_state"]}
        annos_ego.append(anno)
    return annos_ego

def load_denorm(denorm_file):
    with open(denorm_file, 'r') as f:
        lines = f.readlines()
    denorm = np.array([float(item) for item in lines[0].split(' ')])
    return denorm

def process_sample(data_root, subset, frame_id, is_pred=False, img_dest_path=None):
    calib_file = os.path.join(data_root, subset, "calib", frame_id + ".txt") 
    img_file = os.path.join(data_root, subset, "image_2", frame_id + ".jpg")
    if not os.path.exists(img_file):
        img_file = os.path.join(data_root, subset, "image_2", frame_id + ".png")

    img = cv2.imread(img_file)
    img_path, mask_image = "training/image_2", None
    
    Tr_ego2cam, P2 = load_calib_v2(calib_file) # 得到相机外参Tr_cam2ego
    height, _ = parse_height_from_Tr(Tr_ego2cam)
    Tr_cam2ego = np.linalg.inv(Tr_ego2cam)
    Tr_ego2cam = np.linalg.inv(Tr_cam2ego)
    if is_pred:
        # label_path = os.path.join("outputs/data", frame_id + ".txt")
        label_path = os.path.join(data_root, subset, "label_2", frame_id + ".txt")
        annos_cam = load_annos(label_path, is_pred=True)
    else:
        label_path = os.path.join(data_root, subset, "label_2", frame_id + ".txt")
        annos_cam = load_annos(label_path)

    denorm = get_denorm(Tr_ego2cam)
    annos_ego = annos_cam2ego(annos_cam, Tr_cam2ego, denorm)
    if img_dest_path is not None:
        return {"img": img, "Tr_ego2cam": Tr_ego2cam, "P2": P2, "denorm": denorm, "annos_ego": annos_ego, "frame_id": frame_id, "split": subset, "img_path": img_dest_path, 'height': height, "mask_image": mask_image}
    return {"img": img, "Tr_ego2cam": Tr_ego2cam, "P2": P2, "denorm": denorm, "annos_ego": annos_ego, "frame_id": frame_id, "split": subset, "img_path": img_path, "height": height, "mask_image": mask_image}

'''
def process_sample(data_root, subset, frame_id, img_dest_path=None):
    denorm_file = os.path.join(data_root, subset, "denorm", frame_id + ".txt")
    label_path = os.path.join(data_root, subset, "label_2", frame_id + ".txt")
    calib_file = os.path.join(data_root, subset, "calib", frame_id + ".txt")    
    src_dir = data_root
    if subset == 'training':
        img_paths = ["training-image_2a", "training-image_2b", "training-image_2c", "training-image_2d"]
    else:
        img_paths = ["validation-image_2"]
    for img_path in img_paths:
        img_file = os.path.join(src_dir, img_path, frame_id + ".jpg")
        if os.path.exists(img_file):
            img = cv2.imread(img_file)
            path_img = img_path
            break
    
    denorm = load_denorm(denorm_file)
    Tr_cam2ego = get_cam2ego(denorm) # 得到相机外参Tr_cam2ego    
    Tr_ego2cam = np.linalg.inv(Tr_cam2ego)
    height, _ = parse_height_from_Tr(Tr_ego2cam)
    P2 = load_calib(calib_file)      # P2是内参
    annos_cam = load_annos(label_path)
    annos_ego = annos_cam2ego(annos_cam, Tr_cam2ego, denorm)

    if img_dest_path is not None:
        return {"img": img, "Tr_ego2cam": Tr_ego2cam, "P2": P2, "denorm": denorm, "annos_ego": annos_ego, "frame_id": frame_id, "split": subset, "img_path": img_dest_path, 'height': height}
    return {"img": img, "Tr_ego2cam": Tr_ego2cam, "P2": P2, "denorm": denorm, "annos_ego": annos_ego, "frame_id": frame_id, "split": subset, "img_path": path_img, 'height': height}
'''

def visual_sample_info(sample_info, c=(0,0,255)):
    img, Tr_ego2cam, annos_ego, P2 = sample_info["img"], sample_info["Tr_ego2cam"], sample_info["annos_ego"], sample_info["P2"]
    img = draw_3d_box_on_ego(img.copy(), annos_ego, P2, Tr_ego2cam, c)
    roll, pitch, ref_height, fx, fy = parse_tools(Tr_ego2cam, P2)
    return img

def visual_sample(data_root, subset, frame_id):
    sample_info = process_sample(data_root, subset, frame_id)
    img = visual_sample_info(sample_info)
    return img

def parse_tools(Tr_ego2cam, intrin_mat):
    fx, fy = parse_focal_length(intrin_mat)
    ref_height = parse_height_from_Tr(Tr_ego2cam)
    roll, pitch = parse_roll_pitch(Tr_ego2cam)
    return roll, pitch, ref_height, fx, fy

def unify_extrinsic_params_tools(robutness, sample_info, sample_info_dest):
    img, Tr_ego2cam, P2, annos_ego = sample_info["img"], sample_info["Tr_ego2cam"], sample_info["P2"], sample_info["annos_ego"]
    Tr_ego2cam_dest, P2_dest = sample_info_dest["Tr_ego2cam"], sample_info_dest["P2"]

    M = robutness.get_M(Tr_ego2cam[:3,:3], P2[:3,:3], Tr_ego2cam_dest[:3,:3], P2_dest[:3,:3])
    img = robutness.transform_with_M_bilinear(img, M)
    denorm = get_denorm(Tr_ego2cam_dest)

    # height_src, height_dest = sample_info["height"], sample_info_dest["height"]
    Tr_ego2cam, Tr_ego2cam_dest = sample_info["Tr_ego2cam"], sample_info_dest["Tr_ego2cam"]
    Tr_cam2ego, Tr_cam2ego_dest = np.linalg.inv(Tr_ego2cam), np.linalg.inv(Tr_ego2cam_dest)
    delta_loc = (Tr_cam2ego_dest[:3, 3] - Tr_cam2ego[:3, 3])
    for i in range(len(annos_ego)):
        '''
        delta_height = height_dest - height_src
        annos_ego[i]['corners_3d'][2, :] += delta_height
        annos_ego[i]['loc'][2] += delta_height
        '''
        annos_ego[i]['corners_3d'] += delta_loc[:, np.newaxis]
        annos_ego[i]['loc'] += delta_loc

    return {"img": img, "Tr_ego2cam": Tr_ego2cam_dest, "P2": P2_dest, "denorm": denorm, "annos_ego": annos_ego}

# The function of this function: unify the height of the two samples
def unify_ref_height_tools(sample_info, sample_info_dest):
    img, _, P2, denorm, annos_ego = sample_info["img"], sample_info["Tr_ego2cam"], sample_info["P2"], sample_info["denorm"], sample_info["annos_ego"]
    denorm_dest = sample_info_dest["denorm"]
    ref_height_dest = parse_height_from_denorm(denorm_dest)
    denorm_rectify = [denorm[0], denorm[1], denorm[2], ref_height_dest]
    Tr_cam2ego_rectify = get_cam2ego(denorm_rectify)
    Tr_ego2cam_rectify = np.linalg.inv(Tr_cam2ego_rectify)

    annos_rectify = list()
    for anno in annos_ego:
        dim, loc, rotation, corners_3d = anno["dim"], anno["loc"], anno["rotation"], anno["corners_3d"]
        anno_rectify = {"dim": dim, "loc": loc, "rotation": rotation, "name": anno["name"], "box2d": None, "corners_3d": corners_3d, "truncated_state": anno['truncated_state'], "occluded_state": anno["occluded_state"]}
        annos_rectify.append(anno_rectify)
    sample_info_rectify = {"img": img, "Tr_ego2cam": Tr_ego2cam_rectify, "P2": P2, "denorm": denorm_rectify, "annos_ego": annos_rectify}
    return sample_info_rectify

def update_bbox_info(sample_info):
    Tr_ego2cam, P2, annos_ego = sample_info["Tr_ego2cam"], sample_info["P2"], sample_info["annos_ego"]
    annos_ego_ret = list()
    for anno in annos_ego:
        box_3d = anno["corners_3d"]
        box_3d = np.concatenate((box_3d, np.ones((1, box_3d.shape[1]))), axis=0)
        box_3d = np.matmul(Tr_ego2cam, box_3d).T[:, :3]
        box_2d = project_to_image(box_3d, P2)
        xmin, ymin, xmax, ymax = np.min(box_2d[:, 0]), np.min(box_2d[:, 1]), np.max(box_2d[:, 0]), np.max(box_2d[:, 1])
        if xmax <= 0 or ymax <= 0: continue
        xmin, ymin = max(0, xmin), max(0, ymin)
        bbox = [xmin, ymin, xmax, ymax]
        anno["bbox"] = bbox
        annos_ego_ret.append(anno)
    sample_info["annos_ego"] = annos_ego_ret
    return sample_info

def update_mask_info(predictor, sample_info):
    if sample_info["mask_image"] is not None: return sample_info
    img, annos_ego = sample_info["img"], sample_info["annos_ego"]
    bbox_prompts, bbox_labels = [], []
    for anno in annos_ego:
        bbox = np.array(anno["bbox"]).astype(np.int32)
        xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
        if xmax <= 0 or ymax <= 0: continue
        xmin, ymin = max(0, xmin), max(0, ymin)
        xmax, ymax = min(xmax, img.shape[1] - 1), min(ymax, img.shape[0] - 1)
        bbox_prompts.append([xmin, ymin, xmax, ymax])
        bbox_labels.append(class2id[anno["name"].lower()])
    bbox_prompts = torch.tensor(np.array(bbox_prompts))
    mask_image = get_sam_mask(predictor, bbox_prompts, bbox_labels, img)
    sample_info["mask_image"] = mask_image
    return sample_info

def get_sam_mask(predictor, bbox_prompts, bbox_labels, img):
    mask_image = np.zeros((1080, 1920, 1))
    if bbox_prompts.shape[0] > 0:
        masks = mask_inference(predictor, bbox_prompts, img)
        for i in range(masks.shape[0]):
            mask = masks[i]
            mask = mask.cpu().numpy()
            h, w = mask.shape[-2:]
            mask = mask.reshape(h, w, 1).astype(np.uint8)
            mask_image += (mask * bbox_labels[i]) * (mask_image == 0)
    mask_image = np.clip(mask_image, 0, 6).astype(np.uint8)
    return mask_image

def mask_inference(predictor, bboxes, image_ori):
    image = cv2.cvtColor(image_ori, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)
    boxes_filt = bboxes.cpu()
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to("cuda")

    outputs = predictor.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes.to("cuda"),
        multimask_output = False,
    )
    masks = outputs[0]
    return masks

def objects_combine_tools(predictor, sample_info_list, sample_info_dest, sample_ratio):
    sample_info_dest = update_bbox_info(sample_info_dest)
    img_dest, annos_ego_dest = sample_info_dest["img"], sample_info_dest["annos_ego"]
    init_bboxes, bbox_labels = [], []
    if len(annos_ego_dest) > 0:
        for anno in annos_ego_dest:
            init_bboxes.append(anno["bbox"])
            bbox_labels.append(class2id[anno["name"].lower()])
        init_bboxes = np.array(init_bboxes)
    else:
        init_bboxes = np.array([[0, 0, 0, 0]])
        bbox_labels.append(0)
    bbox_prompts = torch.tensor(init_bboxes)
    mask_image_dest = get_sam_mask(predictor, bbox_prompts, bbox_labels, img_dest)
    gray_img_dest = cv2.cvtColor(img_dest, cv2.COLOR_BGR2GRAY)

    for sample_info in sample_info_list:
        img, annos_ego = sample_info["img"], sample_info["annos_ego"]
        Tr_ego2cam, P2, denorm = sample_info["Tr_ego2cam"], sample_info["P2"], sample_info["denorm"]
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness_img, brightness_img_dest = np.mean(gray_img), np.mean(gray_img_dest)
        beta = 100 * (brightness_img_dest - brightness_img) / brightness_img
        beta = (1 if beta > 0 else -1) * min(abs(beta), 60)
        img = cv2.convertScaleAbs(sample_info["img"], alpha=1.0, beta=beta)

        cls_focus = ["car", "van", "truck", "bus", "pedestrian", "cyclist"]
        annos_ego_selected = list()
        for anno in annos_ego:
            if anno['name'].lower() in cls_focus:
                annos_ego_selected.append(anno)
        bbox_prompts, bbox_labels = [], []
        annos_ego = random.sample(annos_ego_selected, int(sample_ratio * len(annos_ego_selected)))
        for anno in annos_ego:
            bbox = np.array(anno["bbox"]).astype(np.int32)
            xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
            if xmax <= 0 or ymax <= 0: continue
            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(xmax, img.shape[1] - 1), min(ymax, img.shape[0] - 1)

            if xmax <= xmin or ymax <= ymin or xmax - xmin <= 1 or ymax - ymin <=1: 
                continue

            bbox = np.array([xmin, ymin, xmax, ymax])
            ious = iou(init_bboxes, bbox[np.newaxis, ...])
            if np.max(ious) < 0.15: 
                init_bboxes = np.vstack((init_bboxes, bbox[np.newaxis, ...]))
                annos_ego_dest.append(anno)
                bbox_prompts.append([xmin, ymin, xmax, ymax])
                bbox_labels.append(class2id[anno['name'].lower()])
        
        bbox_prompts = torch.tensor(np.array(bbox_prompts))
        mask_image_src = get_sam_mask(predictor, bbox_prompts, bbox_labels, img)

        mask_src = (mask_image_src > 0).astype(np.uint8)
        img_dest = img_dest * (1 - mask_src) + img * mask_src
        mask_image_dest = mask_image_dest * (1 - mask_src) + mask_image_src * mask_src
        mask_image_dest = np.clip(mask_image_dest, 0, 6)
    sample_info_combined = {"Tr_ego2cam": Tr_ego2cam, "P2": P2, "denorm": denorm, "img": img_dest, "annos_ego": annos_ego_dest, "mask_image": mask_image_dest}
    return sample_info_combined


def frame_combine_tools(predictor, robutness, data_root, frame_id_list, frame_id_dest, cnt, sample_ratio, is_pred=False, vis=False):
    sample_info_dest = process_sample(data_root, "training", frame_id_dest)
    sample_info_src_list = list()
    for frame_id in frame_id_list:
        sample_info_train = process_sample(data_root, "training", frame_id, is_pred)
        sample_info_src = unify_extrinsic_params_tools(robutness, sample_info_train, sample_info_dest)
        sample_info_src = update_bbox_info(sample_info_src)
        sample_info_src_list.append(sample_info_src)
    sample_info_combined = objects_combine_tools(predictor, sample_info_src_list, sample_info_dest, sample_ratio)
    sample_info_combined["frame_id"] = "{:06d}".format(cnt)
    sample_info_combined["split"] = "training"

    if vis:
        conbined_img = visual_sample_info(sample_info_combined)
        cv2.imwrite("combine2.jpg", conbined_img)
        height_src, _ = parse_height_from_Tr(sample_info_train["Tr_ego2cam"])
        height_dest, _ = parse_height_from_Tr(sample_info_dest["Tr_ego2cam"])
        print("height_src height_dest: ", height_src, height_dest)
    return sample_info_combined


def write_kitti_in_txt(pred_lines, path_txt):
    wf = open(path_txt, "w")
    for line in pred_lines:
        line_string = " ".join(line) + "\n"
        wf.write(line_string)
    wf.close()

def calib_generation(Tr_ego2cam, P2, calib_path):
    kitti_calib = dict()
    kitti_calib["P0"] = np.zeros((3, 4))  # Dummy values.
    kitti_calib["P1"] = np.zeros((3, 4))  # Dummy values.
    kitti_calib["P2"] = P2                # Left camera transform.
    # Cameras are already rectified.
    kitti_calib["Tr_velo_to_cam"] = Tr_ego2cam[:3, :4]
    with open(calib_path, "w") as calib_file:
        for (key, val) in kitti_calib.items():
            val = val.flatten()
            val_str = "%.12e" % val[0]
            for v in val[1:]:
                val_str += " %.12e" % v
            calib_file.write("%s: %s\n" % (key, val_str))

def denorm_generation(denorm, denorm_path):
    with open(denorm_path, "w") as denorm_file:
        denorm = [str(item) for item in denorm]
        line_string = " ".join(denorm) + "\n"
        denorm_file.write(line_string)

def normalize_angle(angle):
    alpha_tan = np.tan(angle)
    alpha_arctan = np.arctan(alpha_tan)
    if np.cos(angle) < 0:
        alpha_arctan = alpha_arctan + math.pi
    return alpha_arctan

def label_generation(Tr_ego2cam, annos_ego, label_path):
    lines = []
    for anno in annos_ego:
        detection_score = anno["score"] if "score" in anno.keys() else 1.0
        dim, name, bbox = anno["dim"], anno["name"], anno["bbox"]
        h, w, l = dim[0], dim[1], dim[2]
        truncated_state, occluded_state = anno["truncated_state"], anno["occluded_state"]
        box_3d = anno["corners_3d"]
        box_3d = np.concatenate((box_3d, np.ones((1, box_3d.shape[1]))), axis=0)
        box_3d = np.matmul(Tr_ego2cam, box_3d).T[:, :3]
        loc = np.mean(box_3d, axis=0)
        loc[1] += h/2

        x0, z0 = box_3d[0, 0], box_3d[0, 2]
        x3, z3 = box_3d[3, 0], box_3d[3, 2]
        dx, dz = x0 - x3, z0 - z3
        rotation = math.atan2(-dz, dx)

        alpha = rotation - math.atan2(loc[0], loc[2])
        if alpha > math.pi:
            alpha = alpha - 2.0 * math.pi
        if alpha <= (-1 * math.pi):
            alpha = alpha + 2.0 * math.pi
        alpha = normalize_angle(alpha)

        i1 = name
        i2 = str(truncated_state)
        i3 = str(occluded_state)
        i4 = str(round(alpha, 4))
        i5, i6, i7, i8 = (
            str(round(bbox[0], 4)),
            str(round(bbox[1], 4)),
            str(round(bbox[2], 4)),
            str(round(bbox[3], 4)),
        )
        i9, i10, i11 = str(round(h, 4)), str(round(w, 4)), str(round(l, 4))
        i12, i13, i14 = str(round(loc[0], 4)), str(round(loc[1], 4)), str(round(loc[2], 4))
        i15 = str(round(rotation, 4))
        line = [i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, str(round(detection_score, 4))]
        lines.append(line)
    write_kitti_in_txt(lines, label_path)


def save_kitti_format(data_root, sample_info, img_path):
    os.makedirs(os.path.join(data_root, "training", "denorm"), exist_ok=True)
    os.makedirs(os.path.join(data_root, "training", "calib"), exist_ok=True)
    os.makedirs(os.path.join(data_root, "training", "label_2"), exist_ok=True)
    os.makedirs(os.path.join(data_root, "training", "mask_image"), exist_ok=True)
    os.makedirs(os.path.join(data_root, img_path), exist_ok=True)

    img, Tr_ego2cam, P2, annos_ego = sample_info["img"], sample_info["Tr_ego2cam"], sample_info["P2"], sample_info["annos_ego"]
    denorm = sample_info['denorm']
    frame_id = sample_info["frame_id"]

    if "mask_image" in sample_info.keys():
        mask_image = np.repeat(sample_info["mask_image"], 3, axis=2) * 40
        np.save(os.path.join(data_root, "training", "mask_image", frame_id + ".npy"), mask_image)
    cv2.imwrite(os.path.join(data_root, img_path, frame_id + ".jpg"), img)
    calib_generation(Tr_ego2cam, P2, os.path.join(data_root, "training", "calib", frame_id + ".txt"))
    denorm_generation(denorm, os.path.join(data_root, "training", "denorm", frame_id + ".txt"))
    label_generation(Tr_ego2cam, annos_ego, os.path.join(data_root, "training", "label_2", frame_id + ".txt"))

'''
def save_kitti_format(data_root, sample_info, img_path):
    os.makedirs(os.path.join(data_root, "training", "denorm"), exist_ok=True)
    os.makedirs(os.path.join(data_root, "training", "calib"), exist_ok=True)
    os.makedirs(os.path.join(data_root, "training", "label_2"), exist_ok=True)
    
    os.makedirs(os.path.join(data_root, "validation", "denorm"), exist_ok=True)
    os.makedirs(os.path.join(data_root, "validation", "calib"), exist_ok=True)
    os.makedirs(os.path.join(data_root, "validation", "label_2"), exist_ok=True)
    
    os.makedirs(os.path.join(data_root, img_path), exist_ok=True)

    img, Tr_ego2cam, P2, annos_ego = sample_info["img"], sample_info["Tr_ego2cam"], sample_info["P2"], sample_info["annos_ego"]
    denorm = sample_info['denorm']
    frame_id = sample_info["frame_id"]

    cv2.imwrite(os.path.join(data_root, img_path, frame_id + ".jpg"), img)

    subset = "validation" if "validation" in img_path else "training"
    calib_generation(Tr_ego2cam, P2, os.path.join(data_root, subset, "calib", frame_id + ".txt"))
    denorm_generation(denorm, os.path.join(data_root, subset, "denorm", frame_id + ".txt"))
    label_generation(Tr_ego2cam, annos_ego, os.path.join(data_root, subset, "label_2", frame_id + ".txt"))
'''