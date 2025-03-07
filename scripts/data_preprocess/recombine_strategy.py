import os
import json
import argparse
import random
import shutil

import numpy as np
from tqdm import tqdm

# segment anything
from segment_anything import (
    build_sam,
    SamPredictor
)

from scripts.data_preprocess.recombine_utils import Robutness
from scripts.data_preprocess.recombine_utils import load_annos, process_sample, update_bbox_info, update_mask_info, unify_extrinsic_params_tools, visual_sample_info
from scripts.data_preprocess.recombine_utils import frame_combine_tools, save_kitti_format, load_calib_v2, parse_height_from_Tr

def parse_option():
    parser = argparse.ArgumentParser('SGV3D data preprocess', add_help=False)
    parser.add_argument('--src-root', type=str, default="data/rope3d-kitti", help='root path to src rope3d-kitti dataset')
    parser.add_argument('--dest-root', type=str, default="data/rope3d-kitti-gen", help='root path to result rope3d-kitti dataset')
    parser.add_argument('--vis', type=bool, default=False, help='flag to control visualization')
    parser.add_argument('--update-train-split', action='store_true', help='flag to update training split file')
    parser.add_argument('--train-split-file', type=str, default="data/rope3d/training/ImageSets/train.txt", help='path to training split file')
    args = parser.parse_args()
    return args

def read_split(split_txt, token2sample_txt):
    with open(token2sample_txt, 'r') as fp:
        token2sample = json.load(fp) 

    with open(split_txt, "r") as file:
        lines = file.readlines()
    split_list = list()
    for line in lines:
        frame_id = line.rstrip('\n') if line.rstrip('\n').isdigit() else token2sample[line.rstrip('\n')]
        split_list.append(frame_id)
    return split_list

def write_split(split_list, split_txt):
    wf = open(split_txt, "w")
    for line in split_list:
        line_string = line + "\n"
        wf.write(line_string)
    wf.close()

def is_exists(data_root, subset, frame_id):
    img_file = os.path.join(data_root, subset, "mask_image", frame_id + ".npy")
    return True if os.path.exists(img_file) else False

def get_background_ids(src_root, frame_ids, cls_focus, cnt_threshold):
    P2_cache, background_count, background_ids = dict(), dict(), list()
    for frame_id in tqdm(frame_ids):
        label_path = os.path.join(src_root, "training", "label_2", frame_id + ".txt")
        calib_path = os.path.join(src_root, "training", "calib", frame_id + ".txt")
        annos_cam = load_annos(label_path)
        Tr_ego2cam, P2 = load_calib_v2(calib_path)
        height, _ = parse_height_from_Tr(Tr_ego2cam)
        obj_cnt = 0
        for anno in annos_cam:
            if anno["name"].lower() in cls_focus:
                obj_cnt += 1
        if obj_cnt < cnt_threshold or P2[0, 0] not in P2_cache.keys():
            background_ids.append((frame_id, height))
            if P2[0, 0] not in background_count.keys():
                background_count[P2[0, 0]] = [(frame_id, obj_cnt, height)]
            else:
                background_count[P2[0, 0]].append((frame_id, obj_cnt, height))
            if P2[0, 0] not in P2_cache.keys():
                P2_cache[P2[0, 0]] = height
    return background_ids, background_count, P2_cache

def split_frame_ids(src_root, frame_ids, P2_cache):
    frame_ids_dict = dict()
    for P2, height in P2_cache.items():
        frame_ids_dict[height] = list()

    for frame_id in tqdm(frame_ids):
        calib_path = os.path.join(src_root, "training", "calib", frame_id + ".txt")
        Tr_ego2cam, P2 = load_calib_v2(calib_path)
        height, _ = parse_height_from_Tr(Tr_ego2cam)
        for h_key in frame_ids_dict.keys():
            if abs(height - h_key) < 0.30:
                frame_ids_dict[h_key].append(frame_id)
    return frame_ids_dict

def combination_process(train_frame_ids, src_root, P2_cache, background_ids, combine_frame_ids, num_frames, count, is_pred, vis, batch_ratio=0.2):
    train_frame_ids_dict = split_frame_ids(src_root, train_frame_ids, P2_cache)
    for back_id, back_height in tqdm(background_ids):
        sample_batch = min(int(float(1 / len(background_ids)) * len(train_frame_ids)), len(train_frame_ids_dict[back_height]))
        random_frame_ids = random.sample(train_frame_ids_dict[back_height], sample_batch)
        for _ in tqdm(random_frame_ids):
            if not is_exists(dest_root, "training", "{:06d}".format(count)):
                train_id_list = random.sample(random_frame_ids, num_frames)
                sample_info_combined = frame_combine_tools(predictor, robutness, src_root, train_id_list, back_id, count, sample_ratio=1.0, is_pred=is_pred, vis=vis)
                combine_frame_ids.append(sample_info_combined["frame_id"])
                save_kitti_format(dest_root, sample_info_combined, "training/image_2")
            count += 1
    return combine_frame_ids, count

def convert_raw_frames(src_root, dest_root, frame_id):
    src_img_file = os.path.join(src_root, "training", "image_2", frame_id + ".jpg")
    src_calib_file = os.path.join(src_root, "training", "calib", frame_id + ".txt") 
    src_label_file = os.path.join(src_root, "training", "label_2", frame_id + ".txt")
    src_denorm_file = os.path.join(src_root, "training", "denorm", frame_id + ".txt")
    dest_img_file = os.path.join(dest_root, "training", "image_2", frame_id + ".jpg")
    dest_calib_file = os.path.join(dest_root, "training", "calib", frame_id + ".txt") 
    dest_label_file = os.path.join(dest_root, "training", "label_2", frame_id + ".txt")
    dest_denorm_file = os.path.join(dest_root, "training", "denorm", frame_id + ".txt")

    shutil.copyfile(src_img_file, dest_img_file)
    shutil.copyfile(src_calib_file, dest_calib_file)
    shutil.copyfile(src_label_file, dest_label_file)
    shutil.copyfile(src_denorm_file, dest_denorm_file)

def safe_copy(src, dst):
    """Safely copy file from src to dst, skip if dst already exists"""
    if os.path.exists(dst):
        print(f"File {dst} already exists, skipping...")
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)

def count_files_in_dir(dir_path):
    """Count the number of files in a directory"""
    if not os.path.exists(dir_path):
        return 0
    return len([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])

if __name__ == "__main__":
    robutness = Robutness()
    args = parse_option()
    split_root = 'data/rope3d-kitti/ImageSets'
    src_root, dest_root, vis = args.src_root, args.dest_root, args.vis
    token2sample_txt = os.path.join(split_root, "../map_token2id.json")
    train_frame_ids = read_split(os.path.join(split_root, "train_dair.txt"), token2sample_txt)
    val_frame_ids = read_split(os.path.join(split_root, "val_dair.txt"), token2sample_txt)
    raw_val_frame_ids = read_split(os.path.join(split_root, "unlabeled_data.txt"), token2sample_txt)
    raw_frame_ids = list(set(raw_val_frame_ids))

    if args.update_train_split:
        print(f"Updating training split file at {args.train_split_file}")
        write_split(train_frame_ids, args.train_split_file)
        print("Training split file updated successfully")
        exit(0)

    print("init sam predictor ....")
    predictor = SamPredictor(build_sam(checkpoint="./sam_vit_h_4b8939.pth").to("cuda"))
    print("stage 01: processing train split ...")
    os.makedirs("data/rope3d-kitti/training/mask_image", exist_ok=True)
    
    for frame_id in tqdm(train_frame_ids):
        if is_exists(dest_root, "training", frame_id): continue
        sample_info= process_sample(src_root, "training", frame_id,)
        sample_info = update_bbox_info(sample_info)
        if os.path.exists(os.path.join("data/rope3d-kitti/training/mask_image", frame_id + ".npy")):
            mask_image = np.load(os.path.join("data/rope3d-kitti/training/mask_image", frame_id + ".npy"))
            mask_image = (mask_image / 40).astype(np.uint8)
            mask_image = mask_image[:, :, 0][:, :, np.newaxis]
            sample_info["mask_image"] = mask_image
        else:
            sample_info = update_mask_info(predictor, sample_info)
        save_kitti_format(dest_root, sample_info, sample_info["img_path"])
    
    print("stage 02: processing val split ...")
    for frame_id in tqdm(val_frame_ids):
        if is_exists(dest_root, "training", frame_id): continue
        sample_info= process_sample(src_root, "training", frame_id)
        sample_info = update_bbox_info(sample_info)
        if os.path.exists(os.path.join("data/rope3d-kitti/training/mask_image", frame_id + ".npy")):
            mask_image = np.load(os.path.join("data/rope3d-kitti/training/mask_image", frame_id + ".npy"))
            mask_image = (mask_image / 40).astype(np.uint8)
            mask_image = mask_image[:, :, 0][:, :, np.newaxis]
            sample_info["mask_image"] = mask_image
        else:
            sample_info = update_mask_info(predictor, sample_info)
        save_kitti_format(dest_root, sample_info, sample_info["img_path"])
        sample_info_val = sample_info
    
    cls_focus = ["car", "van", "truck", "bus", "pedestrian", "cyclist", "motorcyclist", "tricyclist"]
    print("stage 04: processing raw split to select background images")
    val_background_ids, background_count, P2_cache = get_background_ids(src_root, list(set(raw_frame_ids)), cls_focus, 5)
    
    background_ids = list()
    for k, v in background_count.items():
        print(k, len(v))
        sorted_v = sorted(v, key=lambda x: x[1])
        for frame_id, obj_cnt, height in sorted_v[:50]:
            print(frame_id, obj_cnt, height)
            background_ids.append((frame_id, height))
    print("background_ids: ", len(background_ids))
    
    print("stage 05: processing conbine split with labeled data ...")

    combine_frame_ids = list()
    count, num_frames = 100000, 3
    outputs_data_dir = "outputs/data"
    print("stage 06: processing conbine split with unlabeled data ...", count_files_in_dir(outputs_data_dir))
    if os.path.exists(outputs_data_dir) and "gen" in dest_root:
        num_frames = 3
        combine_frame_ids, count = combination_process(raw_frame_ids, src_root, P2_cache, background_ids, combine_frame_ids, num_frames, count, True, vis)
    else:
        print("no unlabeled data, skip this stage")

    print("total conbine samples: ", count)
    print("stage 06: saving split set ...")
    ratio = int(len(combine_frame_ids) / len(train_frame_ids))
    train_frame_ids = combine_frame_ids + train_frame_ids

    os.makedirs(os.path.join(dest_root, "ImageSets"), exist_ok=True)
    if os.path.exists(outputs_data_dir) and "gen" in dest_root:
        print("process train_ssdg split", len(train_frame_ids))
        write_split(train_frame_ids, os.path.join(dest_root, "ImageSets", "train_ssdg.txt"))

    safe_copy(os.path.join(src_root, "map_token2id.json"), os.path.join(dest_root, "map_token2id.json"))
    if not os.path.exists(os.path.join(dest_root, "training", "label_eval")):
        shutil.copytree(os.path.join(src_root, "training", "label_eval"), os.path.join(dest_root, "training", "label_eval"))
    
    safe_copy(os.path.join(src_root, "ImageSets", "train_dair.txt"), os.path.join(dest_root, "ImageSets", "train_dair.txt"))
    safe_copy(os.path.join(src_root, "ImageSets", "val_dair.txt"), os.path.join(dest_root, "ImageSets", "val_dair.txt"))
    safe_copy(os.path.join(src_root, "ImageSets", "unlabeled_data.txt"), os.path.join(dest_root, "ImageSets", "unlabeled_data.txt"))