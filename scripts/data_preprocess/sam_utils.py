import os
import cv2
import csv

import torch

# segment anything
from segment_anything import (
    build_sam,
    build_sam_hq,
    SamPredictor
)

import numpy as np
import matplotlib.pyplot as plt

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    ax.text(x0, y0, label)

def load_annos(label_path):
    fieldnames = ['type', 'truncated', 'occluded', 'alpha', 'xmin', 'ymin', 'xmax', 'ymax', 'dh', 'dw',
                    'dl', 'lx', 'ly', 'lz', 'ry']
    bboxes, labels = [], []
    with open(label_path, 'r') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=' ', fieldnames=fieldnames)
        for line, row in enumerate(reader):
            name = row["type"]    
            dim = [float(row['dh']), float(row['dw']), float(row['dl'])]
            bbox = [float(row['xmin']), float(row['ymin']), float(row['xmax']), float(row['ymax'])]   #2D检测框位置
            if sum(dim) == 0:
                continue
            bboxes.append(bbox)
            labels.append(name)

    bboxes = torch.tensor(np.array(bboxes))
    return bboxes, labels

def mask_inference(predictor, bboxes, image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)
    boxes_filt = bboxes.cpu()
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

    masks, _, _ = predictor.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes.to(device),
        multimask_output = False,
    )
    return masks

def sam_combine_tools(predictor, bboxes, src_image, dest_image):
    masks = mask_inference(predictor, bboxes, src_image)
    mask_image = np.zeros((1080, 1920, 1))
    for mask in masks:
        mask = mask.cpu().numpy()
        h, w = mask.shape[-2:]
        mask = mask.reshape(h, w, 1).astype(np.int8)
        mask_image += mask
        print(np.min(mask), np.max(mask))
    dest_image = dest_image * (1 - mask_image) + src_image * mask_image        
    return dest_image, masks

def sam_init(sam_checkpoint, device="cuda"):
    predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))
    return predictor

if __name__ == "__main__":
    sam_checkpoint, sam_hq_checkpoint = "./sam_vit_h_4b8939.pth", "./sam_hq_vit_h.pth"
    use_sam_hq = False
    device = "cuda"
    output_dir = "/workspace/Grounded-Segment-Anything/output"
    image_path = "/data/Rope3D_0083/training-image_2a/145040_fa2sd4a06W152AIR_420_1626155124_1626155723_160_obstacle.jpg"
    label_path = "/data/Rope3D_0083/training/label_2/145040_fa2sd4a06W152AIR_420_1626155124_1626155723_160_obstacle.txt"

    dest_image_path = "/data/Rope3D_0083/training-image_2a/1679_fa2sd4adatasetNorth151_420_1616054178_1616054482_8_obstacle.jpg"

    # make dir
    os.makedirs(output_dir, exist_ok=True)
    # initialize SAM
    if use_sam_hq:
        predictor = SamPredictor(build_sam_hq(checkpoint=sam_hq_checkpoint).to(device))
    else:
        predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))
    bboxes, labels = load_annos(label_path)
    image = cv2.imread(image_path)
    dest_image = cv2.imread(dest_image_path)
    dest_image, masks = sam_combine_tools(predictor, bboxes, image, dest_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # draw output image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    for box, label in zip(bboxes, labels):
        show_box(box.numpy(), plt.gca(), label)

    plt.axis('off')
    plt.savefig(
        os.path.join(output_dir, "grounded_sam_output.jpg"), 
        bbox_inches="tight", dpi=300, pad_inches=0.0
    )
    cv2.imwrite(os.path.join(output_dir, "sam_combine.jpg"), dest_image)