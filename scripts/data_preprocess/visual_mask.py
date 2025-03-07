import os
import cv2
import torch
import numpy as np
from PIL import Image

ida_aug_conf = {
    'final_dim':
    (864, 1536),
    'H':
    1080,
    'W':
    1920,
    'bot_pct_lim': (0.0, 0.0),
    'cams': ['CAM_FRONT'],
    'Ncams': 1,
}

def sample_ida_augmentation():
    """Generate ida augmentation values based on ida_config."""
    H, W = ida_aug_conf['H'], ida_aug_conf['W']
    fH, fW = ida_aug_conf['final_dim']
    resize = max(fH / H, fW / W)
    resize_dims = (int(W * resize), int(H * resize))
    newW, newH = resize_dims
    crop_h = int(
        (1 - np.mean(ida_aug_conf['bot_pct_lim'])) * newH) - fH
    crop_w = int(max(0, newW - fW) / 2)
    crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
    flip = False
    rotate_ida = 0
    return resize, resize_dims, crop, flip, rotate_ida

def get_rot(h):
    return torch.Tensor([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])

def img_transform(img, resize, resize_dims, crop, flip, rotate):
    ida_rot = torch.eye(2)
    ida_tran = torch.zeros(2)
    # adjust image
    img = img.resize(resize_dims)
    img = img.crop(crop)
    if flip:
        img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
    img = img.rotate(rotate)

    # post-homography transformation
    ida_rot *= resize
    ida_tran -= torch.Tensor(crop[:2])
    if flip:
        A = torch.Tensor([[-1, 0], [0, 1]])
        b = torch.Tensor([crop[2] - crop[0], 0])
        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b
    A = get_rot(rotate / 180 * np.pi)
    b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
    b = A.matmul(-b) + b
    ida_rot = A.matmul(ida_rot)
    ida_tran = A.matmul(ida_tran) + b
    ida_mat = ida_rot.new_zeros(4, 4)
    ida_mat[3, 3] = 1
    ida_mat[2, 2] = 1
    ida_mat[:2, :2] = ida_rot
    ida_mat[:2, 3] = ida_tran
    return img, ida_mat


image_folder, mask_folder = "image_2", "mask_image"
# os.makedirs("demo", exist_ok=True)
if __name__ == "__main__":
    for image_file in os.listdir(mask_folder):
        image_name = os.path.join(image_folder, image_file.replace(".npy", ".jpg"))
        mask_name = os.path.join(mask_folder, image_file)
        image = cv2.imread(image_name, cv2.IMREAD_UNCHANGED)
        mask_image = np.load(mask_name)
        mask_image[:, :, 0] = 0
        mask_image[:, :, 1] = 0
        img = cv2.resize(image, (int(image.shape[1] * 0.8), int(image.shape[0] * 0.8)))
        mask_image = cv2.resize(mask_image, (int(mask_image.shape[1] * 0.8), int(mask_image.shape[0] * 0.8)))
        
        img += mask_image

        # mask_image = mask_image / 40.0
        # mask_image = (mask_image == 1).astype(np.uint8) * 255

        print(os.path.join("demo", image_file))
        print(mask_image.shape)
        cv2.imwrite(os.path.join("demo", image_file.replace(".npy", ".jpg")), img)
        cv2.imwrite(os.path.join("demo", image_file.replace(".npy", "mask_.jpg")), mask_image)





