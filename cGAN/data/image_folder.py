"""A modified image folder class

We modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""

import torch.utils.data as data
import pandas as pd
from curses.ascii import isdigit

from PIL import Image
import os

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF', '.nii.gz',
]

IMG_NAME_PATTERN = "Delay_{:02d}.nii.gz"
AVG_IMG_NAME = "Delay_avg.nii.gz"


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]


def make_dataset_AB(dir, phase, series_type="_de_", slice_num=150, max_dataset_size=float("inf")):
    image_pairs = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for series_name in sorted(os.listdir(dir)):
        series_img_cnt = int(series_name.split("_")[-2])
        ref_phase = series_img_cnt // 2
        
        img_name = IMG_NAME_PATTERN.format(ref_phase)
        avg_path = os.path.join(dir, series_name, AVG_IMG_NAME)

        img_path = os.path.join(dir, series_name, img_name)
        if not os.path.exists(img_path):
            print("Input image missing")
            continue

        for slice_idx in range(slice_num):
            if slice_idx < 10 or slice_num - slice_idx <= 10:
                continue

            image_pairs.append([img_path, avg_path, slice_idx])

    return image_pairs[:min(max_dataset_size, len(image_pairs))]


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
