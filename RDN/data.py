import os
import pydicom
import random
import nibabel
import numpy as np
import pandas as pd
import torch.utils.data as data
from torchvision.transforms import ToTensor


def dicom_to_ndarray(file_name):
    im = pydicom.dcmread(file_name).pixel_array
    return im.astype(np.float32)


def getPatch(im, im_gt, patch_size, margin=20):
    h, w = im.shape
    ix = random.randrange(margin, (w - (margin + patch_size)) + 1)
    iy = random.randrange(margin, (h - (margin + patch_size)) + 1)
    im = im[iy: iy+patch_size, ix: ix+patch_size]
    im_gt = im_gt[iy: iy+patch_size, ix: ix+patch_size]
    return im, im_gt


class MDE_Dataset(data.Dataset):
    def __init__(self, args, dataset_type):
        df = pd.read_csv('./csv/{}_imlist.csv'.format(dataset_type))

        if args.phase == "test" and not args.overwrite:
            existing_series = set()
            for series_name in df['series'].to_list():
                result_path = os.path.join(args.result_dir, "{}.nii.gz".format(series_name))
                if os.path.exists(result_path):
                    existing_series.add(series_name)
            df = df[[x not in existing_series for x in df['series'].to_list()]]

        self.phase = args.phase
        self.patch_size = args.patch_size
        self.series_list = df['series'].tolist()
        self.filelist_LQ = df['LQ'].tolist()
        self.filelist_HQ = df['HQ'].tolist()
        self.filelist_slice = df['slice'].tolist()

    def __len__(self):
        return len(self.filelist_HQ)

    def __getitem__(self, idx):
        path_lq, path_hq, series_name, slice = self.filelist_LQ[idx], self.filelist_HQ[idx], self.series_list[idx], self.filelist_slice[idx]
        im = nibabel.load(path_lq)
        im_np = im.get_data()[slice]
        gt = nibabel.load(path_hq)
        gt_np = gt.get_data()[slice]
        if self.phase == 'train':
            im, im_gt = getPatch(im_np, gt_np, self.patch_size)
        else:
            im, im_gt = im_np, gt_np

        return ToTensor()(im), ToTensor()(im_gt), series_name, slice
