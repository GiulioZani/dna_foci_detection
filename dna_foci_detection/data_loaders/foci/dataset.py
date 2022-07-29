import numpy as np
from torch.utils import data
import torch as t
import matplotlib.pyplot as plt
import h5py
from .utils.utils import mat2gray_nocrop, plot_img_with_labels
import os
import random
import ipdb
from .visualize_predictions import draw_label_img
from scipy.ndimage import gaussian_filter
import monai


def augmentation(img, label_img):
    """
    r = [torch_randint(2), torch_randint(2), torch_randint(4)]
    if r[0]:
        img = np.fliplr(img)
        mask = np.fliplr(mask)
    if r[1]:
        img = np.flipud(img)
        mask = np.flipud(mask)
    img = np.rot90(img, k=r[2])
    mask = np.rot90(mask, k=r[2])
    img = img.numpy()
    min_v = (torch_rand() * 0.96) - 0.48
    max_v = 1 + (torch_rand() * 0.96) - 0.48
    for k in range(img.shape[2]):

        img[:, :, k] = mat2gray_nocrop(img[:, :, k], [min_v, max_v]) - 0.5

    """
    transforms = monai.transforms.Compose(
        monai.transforms.RandAxisFlipd(keys=["image", "mask"], prob=0.5),
        monai.transforms.RandRotate90d(keys=["image", "mask"], prob=0.5),
        monai.transforms.RandGridDistortiond(
            keys=["image", "mask"], prob=0.5, distort_limit=0.2
        ),
        monai.transforms.OneOf(
            [
                monai.transforms.RandShiftIntensityd(
                    keys=["image"], prob=0.5, offsets=(0.1, 0.2)
                ),
                monai.transforms.RandAdjustContrastd(
                    keys=["image"], prob=0.5, gamma=(1.5, 2.5)
                ),
                monai.transforms.RandHistogramShiftd(keys=["image"], prob=0.5),
            ]
        ),
    )
    uba = transforms(dict(image=img, mask=label_img))
    img = uba["image"]
    label_img = uba["mask"]
    """
    if t.rand(1)[0] > 0.5:
        img = t.flipud(img)
        label_img = t.flipud(label_img)
    if t.rand(1)[0] > 0.5:
        img = t.fliplr(img)
        label_img = t.fliplr(label_img)
    if t.rand(1)[0] > 0.5:
        '''
        img = img + t.randn(img.shape) * 0.05
        img = t.clamp(img, 0, 1)
        '''
    if t.rand(1)[0] > 0.5:
        times = t.randint(4, (1,))[0]
        img = t.rot90(img, k=times)
        label_img = t.rot90(label_img, k=times)
    """
    return img, label_img


class FociDataset(data.Dataset):
    def __init__(
        self,
        *,
        hdf5_filename: str,
        filenames: tuple[str, ...],
        split: str,
        crop_size: tuple[int, int],
        out_len: int,
    ):

        self.hdf5_filename = hdf5_filename
        self.split = split
        self.crop_size = crop_size
        self.filenames = filenames
        self.out_len = out_len
        self.h5data = None

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        if self.h5data is None:
            self.h5data = h5py.File(self.hdf5_filename, "r")

        filename = self.filenames[idx]
        img = t.from_numpy(self.h5data[filename + "_image"][...]).permute(
            1, 2, 0
        )
        label = t.from_numpy(self.h5data[filename + "_label"][...]).permute(
            1, 2, 0
        )
        if not self.split == "test":
            in_size = img.shape
            out_size = self.crop_size
            r = [
                t.randint(in_size[i] - out_size[i], (1,))[0] for i in range(2)
            ]
            img = img[
                r[0] : r[0] + out_size[0], r[1] : r[1] + out_size[1], :,
            ]
            label = label[
                r[1] : r[1] + out_size[0],
                r[0] : r[0] + out_size[1],
                :,  # TODO: check this. Why do I need to swap the order of the two indices???
            ]
            label[:, :, 0] = t.from_numpy(
                gaussian_filter(label[:, :, 0].numpy(), sigma=[2, 2])
                * 59.5238
                * 10
            )
            # draw_label_img(label, img)
            """
            x_old, y_old = label[:, 1:-1].T
            x = t.clip(x_old * 292 - r[1], min=0)
            y = t.clip(y_old * 292 - r[0], min=0)
            filter_indices = (x > out_size[0]) | (y > out_size[1])
            x[filter_indices] = 0
            y[filter_indices] = 0
            label[filter_indices, 0] = 0
            x /= out_size[0]
            y /= out_size[1]
            label[:, 1] = x
            label[:, 2] = y
            """
        # nonzero_labels = label[label[:, 0] != 0]
        # label_img = t.zeros(img.shape[0], img.shape[1], 2)
        """
        for nonzero_label in nonzero_labels:
            label_img[
                int(img.shape[0] * nonzero_label[1]),
                int(img.shape[1] * nonzero_label[2]),
                0,
            ] = 1
            label_img[
                int(img.shape[0] * nonzero_label[1]),
                int(img.shape[1] * nonzero_label[2]),
                1,
            ] = nonzero_label[-1]
        _, indices = t.sort(
            label[:, -1], descending=True
        )  # sort them in descending order of radius
        label = label[indices]
        """
        if self.split == "train":
            img, label = augmentation(img, label)
        img = img.permute(2, 0, 1)
        label = label.permute(2, 0, 1)
        return img, label
