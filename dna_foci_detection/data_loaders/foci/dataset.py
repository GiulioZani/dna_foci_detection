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


def torch_randint(max_v):
    return t.randint(max_v, (1, 1)).view(-1).numpy()[0]


def torch_rand(size=1):
    return t.rand(size).numpy()


def augmentation(img, mask):
    img = img.numpy()
    mask = mask.numpy()
    r = [torch_randint(2), torch_randint(2), torch_randint(4)]
    if r[0]:
        img = np.fliplr(img)
        mask = np.fliplr(mask)
    if r[1]:
        img = np.flipud(img)
        mask = np.flipud(mask)
    img = np.rot90(img, k=r[2])
    mask = np.rot90(mask, k=r[2])

    # min_v = (torch_rand() * 0.96) - 0.48
    # max_v = 1 + (torch_rand() * 0.96) - 0.48
    # for k in range(img.shape[-1]):
    #    img[:, :, k] = mat2gray_nocrop(img[:, :, k], [min_v, max_v]) - 0.5

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
    """
    tmp_img = t.cat([img, label_img], dim=-1)

    transforms = monai.transforms.Compose(
        monai.transforms.RandAxisFlipd(keys=["image"], prob=0.5),
        monai.transforms.RandRotate90d(keys=["image"], prob=0.5),
        monai.transforms.RandGridDistortiond(
            keys=["image"], prob=0.5, distort_limit=0.2
        ),
        monai.transforms.OneOf(
            [
                monai.transforms.RandShiftIntensityd(
                    keys=["image"], prob=0.5, offsets=(0.1, 0.2)
                ),
                # monai.transforms.RandAdjustContrastd(
                #    keys=["image"], prob=0.5, gamma=(1.5, 2.5)
                # ),
                # monai.transforms.RandHistogramShiftd(keys=["image"], prob=0.5),
            ]
        ),
    )
    uba = transforms(dict(image=tmp_img))
    img = uba["image"][:, :, :3]
    # print(f"{img.shape=}")
    label_img = uba["image"][:, :, -2:]
    # print(f"{label_img.shape=}")
    """
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
    return t.from_numpy(img.copy()), t.from_numpy(mask.copy())


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
        in_size = img.shape
        out_size = self.crop_size
        if not self.split == "test":
            r = [
                t.randint(in_size[i] - out_size[i], (1,))[0] for i in range(2)
            ]
            img = img[
                r[0] : r[0] + out_size[0], r[1] : r[1] + out_size[1], :,
            ]
            label = label[
                r[0] : r[0] + out_size[0],
                r[1] : r[1] + out_size[1],
                :,  # TODO: check this. Why do I need to swap the order of the two indices???
            ]
        if self.split == "train":
            img, label = augmentation(img, label)

        img = img.permute(2, 0, 1).float()
        label = label.permute(2, 0, 1).float()

        return img, label
