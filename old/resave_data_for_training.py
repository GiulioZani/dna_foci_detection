from glob import glob
import numpy as np
from tifffile import imread
import h5py
from os.path import split
import os
from scipy.ndimage import zoom
from scipy.ndimage import binary_dilation
from scipy.ndimage import binary_erosion
from scipy.ndimage import binary_closing
from skimage.morphology import area_opening
from skimage.morphology import remove_small_objects
import matplotlib.pyplot as plt
import ipdb
import sys
import json
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import cv2 as cv
from tqdm import tqdm

sys.path.insert(0, "../utils")
from utils.utils import norm_percentile_nocrop
from utils.check_outsiders import check_outsiders
from utils.get_overlaped_points import get_overlaped_points


# src_path = '../../data_zenodo/part2'
src_path = "/mnt/train"
# dst_hdf5_file = "../../data_zenodo/part2_resaved/foci_detection.hdf5"
dst_hdf5_file = "/mnt/foci_detection.hdf5"


# img_filenames = glob(src_path + "/**/data_53BP1.tif", recursive=True)
img_filenames = [
    os.path.join(src_path, p)
    for p in os.listdir(src_path)
    if "data" in p and not p.endswith("xlsx")
]  # glob(src_path + "/**/data_53BP1.tif", recursive=True)
# print(f"{json.dumps(img_filenames, indent=4)}")

# resized_img_size = [505, 681, 48]  # image is resized to this size
resized_img_size = [505, 681]  # image is resized to this size

normalization_percentile = (
    0.0001  # image is normalized into this percentile range
)


if not os.path.exists(split(dst_hdf5_file)[0]):
    os.makedirs(split(dst_hdf5_file)[0])


def get_foci_position(mask):
    foci = []
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] != 0:
                foci.append((i, j))
    return foci


if os.path.exists(dst_hdf5_file):
    os.remove(dst_hdf5_file)

with h5py.File(dst_hdf5_file, "w") as hdf5:

    for file_num, img_filename in enumerate(tqdm(img_filenames)):
        print(f"{img_filename=}")

        # if file_num<47:
        #     continue

        img = imread(img_filename).mean(0)
        # img.append(imread(img_filename.replace("53BP1", "gH2AX")))
        # img.append(imread(img_filename.replace("53BP1", "DAPI")))

        img_orig_size = img.shape[:-1]
        factor = np.array(resized_img_size) / np.array(img_orig_size)
        # factor[-1] = 0
        img_resized = np.zeros(
            (resized_img_size[0], resized_img_size[1], img.shape[-1]),
            dtype=np.float32,
        )
        for channel in range(img.shape[-1]):
            data_one_channel = img[:, :, channel].squeeze()
            data_one_channel = zoom(data_one_channel, factor, order=1)
            data_one_channel = norm_percentile_nocrop(
                data_one_channel, normalization_percentile
            )
            img_resized[..., channel] = data_one_channel

        img = img_resized
        mask_img = (
            imread(img_filename.replace("data", "mask"))
            .transpose(2, 0, 1)
            .max(0)
        )
        foci_positions = (
            (np.array(get_foci_position(mask_img)) / 2).round().astype(int)
        )
        mask = np.zeros(resized_img_size)
        for pos in foci_positions:
            mask = cv.circle(
                mask,
                (pos[1], pos[0]),
                radius=2,
                color=(255, 255, 255),
                thickness=-1,
            )
        if False:
            fig, ax = plt.subplots(nrows=1, ncols=2)
            ax[1].imshow(mask)
            norm_img = (img_resized - img_resized.min()) / (
                img_resized.max() - img_resized.min()
            )
            ax[0].imshow(norm_img)
            # plt.imshow(norm_img)
            plt.show()
        # ipdb.set_trace()
        """
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = fig.gca()
        ax.axis('off')
        #ax.margins(0)
        #fig.tight_layout(pad=0)
        zero_mask = np.zeros(resized_img_size)
        ax.imshow(zero_mask)
        ax.scatter(foci_positions[1], foci_positions[0])
        canvas.draw()
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        image_from_plot = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        ipdb.set_trace()
        """
        if False:
            norm_img = (img_resized - img_resized.min()) / (
                img_resized.max() - img_resized.min()
            )
            plt.imshow(norm_img)
            plt.scatter(foci_positions[1], foci_positions[0])
            plt.show()
        # mask_resized = zoom(mask_img, factor, order=2)
        """
        with open(
            img_filename.replace("data_53BP1.tif", "labels.json"), "r"
        ) as f:
            lbls = json.load(f)
        lbls["points_53BP1_gH2AX_overlap"] = get_overlaped_points(
            lbls["points_53BP1"], lbls["points_gH2AX"]
        )

        mask_resize_faktor = np.array(img_orig_size) / np.array(
            resized_img_size
        )

        mask = []

        for key in lbls.keys():
            mask_tmp = np.zeros(resized_img_size, dtype=bool)
            points = np.array(lbls[key]) - 1
            points = points[:, [1, 0, 2]]

            points = np.round(points / mask_resize_faktor).astype(np.int32)
            points = check_outsiders(points, resized_img_size)

            mask_tmp[tuple(points.T)] = True
            mask.append(mask_tmp)

        mask = np.stack(mask, axis=3)
        """
        name = os.path.basename(img_filename).split(".")[0]
        hdf5.create_dataset(
            name + "mask",
            data=mask,
            chunks=(128, 128),
            compression="gzip",
            compression_opts=2,
        )
        hdf5.create_dataset(
            name + "data",
            data=img,
            chunks=(128, 128, 3),
            compression="gzip",
            compression_opts=2,
        )
