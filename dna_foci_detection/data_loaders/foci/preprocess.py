from glob import glob
import numpy as np
from tifffile import imread
import h5py
from os.path import split
import os
import torch as t
import imageio
import glob
import matplotlib.pyplot as plt
import ipdb
import sys
import json
import cv2 as cv
from tqdm import tqdm


def main():
    sys.path.insert(0, "../utils")
    # from utils.utils import plot_img_with_labels
    # from utils.check_outsiders import check_outsiders
    # from utils.get_overlaped_points import get_overlaped_points
    # from visualize_predictions import draw_label_img

    # src_path = '../../data_zenodo/part2'
    src_path = "/mnt/deep_foci/new_images"
    src_labels = "/mnt/deep_foci/labels"
    # dst_hdf5_file = "../../data_zenodo/part2_resaved/foci_detection.hdf5"
    dst_hdf5_file = "/mnt/foci_detection.hdf5"

    # img_filenames = glob(src_path + "/**/data_53BP1.tif", recursive=True)
    img_filenames = [
        os.path.join(src_path, p)
        for p in os.listdir(src_path)
        if p.endswith("png")
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

    labels = tuple(glob.glob(src_labels + "/*.json"))
    with h5py.File(dst_hdf5_file, "w") as hdf5:
        for file_num, img_filename in enumerate(
            glob.glob(os.path.join(src_path, "*.png"))
        ):
            print(f"{img_filename=}")
            img = (t.from_numpy(imageio.imread(img_filename)) / 255).permute(
                2, 0, 1
            )
            label_file_name = os.path.join(
                src_labels,
                os.path.basename(img_filename.replace(".png", ".json")),
            )
            label = t.zeros(2, img.shape[1], img.shape[2])
            # label = t.zeros(60, 4)
            if os.path.exists(label_file_name):
                raw_data = json.loads(open(label_file_name).read())
                raw_label = [row for row in raw_data if not None in row]
                """
                if len(raw_label) > 0:
                    read_label = t.tensor(raw_label)
                    read_label[:, :-1] /= 292  # max x and y value
                    read_label[
                        :, -1
                    ] /= 0.01544943820224719  # max radius value (its already relative to img size)
                    label[: len(read_label), 1:] = read_label
                    label[: len(read_label), 0] = 1
                    # crop the image by: [20:-20,15:-15]
                    # plot_img_with_labels(img, label)
                """
                max_r = 0.01544943820224719
                min_r = 0.0014044943820224719
                for x, y, r in raw_label:
                    label[0, x, y] = 1
                    label[1, x, y] = (r - min_r) / (max_r - min_r)

                # draw_label_img(label, img)

            hdf5.create_dataset(
                f"{file_num}_label",
                data=label.numpy(),
                # chunks=(1),
                compression="gzip",
                compression_opts=2,
            )
            hdf5.create_dataset(
                f"{file_num}_image",
                data=img.numpy(),
                # chunks=(128, 128, 3),
                compression="gzip",
                compression_opts=2,
            )


if __name__ == "__main__":
    main()
