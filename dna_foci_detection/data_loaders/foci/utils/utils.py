import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import torch as t
import ipdb


def draw_label(label, img):
    label = label.clone()
    label = label[label[:, 0] > 0, 1:]
    label *= img.shape[0]
    label[:, -1] *= 0.01544943820224719
    label = t.round(label).int().tolist()
    img = np.array(img.tolist())
    print(f"{img.shape=}")
    for x, y, r in label:
        print(f"{x, y, r=}")

        img = cv.circle(img, (x, y), radius=r, color=(255, 255, 255), thickness=1,)

    return img


def plot_img_with_labels(img, labels):
    # fig, axes = plt.subplots(1, 2)
    # axes[0].imshow(img.permute(1, 2, 0))
    # axes[1].imshow(mask_from_label(labels))
    img = t.from_numpy(draw_label(labels, img.numpy()))
    plt.imshow(img)
    plt.show()


def norm_percentile_nocrop(data, perc):

    norm = [
        np.percentile(data, perc * 100),
        np.percentile(data, 100 - perc * 100),
    ]
    data = (data - norm[0]) / (norm[1] - norm[0]) - 0.5

    return data


def dice_loss(pred, target):

    smooth = 1.0
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    A_sum = torch.sum(iflat)
    B_sum = torch.sum(tflat)

    return 1 - ((2.0 * intersection + smooth) / (A_sum + B_sum + smooth))


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def mat2gray_nocrop(data, min_max):

    return (data - min_max[0]) / (min_max[1] - min_max[0])


def minmax(img):
    return (img - img.min()) / (img.max() - img.min())


def plot_img_with_mask(img, mask):
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(minmax(img))
    axes[1].imshow(mask)
    plt.show()
