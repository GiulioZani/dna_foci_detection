import torch
import matplotlib.pyplot as plt


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
