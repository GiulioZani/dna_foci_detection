import matplotlib.pyplot as plt
import torch as t
import cv2 as cv
import numpy as np
import ipdb

"""
def where(label_img, threshold=0.8):
    return indices
"""


def draw_label_img(label_img, img, color=(1, 0, 1)):
    print(f"{label_img.shape=}")
    label_img = t.tensor(label_img.tolist())
    if label_img.shape[0] == 2:
        label_img = label_img.permute(1, 2, 0)
    if img.shape[0] == 3:
        img = img.permute(1, 2, 0)
    assert (
        label_img.shape[:-1] == img.shape[:-1]
    ), f"{label_img.shape=} not compatible with {img.shape=}"
    indices = t.where(
        label_img > t.tensor([0.8, 0])
    )  # confidence > 0.8 and radius > 0
    indices = t.tensor([indices[0].tolist(), indices[1].tolist()]).T
    labels = []
    for x, y in indices:
        labels.append([label_img[x, y, 0], x, y, label_img[x, y, 1]])
    labels = t.tensor(labels)
    if len(labels) > 0:
        new_img = draw_label(labels, img, color)
        plt.imshow(new_img)
        plt.show()


def draw_label(raw_label, img, color=(1, 1, 1)):
    max_radius = 0.01544943820224719
    min_radius = 0.0014044943820224719
    label = raw_label.clone()
    label = label[label[:, 0] > 0, 1:]
    label[:, -1] = (
        label[:, -1] * (max_radius - min_radius) + min_radius
    ) * img.shape[0]
    label = t.round(label).int().tolist()
    img = np.array(img.tolist())
    print(f"{img.shape=}")
    for x, y, r in label:
        print(f"{x, y, r=}")

        img = cv.circle(img, (x, y), radius=r, color=color, thickness=1,)

    return img


def visualize_predictions(img, labels, pred_labels):
    # fig, axes = plt.subplots(1, 2)
    # axes[0].imshow(img.permute(1, 2, 0))
    # axes[1].imshow(mask_from_label(labels))
    img = t.from_numpy(draw_label(labels.cpu(), img.cpu().numpy()))
    img = t.from_numpy(
        draw_label(pred_labels.cpu(), img.numpy(), color=(255, 0, 0))
    )
    plt.imshow(img)
    plt.show()
