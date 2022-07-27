import matplotlib.pyplot as plt
import torch as t
import cv2 as cv
import numpy as np


def draw_label(label, img, color=(255, 255, 255)):
    label = label.clone()
    label = label[label[:, 0] > 0, 1:]
    label *= img.shape[0]
    label[:, -1] *= 0.01544943820224719
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
