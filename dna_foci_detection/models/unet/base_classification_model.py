from pytorch_lightning import LightningModule
import torch as t
import torch.nn.functional as F
from argparse import Namespace

import matplotlib.pyplot as plt
import ipdb
import os
import mate
import numpy as np
import cv2 as cv


def fix_foci(img):
    img = img.copy()
    img[img < 1] = 0
    visited = set()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] > 0:
                img[i, j] = 1
                visited.add((i, j))
                for x, y in [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]:
                    if (x, y) in visited:
                        continue
                    if (
                        x < 0
                        or y < 0
                        or x >= img.shape[0]
                        or y >= img.shape[1]
                    ):
                        continue
                    img[x, y] = 1
                    visited.add((x, y))
    return img


"""
def draw_label_img(label_img, img, color=(1, 0, 1)):
    label_img = t.tensor(label_img.tolist())
    if label_img.shape[0] == 2:
        label_img = label_img.permute(1, 2, 0)
    if img.shape[0] == 3:
        img = img.permute(1, 2, 0)
    assert (
        label_img.shape[:-1] == img.shape[:-1]
    ), f"{label_img.shape=} not compatible with {img.shape=}"
    indices = t.where(
        (label_img[:, :, 0] >= label_img[:, :, 0].max())
        & (label_img[:, :, 1] > 0)
    )  # confidence > 0.8 and radius > 0
    indices = t.tensor([indices[0].tolist(), indices[1].tolist()]).T
    labels = []
    for x, y in indices:
        labels.append([label_img[x, y, 0], x, y, label_img[x, y, 1]])
    labels = t.tensor(labels)
    # if len(labels) > 0:
    #    new_img = draw_label(labels, img, color)
    #    plt.imshow(new_img)
    #    plt.show()
    return draw_label(labels, img, color) if len(labels) > 0 else img
"""


def max_distance(coords, debug=False):
    max_x = t.max(coords[0])
    max_y = t.max(coords[1])
    min_x = t.min(coords[0])
    min_y = t.min(coords[1])
    return np.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2)


def draw_label_img(label_img, img, color=(1, 0, 1)):
    label_img = label_img.squeeze(0)
    values = np.unique(label_img)
    labels = []
    for value in values:
        if value == 0:
            continue
        value_labels = []
        labels.append(value_labels)
        for i in range(label_img.shape[0]):
            for j in range(label_img.shape[1]):
                if label_img[i, j] == value:
                    value_labels.append([i, j, 1])
    foci = []
    for label in labels:
        label = t.tensor(label).T.float()
        label_means = t.mean(label, dim=1)
        r = max_distance(label) / 2
        if r > 0:
            label_means[-1] = r
            foci.append([label_means[0], label_means[1], label_means[2]])
    labels = t.tensor(foci)
    if len(labels) > 0:
        return draw_label(labels, img, color)
    return img


def draw_label(raw_label, img, color=(1, 1, 1)):
    label = raw_label.clone()
    label = t.round(label).int().tolist()
    img = np.array(img.tolist())
    for x, y, r in label:
        img = cv.circle(img, (y, x), radius=r, color=color, thickness=1)
    return img


"""
def sharpen(img):
    # kernel = np.array([[-1,-1,-1, -1, -1], [-1, -1, 15, -1, -1], [-1, -1, -1, -1, -1]])
    size = 3
    # kernel = np.zeros((size, size)) - 1
    # kernel[size // 2, size //2] = int(size**2 - 1)
    # img = cv.filter2D(
    #    src=img, ddepth=-1, kernel=kernel
    # )
    size = 17
    kernel = np.zeros((size, size)) - 1
    kernel[size // 2, size // 2] = int(size ** 2 - 1)
    img = cv.filter2D(src=img, ddepth=-1, kernel=kernel)

    # img = cv.filter2D(
    #    src=img, ddepth=-1, kernel=kernel
    # )
    # img = cv.filter2D(
    #    src=img, ddepth=-1, kernel=kernel
    # )
    # sharpened_label = cv.filter2D(
    #    src=sharpened_label, ddepth=-1, kernel=kernel
    # )
    # img = np.clip(img, 0, img.max())
    return (img - img.min()) / (img.max() - img.min())
"""

"""
def sharpen(img):

    # img = img.img_to_array(img, dtype="uint8")
    color_img = cv.cvtColor(img, cv.COLOR_GRAY2RGB).astype(np.uint8)
    img = img.astype(np.uint8)
    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel, iterations=2)
    ipdb.set_trace()
    # sure background area
    sure_bg = cv.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    ret, sure_fg = cv.threshold(
        dist_transform, 0.7 * dist_transform.max(), 255, 0
    )

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)
    # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    markers = cv.watershed(color_img, markers)
    color_img[markers == -1] = [255, 0, 0]
    ipdb.set_trace()
    return color_img
"""


def sharpen(image):
    from scipy import ndimage as ndi
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max

    # Now we want to separate the two objects in image
    # Generate the markers as local maxima of the distance to the background
    image = image.astype(int)
    distance = ndi.distance_transform_edt(image)
    coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=image)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=image)
    """
    plt.close()
    plt.clf()
    fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
    ax = axes.ravel()
    ax[0].imshow(image, cmap=plt.cm.gray)
    ax[0].set_title("Overlapping objects")
    ax[1].imshow(-distance, cmap=plt.cm.gray)
    ax[1].set_title("Distances")
    ax[2].imshow(labels, cmap=plt.cm.nipy_spectral)
    ax[2].set_title("Separated objects")
    plt.show()
    """
    return t.from_numpy(labels)


def visualize_predictions(
    imgs, labels, pred_labels, save_path: str, epoch=-1, plot=False
):
    # fig, axes = plt.subplots(1, 2)
    # axes[0].imshow(img.permute(1, 2, 0))
    # axes[1].imshow(mask_from_label(labels))
    # ipdb.set_trace()
    indices = [i for i in range(len(labels)) if (labels[i] > 0).any()][:2]
    if len(indices) < 2:
        return
    fig, axes = plt.subplots(nrows=1, ncols=len(indices))
    for index, i in enumerate(indices):
        img = imgs[i]
        label = labels[i]
        pred_label = pred_labels[i]
        img = img.permute(1, 2, 0)
        # pred_labels = pred_labels[pred_labels[:, 0] >= 0.5]
        # print(f"{img.shape=}")
        # print(f"{img.shape=}")
        sharpened_label = label[0].detach().cpu().numpy()
        sharpened_label = sharpen(sharpened_label)

        sharpened_disc_label = (
            (pred_label[0].detach().cpu() > 0.5).float().numpy()
        )
        sharpened_pred_label = sharpen(sharpened_disc_label)
        img = t.from_numpy(
            draw_label_img(
                sharpened_pred_label.cpu(), img.cpu().numpy(), color=(1, 0, 1)
            )
        )
        img = t.from_numpy(
            draw_label_img(
                sharpened_label.cpu(), img.cpu().numpy(), color=(0, 1, 0)
            )
        )
        # ax = axes[0][index]
        fig = axes[index].imshow(img)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        axes[index].set_title(f"Figure")
        """

        ax = axes[1][index]
        fig = ax.imshow(sharpened_disc_label)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        ax.set_title(f"Disc")

        ax = axes[2][index]
        fig = ax.imshow(sharpened_pred_label)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        ax.set_title(f"Sharpened predicted")

        ax = axes[3][index]
        fig = ax.imshow(sharpened_label)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        ax.set_title(f"Sharpened original")
        """

    plt.axis("off")
    plt.tight_layout()
    plt.suptitle(f"Epoch {epoch}")
    plt.savefig(save_path, dpi=500) if not plot else plt.show()
    plt.clf()
    plt.close()


class Model(LightningModule):
    def __init__(self, params: Namespace):
        super().__init__()
        self.params = params
        self.classifier: t.nn.Module
        self.criterion = t.nn.BCELoss()
        self.loss = lambda x, y: self.criterion(x.flatten(), y.flatten())
        self.best_loss = float("inf")

    def forward(self, z: t.Tensor) -> t.Tensor:
        out = self.classifier(z)
        return out

    def training_step(self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int):
        x, labels = batch
        y_pred = self.classifier(x)
        loss = self.loss(y_pred, labels)
        if batch_idx == 0:
            print(f"{y_pred.max().item()=} {y_pred.min().item()=}")
            """
            visualize_predictions(
                x,
                labels,
                y_pred,
                os.path.join(self.params.save_path, "train_pred.png"),
                epoch=self.current_epoch,
            )
            print("visualized!!")
            """
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        avg_loss = t.stack([x["loss"] for x in outputs]).mean()
        self.log("train_loss", avg_loss, prog_bar=True)

    def validation_step(
        self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int
    ):
        x, labels = batch
        y_pred = self.classifier(x)
        loss = self.loss(y_pred, labels)
        if batch_idx == 0:
            visualize_predictions(
                x,
                labels,
                y_pred,
                os.path.join(self.params.save_path, "val_pred.png"),
                epoch=self.current_epoch,
            )
        return {
            "val_loss": loss,
        }

    def validation_epoch_end(self, outputs):
        avg_loss = t.stack([x["val_loss"] for x in outputs]).mean()
        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            print("Model reached new best. Saving it.")
            for torch_model_name in self.params.model:
                with open(
                    os.path.join(
                        self.params.save_path,
                        "checkpoint",
                        torch_model_name + ".pt",
                    ),
                    "wb",
                ) as f:
                    t.save(self.__getattr__(torch_model_name).state_dict(), f)
        self.log("val_loss", avg_loss, prog_bar=True)
        return {"val_loss": avg_loss}

    def test_step(self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int):

        x, labels = batch
        y_pred = self.classifier(x)
        loss = self.loss(y_pred, labels)
        if batch_idx == 0:
            visualize_predictions(
                x,
                labels,
                y_pred,
                os.path.join(self.params.save_path, "test_pred.png"),
                epoch=self.current_epoch,
                plot=True,
            )
        return {
            "test_loss": loss,
        }

    def test_epoch_end(self, outputs):
        avg_loss = t.stack([x["test_loss"] for x in outputs]).mean()
        return {"test_loss": avg_loss}

    def configure_optimizers(self):
        return mate.Optimizer(
            self.params.configure_optimizers, self.classifier
        )()
