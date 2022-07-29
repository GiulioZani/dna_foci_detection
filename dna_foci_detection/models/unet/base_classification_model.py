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
        (label_img[:, :, 0] >= label_img[:, :, 0].max()) & (label_img[:, :, 1] > 0)
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


def draw_label(raw_label, img, color=(1, 1, 1)):
    max_radius = 0.01544943820224719
    min_radius = 0.0014044943820224719
    label = raw_label.clone()
    label = label[label[:, 0] > 0, 1:]
    label[:, -1] = (label[:, -1] * (max_radius - min_radius) + min_radius) * img.shape[
        0
    ]
    label = t.round(label).int().tolist()
    img = np.array(img.tolist())
    for x, y, r in label:
        img = cv.circle(img, (x, y), radius=r, color=color, thickness=1)
    return img

def sharpen(label):
    kernel = np.array([
              [0, -1, 0],
               [-1, 5,-1],
               [0, -1, 0]])
    sharpened_label = cv.filter2D(src=label[0].cpu().numpy(), ddepth=-1, kernel=kernel)
    sharpened_label = cv.filter2D(src=sharpened_label, ddepth=-1, kernel=kernel)
    return sharpened_label


def visualize_predictions(
    imgs, labels, pred_labels, save_path: str, epoch=-1, plot=False
):
    # fig, axes = plt.subplots(1, 2)
    # axes[0].imshow(img.permute(1, 2, 0))
    # axes[1].imshow(mask_from_label(labels))
    # ipdb.set_trace()
    indices = [i for i in range(len(labels)) if (labels[i] > 0).any()][:2]
    _, axes = plt.subplots(nrows=3, ncols=len(indices))
    for index, i in enumerate(indices):
        img = imgs[i]
        label = labels[i]
        pred_label = pred_labels[i]
        img = img.permute(1, 2, 0)
        # pred_labels = pred_labels[pred_labels[:, 0] >= 0.5]
        # print(f"{img.shape=}")
        # print(f"{img.shape=}")
        img = t.from_numpy(
            draw_label_img(pred_label.cpu(), img.cpu().numpy(), color=(1, 0, 1))
        )
        img = t.from_numpy(draw_label_img(label.cpu(), img.cpu().numpy(), color=(0, 1, 0)))
        axes[0][index].imshow(img)
        #sharpened_label = sharpen(label)
        sharpened_label = label[0].cpu()
        axes[1][index].imshow(sharpened_label)
        #sharpened_pred_label = sharpen(pred_label)
        sharpened_pred_label = pred_label[0].detach().cpu()
        axes[2][index].imshow(sharpened_pred_label)

        # axes[1][index]
    plt.title(f"Epoch {epoch}")
    plt.savefig(save_path, dpi=300) if not plot else plt.show()
    plt.clf()
    plt.close()


class Model(LightningModule):
    def __init__(self, params: Namespace):
        super().__init__()
        self.params = params
        self.classifier: t.nn.Module
        self.criterion = t.nn.BCELoss()
        self.loss = lambda x, y: self.criterion(x.flatten(), y.flatten())

    def forward(self, z: t.Tensor) -> t.Tensor:
        out = self.classifier(z)
        return out

    def training_step(self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int):
        x, labels = batch
        y_pred = self.classifier(x)
        loss = self.loss(y_pred, labels)
        if batch_idx == 0:
            print(f"{y_pred.max().item()=} {y_pred.min().item()=}")
            visualize_predictions(
                x,
                labels,
                y_pred,
                os.path.join(self.params.save_path, "train_pred.png"),
                epoch=self.current_epoch,
            )
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        avg_loss = t.stack([x["loss"] for x in outputs]).mean()
        self.log("train_loss", avg_loss, prog_bar=True)

    def validation_step(self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int):
        x, labels = batch
        y_pred = self.classifier(x)
        loss = self.loss(y_pred, labels)
        if batch_idx == 0:
            visualize_predictions(
                x,
                labels,
                y_pred,
                os.path.join(self.params.save_path, "pred.png"),
                epoch=self.current_epoch,
            )
        return {
            "val_loss": loss,
        }

    def validation_epoch_end(self, outputs):
        avg_loss = t.stack([x["val_loss"] for x in outputs]).mean()
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
                os.path.join(self.params.save_path, "pred.png"),
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
        return mate.Optimizer(self.params.configure_optimizers, self.classifier)()
