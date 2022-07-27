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


def draw_label(label, img, color=(1, 1, 1)):
    #label = label.clone()
    label = label[label[:, 0] > 0, 1:]
    label *= img.shape[0]
    label[:, -1] *= 0.01544943820224719
    label = t.round(label).int().tolist()
    img = np.array(img.tolist())
    for x, y, r in label:
        img = cv.circle(img, (x, y), radius=r, color=color, thickness=1)
    return img


def visualize_predictions(img, labels, pred_labels, save_path:str, epoch=-1):
    # fig, axes = plt.subplots(1, 2)
    # axes[0].imshow(img.permute(1, 2, 0))
    # axes[1].imshow(mask_from_label(labels))
    img = img.permute(1, 2, 0)
    pred_labels = pred_labels[pred_labels[:, 0] >= 0.5]
    #print(f"{img.shape=}")
    img = t.from_numpy(draw_label(labels.cpu(), img.cpu().numpy()))
    #print(f"{img.shape=}")
    img = t.from_numpy(draw_label(pred_labels.cpu(), img.cpu().numpy(), color=(1, 0, 1)))
    plt.title(f"Epoch {epoch}")
    plt.imshow(img)
    plt.savefig(save_path, dpi=300)
    plt.clf()
    plt.close()

class BaseClassificationModel(LightningModule):
    def __init__(self, params: Namespace):
        super().__init__()
        self.params = params
        self.classifier: t.nn.Module
        self.critarion = t.nn.BCELoss()
        self.loss = lambda x, y: self.critarion(x.flatten(), y.flatten())

    def forward(self, z: t.Tensor) -> t.Tensor:
        out = self.classifier(z)
        return out

    def training_step(self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int):
        x, labels = batch
        y_pred = self.classifier(x)
        loss = self.loss(y_pred, labels)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        avg_loss = t.stack([x["loss"] for x in outputs]).mean()
        self.log("train_loss", avg_loss, prog_bar=True)

    def validation_step(self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int):
        x, labels = batch
        y_pred = self.classifier(x)
        loss = self.loss(y_pred, labels)
        if batch_idx == 0:
            visualize_predictions(x[0], labels[0], y_pred[0], os.path.join(self.params.save_path, "pred.png"), epoch=self.current_epoch)
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
        return {
            "test_loss": loss,
        }

    def test_epoch_end(self, outputs):
        avg_loss = t.stack([x["test_loss"] for x in outputs]).mean()

        return {"test_loss": avg_loss}

    def configure_optimizers(self):

        return mate.Optimizer(self.params.configure_optimizers, self.classifier)()
