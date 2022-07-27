from argparse import Namespace
from .base_classification_model import BaseClassificationModel
import torch as t
from .resnet import ResNet18


class Model(BaseClassificationModel):
    def __init__(self, params: Namespace):
        super().__init__(params)
        self.classifier = ResNet18(params.out_len)
