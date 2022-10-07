import sys

import torch
import torch.nn as nn
import yaml
from colorama import Fore
from .loss import Loss
from .commons import (Conv, Detect, ResidualBlock, Neck, C3, C4P, MP, UC1, CV1, RepConv, ConvSc, LP)
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from utils.utils import module_creator

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class ObjectDetectorModule(pl.LightningModule):
    def __init__(self, cfg: str = 'cfg/tiny.yaml'):
        super(ObjectDetectorModule, self).__init__()
        self.save = None
        self.model, self.nc, self.cfg, self.fr = None, 1, cfg, False
        self.layer_creator()
        self.to(DEVICE)
        self.loss = Loss()
        self.save_hyperparameters()

    def layer_creator(self):
        with open(self.cfg, 'r') as r:
            data = yaml.full_load(r)
        self.nc = data['nc']
        bone_list, head_list = data['backbone'], data['head']
        self.model, self.save = module_creator(
            bone_list, head_list, False,
            3, )  # backbone list , head list , print Status, image channel backbone and head

    def size(self):
        ps = 0
        for name, pr in self.layers.named_parameters():
            sz = (pr.numel() * torch.finfo(pr.data.dtype).bits) / (1024 * 10000)
            ps += sz
            print("| {:<30} | {:<25} |".format(name, f"{sz} Mb"))
        print('-' * 50)
        print(f' TOTAL SIZE  :  {ps} MB')

    def forward(self, x):
        route = []
        for i, m in enumerate(self.model):
            if m.form != -1:
                x = route[m.form] if isinstance(m.form, int) else [x if j == -1 else route[j] for j in m.form]
            x = m(x)
            route.append(x if i in self.save else None)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-4)
        lr_lambda = lambda epoch: 0.85 * epoch
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda
        )
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_index):
        x, y = batch
        x_ = self(x)
        loss = (self.loss.forward(x_[0], y[0]) +
                self.loss.forward(x_[1], y[1]) +
                self.loss.forward(x_[2], y[2])
                )

        self.log('train_loss', loss, prog_bar=True, on_step=False,
                 on_epoch=True)
        return loss

    def validation_step(self, batch, batch_index):
        x, y = batch
        x_ = self(x)

        loss = (self.loss.forward(x_[0], y[0]) +
                self.loss.forward(x_[1], y[1]) +
                self.loss.forward(x_[2], y[2])
                )
        self.log('val_loss', loss, prog_bar=True, on_step=False,
                 on_epoch=True)

        return loss
