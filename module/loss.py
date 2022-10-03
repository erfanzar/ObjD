import numpy as np
import torch
from utils.utils import intersection_over_union
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


def iou(box1, box2):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3],
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3],

    intersect_x1 = torch.max(b1_x1, b2_x1)
    intersect_y1 = torch.max(b1_y1, b2_y1)
    intersect_x2 = torch.min(b1_x2, b2_x2)
    intersect_y2 = torch.min(b1_y2, b2_y2)

    intersect_area = (intersect_x2 - intersect_x1 + 1) * (intersect_y2 - intersect_y1 + 1)

    # union area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    return intersect_area / (b1_area + b2_area - intersect_area + 1e-16)


class Loss(pl.LightningModule):
    def __init__(self):
        super(Loss, self).__init__()
        self.ac = 1e-16
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.ca = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        device = x.device

        obj = y[..., 0] == 1
        no_obj = y[..., 0] == 0
        nol = self.bce(
            (x[..., 0:1][no_obj]).to(device), (y[..., 0:1][no_obj].to(device))
        )

        box_pred = torch.cat(
            [self.sigmoid(x[..., 1:3]).to(device), torch.exp(x[..., 3:5]).to(device)],
            dim=-1)
        iou_s = intersection_over_union(box_pred[obj].to(device), y[..., 1:5][obj].to(device)).detach()

        ol = self.mse(
            self.sigmoid(x[..., 0:1][obj]).to(device), (iou_s.to(device) * y[..., 0:1][obj].to(device))
        )

        x[..., 1:3] = self.sigmoid(x[..., 1:3])
        y[..., 3:5] = torch.log(
            (1e-16 + y[..., 3:5])
        )

        box_loss = self.mse(x[..., 1:5][obj], y[..., 1:5][obj].to(device))

        class_loss = self.bce(
            (F.softmax(x[..., 5:][obj], dim=-1).to(device)), (y[..., 5:][obj].float().to(device))
        )

        loss = nol * 1 + box_loss * 1 + ol * 1 + class_loss * 1

        return loss
