import torch

import torch.nn as nn


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


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.ac = 1e-16

    def forward(self, x, y):
        tx = y[0, ..., 0]
        ty = y[0, ..., 1]
        tw = y[0, ..., 2]
        th = y[0, ..., 3]

        px = x[0, ..., 0]
        py = x[0, ..., 1]
        pw = x[0, ..., 2]
        ph = x[0, ..., 3]

        bx = nn.Sigmoid()(px) + tx
        by = nn.Sigmoid()(py) + ty
        bw = self.ac * torch.exp(px)
        bh = self.ac * torch.exp(py)
        print(x.shape)

        box1 = x[0:, 0:4]
        box2 = y[0:, 0:4]
        y_loss = torch.zeros(y.shape)
        for v in range(box2.shape[0]):
            box2_t = y[v, 0:4].reshape(1, 4)
            clss = y[v, 5:]
            iou_s = iou(box1=box1, box2=box2_t)

            y_loss[v, 0] = bx
            y_loss[v, 1] = by
            y_loss[v, 2] = bw
            y_loss[v, 3] = bh
            y_loss[v, 5:] = clss

        return y_loss
