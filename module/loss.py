import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from utils.utils import bbox_iou, de_parallel
import math


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


def smooth_BCE(eps=0.1):
    return 1.0 - 0.5 * eps, 0.5 * eps


class ComputeLoss(pl.LightningModule):
    def __init__(self, model, autobalance=False, last_layer=None):
        super(ComputeLoss, self).__init__()
        device = next(model.parameters()).device
        self.gr = 1.0
        BCEcls = nn.BCEWithLogitsLoss()
        BCEobj = nn.BCEWithLogitsLoss()

        self.cp, self.cn = smooth_BCE(eps=0.3)  # positive, negative BCE targets

        # Focal loss
        g = 0
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)
        # m = nn.ModuleList().modules()[-1]
        # det = de_parallel(model).model[-1]
        if last_layer is None:
            det = model[-1]
        else:
            det = last_layer
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.autobalance = BCEcls, BCEobj, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):
        device = targets[0].device if isinstance(targets, list) else targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)

        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets
        targets = targets
        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions

            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx

            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets

            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                iou = iou.to(device)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=pi.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    # t[t==self.cp] = iou.detach().clamp(0).type(t.dtype)

                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= 0.05
        lobj *= 0.7
        lcls *= 0.5
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()

    def build_targets(self, p, targets):
        targets = targets.to(self.device)
        self.anchors = self.anchors.to(targets.device)
        na, nt = self.na, targets.shape[0]
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device).long()
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)
        tt = targets.repeat(na, 1, 1)
        targets = torch.cat((tt, ai[:, :, None]), 2)
        g = 0.5
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],
                            ], device=targets.device).float() * g
        for i in range(self.nl):
            anchors = self.anchors[i]

            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]

            # Match targets to anchors
            t = targets * gain

            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]
                j = torch.max(r, 1. / r).max(2)[0] < 4

                t = t[j]

                # Offsets
                gxy = t[:, 2:4]
                gxi = gain[[2, 3]] - gxy
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh

            gij = (gxy - offsets).long()
            # print('GIJ : ', gij.T)
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()

            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
