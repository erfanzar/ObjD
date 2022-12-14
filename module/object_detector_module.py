import torch
import torch.nn as nn
import yaml

from utils.utils import module_creator
from .loss import ComputeLoss

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class ObjectDetectorModule(nn.Module):
    def __init__(self, cfg: str = 'cfg/objd-s.yaml'):
        super(ObjectDetectorModule, self).__init__()
        self.anchors = None
        self.save = None
        self.m, self.nc, self.cfg, self.fr = None, 1, cfg, False
        self.to(DEVICE)
        self.device = DEVICE

    def init(self):
        self.layer_creator()
        self.m = self.m.to(self.device)
        self.loss = ComputeLoss(self.m)

    def layer_creator(self):
        with open(self.cfg, 'r') as r:
            data = yaml.full_load(r)
        self.nc = data['nc']
        self.anchors = data['anchors']
        bone_list, head_list = data['backbone'], data['head']
        self.m, self.save = module_creator(
            bone_list, head_list, False,
            3, nc=self.nc,
            anchors=self.anchors)  # backbone list , head list , print Status, image channel backbone and head

    def size(self):
        ps = 0
        for name, pr in self.layers.named_parameters():
            sz = (pr.numel() * torch.finfo(pr.data.dtype).bits) / (1024 * 10000)
            ps += sz
            print("| {:<30} | {:<25} |".format(name, f"{sz} Mb"))
        print('-' * 50)
        print(f' TOTAL SIZE  :  {ps} MB')

    def forward(self, x):
        x = x.float()
        route = []

        for i, m in enumerate(self.m):
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
        y = y.view(-1, 6).to(self.device)
        x_ = self(x)
        loss = self.loss(x_, y)

        self.log('lbox', loss[1][0], prog_bar=True, on_step=True)
        self.log('lobj', loss[1][1], prog_bar=True, on_step=True)
        self.log('lcls', loss[1][2], prog_bar=True, on_step=True)
        self.log('loss', loss[1][3], prog_bar=True, on_step=True)
        self.log('train_loss', loss[0])
        return loss[0]

    def validation_step(self, batch, batch_index):
        x, y = batch
        y = y.view(-1, 6).to(self.device)
        x_ = self(x)
        loss = self.loss(x_, y)
        return loss[0]
