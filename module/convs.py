import torch as T
import torch.nn as nn
import yaml


class Conv(nn.Module):
    def __init__(self, in_c: int, out_c: int, act: bool = True, batch: bool = False, **kwargs):
        super(Conv, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.act = act
        self.batch = batch
        self.conv = nn.Conv2d(in_c, out_c, **kwargs)
        self.r = nn.LeakyReLU(0.1)
        self.n = nn.BatchNorm2d(out_c)

    def forward(self, x) -> T.Tensor:
        x = self.conv(x)
        if self.batch:
            x = self.n(x)
        if self.act:
            x = self.r(x)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_c, time: int = 4, use_residual: bool = False):
        super(ResidualBlock, self).__init__()
        self.use_residual = use_residual
        self.time = time
        self.layer = nn.ModuleList()

        for _ in range(time):
            self.layer.append(
                nn.Sequential(
                    Conv(in_c, in_c * 2, act=True, batch=True, stride=1, padding=0, kernel_size=1),
                    Conv(in_c * 2, in_c, act=True, batch=True, stride=1, padding=1, kernel_size=3)
                )
            )

    def forward(self, x):
        c = x
        for layer in self.layer:
            x = layer(x)
        return x + c if self.use_residual else x


class Detect(nn.Module):
    def __init__(self, in_c, nc):
        super(Detect, self).__init__()
        self.nc = nc
        self.layer = nn.Sequential(
            Conv(in_c=in_c, out_c=in_c * 2, act=True, batch=True, kernel_size=1),
            Conv(in_c=in_c * 2, out_c=(5 + self.nc), kernel_size=1, batch=False, padding=0, stride=1, act=True)
        )

    def forward(self, x):
        return self.layer(x)


class Connect(nn.Module):
    def __init__(self, s, e, d):
        super(Connect, self).__init__()
        self.s = s
        self.e = e
        self.d = d

    def forward(self, r, x):
        return T.concat((x, r[self.s]), dim=self.d)


class ObjD(nn.Module):
    def __init__(self, nc: int = 4, cfg_path: str = None):
        super(ObjD, self).__init__()
        self.nc = nc
        with open(cfg_path, 'r') as r:
            self.cfg = yaml.full_load(r)

        # for idx, l in enumerate(self.cfg):
        #     print(f"Layer Index : {idx} | layer : {l[0]}")
        self.layers = self.layer_creator()

    def layer_creator(self):
        layers = nn.ModuleList()
        for cfg in self.cfg:
            if cfg[0] == 'Conv':
                layers.append(
                    Conv(in_c=cfg[1], out_c=cfg[2], act=cfg[3], batch=cfg[4], kernel_size=cfg[5], stride=cfg[6],
                         padding=cfg[7]))
            if cfg[0] == 'ResidualBlock':
                layers.append(
                    ResidualBlock(in_c=cfg[1], time=cfg[2], use_residual=cfg[3])
                )
            if cfg[0] == 'Detect':
                layers.append(
                    Detect(in_c=cfg[1], nc=cfg[2])
                )
            if cfg[0] == 'Connect':
                layers.append(
                    Connect(s=cfg[1][0], e=cfg[1][1], d=cfg[1][2])
                )
            if cfg[0] == 'UpSample':
                layers.append(
                    nn.Upsample(cfg[1])
                )
        return layers

    def forward(self, x):
        residual_l = []
        bpm = None
        dtt = []
        vi = 0
        for layer in self.layers:

            if not isinstance(layer, Connect) and not isinstance(layer, nn.Upsample) and not isinstance(layer, Detect):
                x = layer(x)
                print('run')
            if isinstance(layer, Connect):
                print(f'Trying to pair x : {x.shape} to residual {residual_l[vi].shape}')
                bpm = layer(residual_l, x)
                vi += 1
            if isinstance(layer, ResidualBlock):
                residual_l.append(x)
            if isinstance(layer, Detect):
                dtt.append(layer(x))
        return x, dtt
