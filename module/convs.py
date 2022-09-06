import torch as T
import torch.nn as nn

DEVICE = 'cuda:0' if T.cuda.is_available() else 'cpu'


class Conv(nn.Module):
    def __init__(self, in_c: int, out_c: int, act: bool = True, batch: bool = False, **kwargs):
        super(Conv, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.act = act
        self.batch = batch
        self.to(DEVICE)
        self.conv = nn.Conv2d(in_c, out_c, **kwargs).to(DEVICE)
        self.r = nn.LeakyReLU(0.1).to(DEVICE)
        self.n = nn.BatchNorm2d(out_c).to(DEVICE)

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
        self.to(DEVICE)
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
        self.to(DEVICE)
        self.layer = nn.Sequential(
            Conv(in_c=in_c, out_c=in_c * 2, act=True, batch=True, kernel_size=1),
            Conv(in_c=in_c * 2, out_c=(5 + self.nc), kernel_size=1, batch=False, padding=0, stride=1, act=True)
        )

    def forward(self, x):
        return self.layer(x).permute(0, 2, 3, 1).reshape(1, -1, self.nc + 5)


class Connect(nn.Module):
    def __init__(self, s, d):
        super(Connect, self).__init__()
        self.s = s
        self.d = d
        self.to(DEVICE)

    def forward(self, r, x):
        return T.concat((x, r[self.s]), dim=self.d)
