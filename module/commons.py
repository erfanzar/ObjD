import torch as T
import torch.jit
import torch.nn as nn
import pytorch_lightning as pl

DEVICE = 'cuda:0' if T.cuda.is_available() else 'cpu'


class Conv(pl.LightningModule):
    def __init__(self, c1: int, c2: int, act: bool = True, batch: bool = False, **kwargs):
        super(Conv, self).__init__()
        self.c1 = c1
        self.c2 = c2
        self.act = act
        self.batch = batch
        self.to(DEVICE)
        self.conv = nn.Conv2d(c1, c2, **kwargs).to(DEVICE)
        nn.init.xavier_normal_(self.conv.weight.data)
        self.r = nn.LeakyReLU(0.02).to(DEVICE)
        self.n = nn.BatchNorm2d(c2).to(DEVICE)

    def forward(self, x) -> T.Tensor:
        x += 1e-32
        x = self.conv(x)
        if self.batch:
            x = self.n(x)
        if self.act:
            x = self.r(x)

        return x + 1e-32


class Neck(pl.LightningModule):
    def __init__(self, c1, c2, e=0.5, shortcut=False):
        super(Neck, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, kernel_size=1, stride=1)
        self.cv2 = Conv(c_, c2, kernel_size=3, stride=1, padding=1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        ck = self.cv2(self.cv1(x))

        k = x + ck if self.add else ck

        return k


class C3(pl.LightningModule):
    def __init__(self, c1, c2, e=0.5, n=1, shortcut=True):
        super(C3, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, kernel_size=3, stride=1, padding=1)
        self.cv2 = Conv(c1, c_, kernel_size=3, stride=1, padding=1)
        self.cv3 = Conv(c_ * 2, c2, kernel_size=3, padding=1)
        self.m = nn.Sequential(*(Neck(c_, c_, shortcut=shortcut, e=0.5) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv2(x)), self.cv1(x)), dim=1))


class C4P(C3):
    def __init__(self, c, e=0.5, n=1, ct=2):
        super(C4P, self).__init__(c1=c, c2=c, e=e, n=n)
        self.ct = ct

    def forward(self, x):
        for _ in range(self.ct):
            x = self.cv3(torch.cat((self.m(self.cv2(x)), self.cv1(x)), dim=1)) + x
        return x


class RepConv(pl.LightningModule):
    def __init__(self, c, e=0.5, n=3):
        super(RepConv, self).__init__()
        c_ = int(c * e)
        self.layer = nn.ModuleList()
        # self.layer.append(
        #     *(Conv(c1=c if i == 0 else c_, c2=c_ if i == 0 else c, kernel_size=3, padding=1, stride=1, batch=False)
        #       for i in range(n)))
        for i in range(n):
            self.layer.append(
                Conv(c1=c if i == 0 else c_, c2=c_ if i == 0 else c, kernel_size=3, padding=1, stride=1, batch=False))

    def forward(self, x):
        x_ = x
        for layer in self.layer:
            x = layer.forward(x)
        return x_ + x


class ConvSc(RepConv):
    def __init__(self, c, n=4):
        super(ConvSc, self).__init__(c=c, e=1, n=n)

    def forward(self, x):
        x_ = x.detach().clone()
        for layer in self.layer:
            x = layer(x) + x
        return x + x_


class ResidualBlock(pl.LightningModule):
    def __init__(self, c1, n: int = 4, use_residual: bool = True):
        super(ResidualBlock, self).__init__()
        self.use_residual = use_residual
        self.n = n
        self.to(DEVICE)
        self.layer = nn.ModuleList()

        for _ in range(n):
            self.layer.append(
                nn.Sequential(
                    Conv(c1, c1 * 2, act=True, batch=True, stride=1, padding=0, kernel_size=1),
                    Conv(c1 * 2, c1, act=True, batch=True, stride=1, padding=1, kernel_size=3)
                )
            )

    def forward(self, x) -> T.Tensor:
        c = x
        for layer in self.layer:
            x = layer(x)
        return x + c if self.use_residual else x


class Detect(pl.LightningModule):
    def __init__(self, c1, nc, use_anc: bool = False):
        super(Detect, self).__init__()
        self.nc = nc
        self.to(DEVICE)
        self.use_anc = use_anc
        self.layer = nn.Sequential(
            Conv(c1=c1, c2=c1 * 2, act=True, batch=False, kernel_size=1),
            Conv(c1=c1 * 2, c2=(5 + self.nc) * 3 if use_anc else (5 + self.nc), kernel_size=1, batch=False, padding=0,
                 stride=1, act=True)
        )

    def forward(self, x) -> T.Tensor:
        return self.layer(x).reshape(x.shape[0], 3 if self.use_anc else 1, self.nc + 5, x.shape[2], x.shape[3]).permute(
            0, 1, 3, 4, 2)


class CV1(pl.LightningModule):
    def __init__(self, c1, c2, e=0.5, n=1, shortcut=False, dim=-3):
        super(CV1, self).__init__()
        c_ = int(c2 * e)
        if shortcut:
            c2 = c1
        self.c = Conv(c1, c_, kernel_size=3, padding=1, stride=1)
        self.v = Conv(c1, c_, kernel_size=3, padding=1, stride=1)
        self.m = nn.Sequential(
            *(Conv(c_ * 2 if i == 0 else c2, c2, kernel_size=3, stride=1, padding=1) for i in range(n)))
        self.sh = c1 == c2
        self.dim = dim

    def forward(self, x):
        c = torch.cat((self.c(x), self.v(x)), dim=self.dim)
        return self.m(c) if not self.sh else self.m(
            torch.cat((self.c(x), self.v(x)), dim=self.dim)) + x


class UC1(pl.LightningModule):
    def __init__(self, c1, c2, e=0.5, dim=-3):
        super(UC1, self).__init__()
        c_ = int(c2 * e)
        self.c = Conv(c1=c1, c2=c_, kernel_size=1, stride=1)
        self.v = Conv(c1=c1, c2=c_, kernel_size=1, stride=1)
        self.m = Conv(c1=c_, c2=c2, kernel_size=1, stride=1)
        self.dim = dim

    def forward(self, x):
        return self.m(torch.cat((self.c(x), self.v(x)), dim=self.dim))


class MP(pl.LightningModule):
    def __init__(self):
        super(MP, self).__init__()
        self.ls = None

    def forward(self, x, ls):
        ls.append(x)
        return ls


class LP(pl.LightningModule):
    def __init__(self, dim: int = None):
        super(LP, self).__init__()
        self.dim = dim

    def forward(self, l1, l2, dim_f: int = 1):
        return torch.cat((l1, l2), dim=dim_f if self.dim is None else self.dim)
