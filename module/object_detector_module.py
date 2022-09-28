import torch
import torch.nn as nn
import yaml
from colorama import Fore
from .loss import Loss
from .commons import (Conv, Detect, ResidualBlock, Neck, C3, C4P, MP, UC1, CV1, RepConv, ConvSc)
import pytorch_lightning as pl
from torchmetrics.functional import accuracy

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class ObjectDetectorModule(pl.LightningModule):
    def __init__(self, nc: int = 4, cfg=None):
        super(ObjectDetectorModule, self).__init__()
        """
        Nc : Number of Classes
        cfg : pass dict version of configs to create module
        """
        if cfg is None:
            cfg = {}
        self.nc = nc
        if isinstance(cfg, str):
            with open(cfg, 'r') as r:
                self.cfg = yaml.full_load(r)
        else:
            self.cfg = cfg

        self.layers = self.layer_creator()
        self.fr = False

        self.to(DEVICE)
        self.loss = Loss()
        self.save_hyperparameters()

    def layer_creator(self):
        layers = nn.ModuleList()
        for cfg in self.cfg:
            at = cfg['attributes']
            if cfg['name'] == 'Conv':
                layers.append(Conv(c1=at[0], c2=at[1], kernel_size=at[2], stride=at[3], act=at[4], batch=at[5],
                                   padding=1 if at[2] == 3 else 0)).to(DEVICE)
            if cfg['name'] == 'ResidualBlock':
                layers.append(ResidualBlock(c1=at[0], n=at[1], use_residual=at[2]).to(DEVICE))
            if cfg['name'] == 'Detect':
                layers.append(Detect(c1=at[0], nc=at[1]).to(DEVICE))
            if cfg['name'] == 'UpSample':
                layers.appendnn.Upsample(scale_factor=at[0]).to(DEVICE)
            if cfg['name'] == 'C3':
                layers.append(C3(c1=at[0], c2=at[1], shortcut=at[2], n=at[3], e=at[4]).to(DEVICE))
            if cfg['name'] == 'Neck':
                layers.append(Neck(c1=at[0], c2=at[1], shortcut=at[2], e=at[3], ).to(DEVICE))
            if cfg['name'] == 'C4P':
                layers.append(C4P(c=at[0], e=at[1], n=at[2], ct=at[3]).to(DEVICE))
            if cfg['name'] == 'MP':
                layers.append(MP())
            if cfg['name'] == 'UC1':
                layers.append(UC1(c1=at[0], c2=at[1], e=at[2], dim=at[3]))
            if cfg['name'] == 'CV1':
                layers.append(CV1(c1=at[0], c2=at[1], e=at[2], n=at[3], shortcut=at[4]))
            if cfg['name'] == 'RepConv':
                layers.append(RepConv(c=at[0], e=at[1], n=at[2]))
            if cfg['name'] == 'ConvSc':
                layers.append(ConvSc(c=at[0], n=at[1]))
        return nn.Sequential(*layers)

    # @torch.jit.script
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
        bpm = None
        dtt = []
        vi = 0

        for index, layer in enumerate(self.layers):
            # print(x.shape, '\n')
            if not isinstance(layer, (nn.Upsample, Detect, MP)):
                if self.fr:
                    print('{}{:>50} {:>20}   {:>20}'.format(Fore.BLUE,
                                                            f'Shape Before RunTime {[l for l in x.shape]}', "[!]",
                                                            f"Layer : {vi}"))
                    print('{:>50} {:>20}   {:>20}'.format(f'Pass To {type(layer).__name__}', "[->]",
                                                          f"Layer : {vi}"))
                x = layer(x)
                if self.fr:
                    print('{:>50} {:>20}   {:>20}'.format(f'Shape After  RunTime {[l for l in x.shape]}', "[*]",
                                                          f"Layer : {vi}"))
                    print('-' * 100)

            if isinstance(layer, MP):
                route = layer(x, route)
                if self.fr:
                    print('{:>50}  {:>20}'.format(f'NOTICE ! add To Route shape {[v for v in x.shape]}',
                                                  '! WARNING !'))
                    print('-' * 100)
            if isinstance(layer, Detect):
                if self.fr:
                    print('{:>50} {:>20}   {:>20}'.format('Detect Layer on RunTime', '[!]', f"Layer : {vi}",
                                                          ))
                    print('{:>20}'.format('Before Detect Layer : {}'.format(x.shape)))
                f = layer(x)
                dtt.append(f)
                if self.fr:
                    print(
                        '{:>50} {:>20}   {:>20}'.format(f'Detect Layer Done shape {[v for v in f.shape]}]', '[*]',
                                                        f"Layer : {vi}"))
                    print('-' * 100)
            if isinstance(layer, nn.Upsample):
                x = layer(x)
                if len(route[-1].shape) == 3:
                    c = torch.unsqueeze(route[-1], dim=0)
                else:
                    c = route[-1]
                if self.fr:
                    print("\n{:>100}\n".format(f'Trying to pair x : {x.shape} to residual {route[-1].shape}',
                                               ))

                if self.fr:
                    print('{:>50} {:>20}   {:>20}\n'.format(f'UpSample Layer {[v for v in x.shape]}]', '[!]',
                                                            f"Layer : {vi}"))
                    print('-' * 100)

                x = torch.concat((c, x), dim=1)
                if isinstance(route, list):
                    route = route.pop(0)

            vi += 1
        if len(dtt) != 0:
            cv = dtt
        else:
            cv = x
        if self.fr:
            self.fr = False
        return cv

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
        loss = (self.loss.forward(x_[0], y[2], 0) + self.loss.forward(x_[1], y[1], 1) + self.loss.forward(x_[2], y[0],
                                                                                                          2))
        # acc_train_v1 = accuracy(x_[0], target=y[2].int())
        # acc_train_v2 = accuracy(x_[1], target=y[1].int())
        # acc_train_v3 = accuracy(x_[2], target=y[0].int())
        # self.log('train_acc', torch.tensor((acc_train_v1, acc_train_v2, acc_train_v3)), prog_bar=True, on_step=True,
        #          on_epoch=True)
        self.log('train_loss', loss, prog_bar=True, on_step=True,
                 on_epoch=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_index):
        x, y = batch
        x_ = self(x)

        loss = (self.loss.forward(x_[0], y[2], 0) + self.loss.forward(x_[1], y[1], 1) + self.loss.forward(x_[2], y[0],
                                                                                                          2))
        # acc_val_v1 = accuracy(x_[0], target=y[2].int())
        # acc_val_v2 = accuracy(x_[1], target=y[1].int())
        # acc_val_v3 = accuracy(x_[2], target=y[0].int())
        # self.log('val_acc', torch.tensor((acc_val_v1, acc_val_v2, acc_val_v3)), prog_bar=True, on_step=True,
        #          on_epoch=True)
        self.log('val_loss', loss, prog_bar=True, on_step=True,
                 on_epoch=True)
        return {"loss": loss}
