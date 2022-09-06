import torch
import torch.nn as nn
import yaml

from .convs import Conv, Detect, Connect, ResidualBlock

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class ObjD(nn.Module):
    def __init__(self, nc: int = 4, cfg_path: str = None):
        super(ObjD, self).__init__()
        self.nc = nc
        with open(cfg_path, 'r') as r:
            self.cfg = yaml.full_load(r)
        self.layers = self.layer_creator()
        self.fr = True
        self.to(DEVICE)

    def layer_creator(self):
        layers = nn.ModuleList()
        for cfg in self.cfg:
            if cfg[0] == 'Conv':
                layers.append(
                    Conv(in_c=cfg[1], out_c=cfg[2], act=cfg[3], batch=cfg[4], kernel_size=cfg[5], stride=cfg[6],
                         padding=1 if cfg[5] == 3 else 0)).to(DEVICE)
            if cfg[0] == 'ResidualBlock':
                layers.append(
                    ResidualBlock(in_c=cfg[1], time=cfg[2], use_residual=cfg[3]).to(DEVICE)
                )
            if cfg[0] == 'Detect':
                layers.append(
                    Detect(in_c=cfg[1], nc=cfg[2]).to(DEVICE)
                )
            # if cfg[0] == 'Connect':
            #     layers.append(
            #         Connect(s=cfg[1][0], d=cfg[1][1])
            #     )
            if cfg[0] == 'UpSample':
                layers.append(
                    nn.Upsample(scale_factor=cfg[1]).to(DEVICE)
                )
        return layers

    def forward(self, x):
        residual_l = []
        bpm = None
        dtt = []
        vi = 0
        for layer in self.layers:

            if not isinstance(layer, Connect) and not isinstance(layer, nn.Upsample) and not isinstance(layer, Detect):
                if self.fr:
                    print('{:>50} {:>20}  {:>20}'.format(
                        f'Shape Before RunTime {[l for l in x.shape]}', "[!]", f"Layer : {vi}"))
                    print('{:>50} {:>20}  {:>20}'.format(f'Pass To {type(layer).__name__}', "[->]", f"Layer : {vi}"))
                x = layer(x)
                if self.fr:
                    print('{:>50} {:>20}  {:>20}'.format(f'Shape After  RunTime {[l for l in x.shape]}', "[*]",
                                                         f"Layer : {vi}"))
                    print('-' * 100)
            if isinstance(layer, Connect):
                if self.fr:
                    print("{:>100}".format(f'Trying to pair x : {x.shape} to residual {residual_l[-1].shape}'))
                    print('-' * 100)
                bpm = layer(residual_l, x)
                residual_l.pop()

            if isinstance(layer, ResidualBlock):
                residual_l.append(x)
                if self.fr:
                    print('{:>50} {:>20}'.format(f'NOTICE ! add To Route shape {[v for v in x.shape]}', '! WARNING !'))
                    print('-' * 100)
            if isinstance(layer, Detect):
                if self.fr:
                    print('{:>50} {:>20}  {:>20}'.format('Detect Layer on RunTime', '[!]', f"Layer : {vi}"))
                f = layer(x)
                dtt.append(f)
                if self.fr:
                    print('{:>50} {:>20}  {:>20}'.format(f'Detect Layer Done shape {[v for v in f.shape]}]', '[*]',
                                                         f"Layer : {vi}"))
                    print('-' * 100)
            if isinstance(layer, nn.Upsample):
                if self.fr:
                    print("\n{:>100}\n".format(f'Trying to pair x : {x.shape} to residual {residual_l[-1].shape}'))

                x = layer(x)
                if self.fr:
                    print('{:>50} {:>20}  {:>20}\n'.format(f'UpSample Layer {[v for v in x.shape]}]', '[!]',
                                                           f"Layer : {vi}"))
                    print('-' * 100)

                x = torch.concat((residual_l[-1], x), dim=1)
                residual_l.pop()
            vi += 1

        if self.fr:
            ps = 0
            for prm in self.layers.parameters():
                ps += prm.nelement() * prm.element_size()
            bs = 0
            for buffer in self.layers.buffers():
                bs += buffer.nelement() * buffer.element_size()
            size_all_mb = (ps + bs) / 1024 ** 2
            print('{:>5} {}'.format('Model Size : ', size_all_mb))
            print('-' * 100)
        if self.fr:
            for i in range(len(dtt)):
                print('{:>50} {:>30}'.format(f"{dtt[i].shape}", f'Detect Layer {i}'))
            print('-' * 100)
        cv = torch.concat((dtt[0], dtt[1], dtt[2]), dim=1)
        if self.fr:
            self.fr = False
        return cv
