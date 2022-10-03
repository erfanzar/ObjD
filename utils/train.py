import sys

import numpy as np
import torch
import torch.optim as optim
from colorama import Fore
from module.loss import Loss
from module.ObjectDetectorModule import ObjectDetectorModule
from utils.dataset import DataReader, DataLoaderLightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import BackboneFinetuning, Checkpoint, LearningRateMonitor, ModelCheckpoint, Timer, \
    EarlyStopping
import warnings


class TrainDi:
    def __init__(self):
        self.DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    def run(self):
        return NotImplementedError

    def load(self, path):
        return NotImplementedError

    def jit_save(self):
        return NotImplementedError


class OldMethodTrain(TrainDi):
    def __init__(self, nc: int = 4, cfg_path: str = 'cfg.yaml'):
        super(Train, self).__init__()
        self.nc = nc
        self.cfg_path = cfg_path
        self.net = ObjectDetectorModule(nc=nc, cfg_path=cfg_path).to(self.DEVICE)
        self.dr = DataReader(yaml_path='data/path.yaml', nc=nc, debug=True)
        self.loss = Loss()
        self.epochs = 100
        self.c_epoch = 0
        self.optimizer = optim.SGD(self.net.parameters(), lr=1e-4)
        # self.lambda_lr = lambda epoch: 0.65 ** epoch
        self.scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=0.1, total_iters=2)
        self.grad_scalar = torch.cuda.amp.GradScaler()

    def white_space(self):
        # for idx in range(self.dr.total):
        x, y = self.dr.__getitem__(0)
        print('-' * 20)

        c = torch.tensor(y[0])
        print(c.shape, '\n')

        c = torch.tensor(y[1])
        print(c.shape, '\n')

        c = torch.tensor(y[2])
        print(c.shape, '\n')
        print('-' * 20)

    def run(self):
        while self.c_epoch <= self.epochs:
            fr = True
            for idx in range(self.dr.total):
                x, y = self.dr.__getitem__(idx)
                self.optimizer.zero_grad()

                x = x.to(self.DEVICE)

                tvm = True

                with torch.cuda.amp.autocast():
                    x = self.net.forward(x)
                    y = [torch.unsqueeze(v, 0) for v in y]

                    loss = self.loss(x, y)

                acc = None
                if fr:
                    fr = False
                    sys.stdout.write('\r {}{:>20}/{:<15}{:>15}{:>15}{:>10}'.format(Fore.YELLOW, 'C Ep|', 'Ep|',

                                                                                   'class_loss|',
                                                                                   'accuracy|',
                                                                                   'lr|'))
                    print('/n')
                if idx != 0:
                    sys.stdout.write(
                        '\r {}{:>20}/{:<15}{:>15}{:>15}{:>10}'.format(Fore.YELLOW, f"{self.c_epoch}", f"{self.epochs}",
                                                                      f"{loss:.5f}|",
                                                                      f"{None}|",
                                                                      f"{self.scheduler.get_lr()[0]}"))
                    sys.stdout.flush()
                self.grad_scalar.scale(loss).backward()
                self.grad_scalar.step(optimizer=self.optimizer)
                self.grad_scalar.update()
                if idx % 500 == 0:
                    model_ckpt = {
                        'model': self.net.state_dict(),
                        'optim': self.optimizer.state_dict(),
                        'optim_scheduler': self.scheduler.state_dict(),
                        'epoch': self.c_epoch
                    }
                    torch.save(model_ckpt, 'model_grad.pt')
            # self.scheduler.step()
            print('\n')
            self.c_epoch += 1
            # if self.c_epoch <= self.epochs:
            #     self.jit_save()

    def jit_save(self):
        model_ckpt = {
            'model': self.net.state_dict(),
            'optim': self.optimizer.state_dict(),
            'optim_scheduler': self.scheduler.state_dict(),
            'epoch': self.c_epoch
        }
        di = torch.randn((1, 3, 416, 416)).to(self.DEVICE)
        j = torch.jit.trace(self.net, di, check_trace=False)
        s = torch.jit.script(j)
        torch.jit.save(s, 'model-jit.pt',
                       model_ckpt
                       )

    def load(self, path):
        lod = LoadObjectDetectorModule('model.pt')
        m, o, e = lod.load()
        self.net.load_state_dict(m)
        self.optimizer.load_state_dict(o)
        self.c_epoch = e
        lod.show()
        print('{:>35}{:>20}'.format('Status :', " Done *" if m else 'Error !'))
