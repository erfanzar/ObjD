import torch
import torch.nn as nn


class LoadObjectDetectorModule(pl.LightningModule):
    def __init__(self, path):
        super(LoadObjectDetectorModule, self).__init__()
        self.path = path
        self.model = self.load_mo()

    def load_mo(self):
        return torch.load(self.path)

    def show(self):
        print('{:>35}{:>20}'.format('Ran Epochs :', self.model['epoch']))
        print('{:>35}{:>20}'.format('Model Load Status :', 'True' if self.model['model'] else 'False'))
        print('{:>35}{:>20}'.format('Optim Load Status :', 'True' if self.model['optim'] else 'False'))

    def load(self):
        return self.model['model'], self.model['optim'], self.model['epoch']
