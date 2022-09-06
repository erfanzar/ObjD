import torch

from module.loss import Loss
from module.objd import ObjD
from utils.dataset import DataReader

if __name__ == "__main__":
    net = ObjD(nc=4, cfg_path='cfg.yaml')
    dr = DataReader(yaml_path='E:/Python/objd/data/path.yaml', nc=4)
    loss = Loss()
    epochs = 10
    for epoch in range(epochs):
        for idx in range(dr.__len__()):
            x, y = dr.__getitem__(idx)
            x = net.forward(x)
            x = x.view(-1, 9)
            ls = loss.forward(x, y)
            print(ls)
