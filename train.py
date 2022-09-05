import torch
from torchsummary import summary
from module.convs import ObjD
from utils.dataset import DataReader

if __name__ == "__main__":
    dr = DataReader(yaml_path='E:/Python/objd/data/path.yaml', nc=4)
    x, y, img_rgb, img_bgr = dr.__getitem__(4)

    net = ObjD(nc=4, cfg_path='cfg.yaml')
    # summary(model=net, input_size=(3, 416, 416))
    dummy_inp = torch.rand((1, 3, 416, 416))
    net.forward(dummy_inp)
