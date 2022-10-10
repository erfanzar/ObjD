import numpy as np
import torch

from module.object_detector_module import ObjectDetectorModule
from utils.dataset import DataLoaderLightning
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from module.loss import ComputeLoss
from sklearn.cluster import KMeans

if __name__ == "__main__":
    net = ObjectDetectorModule(cfg='cfg/objd-n.yaml')
    x = torch.randn((1, 3, 640, 640)).to("cuda:0")
    x = net.forward(x)
    print(*(xs.shape for xs in x))
