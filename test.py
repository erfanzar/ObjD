import time
import numpy as np
import torch
import yaml
from module.object_detector_module import ObjectDetectorModule
from utils.dataset import DataLoaderLightning
import pytorch_lightning as pl
import onnxruntime as rt
import torch.nn as nn
import onnx
from torch.utils.data import Dataset, DataLoader
from module.loss import ComputeLoss
from sklearn.cluster import KMeans
from utils.utils import printf
from utils.dataset import DataLoaderLightning
import cv2 as cv

if __name__ == "__main__":
    model = ObjectDetectorModule(cfg='cfg/objd-n.yaml')
    x = torch.zeros((1, 3, 640, 640)).to('cuda:0')
    x = model(x)
    loss = ComputeLoss(model)
    target = torch.zeros((1, 6)).to('cuda:0')
    target[0, :] = torch.tensor([0, 1, 0.3, 0.1, 0.5, 0.6])
    l = loss(p=x, targets=target)
    print(l)
