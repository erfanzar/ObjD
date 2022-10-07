import torch

from module.object_detector_module import ObjectDetectorModule
from utils.dataset import DataLoaderLightning
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader


class Cd(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return 2

    def __getitem__(self, item):
        return torch.zeros(( 3, 640, 640)), torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0])


if __name__ == "__main__":
    net = ObjectDetectorModule(cfg='cfg/objd-n.yaml')

