import os
import sys

import numpy as np
import torch
import torch as T
import yaml
from PIL import Image
from .utils import iou_width_height
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

DEVICE = 'cuda:0' if T.cuda.is_available() else 'cpu'


class DataReader(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item, train: bool = True):
        x, y = [self.x[item], self.y[item]]
        return x, y


class DataLoaderLightning(LightningDataModule):
    def __init__(self, path, debug: bool = False, nc: int = 4, val_pers=0.3, batch_size: int = 6, prc: float = 0.3,
                 img_shape: int = 416, val_perc: float = 0.9):
        super(DataLoaderLightning, self).__init__()
        with open(path, 'r') as r:
            iw = yaml.full_load(r)
        self.debug = debug
        self.nc = nc
        self.val_pers = val_pers
        self.batch_size = batch_size
        self.img_shape = img_shape
        self.val_perc = val_perc
        self.debug = debug
        self.prc = prc
        self.path_train = os.path.join(os.getcwd(), iw['train'])
        self.path_valid = os.path.join(os.getcwd(), iw['valid'])
        # self.path_train = iw['train']
        # self.path_valid = iw['valid']
        # self.path_train = os.path.join('E:/Programming/Python/Ai-Projects/ObjectDetectorModule', iw['train'])
        # self.path_valid = os.path.join('E:/Programming/Python/Ai-Projects/ObjectDetectorModule', iw['valid'])
        self.nc = nc
        self.ti = [t for t in os.listdir(self.path_train) if os.path.exists(os.path.join(self.path_train, t)) and
                   t.endswith('.jpg')]
        self.vi = [v for v in os.listdir(self.path_valid) if os.path.exists(os.path.join(self.path_valid, v)) and
                   v.endswith('.jpg')]
        self.s = [13, 26, 52]

        np.seterr(all='ignore')
        self.total = len(self.ti) if not self.debug else int(len(self.ti) / self.prc)
        self.x_train, self.y_train = self.__start__(current=self.ti)
        self.x_val, self.y_val = self.__start__(current=self.vi, is_val=True)

    def __start__(self, current, is_val: bool = False):
        xsl, ysl = [], []
        path = self.path_valid if is_val else self.path_train
        tm = len(current) if not self.debug else int(len(current) * (self.prc if not is_val else self.val_perc))
        print(f"Loading {tm} Samples")
        for item in range(tm):

            with open(f'{path}/{current[item][:-4]}.txt', 'r') as r:
                sr = r.readline()

            bboxes = np.roll(
                np.loadtxt(f'{path}/{current[item][:-4]}.txt', delimiter=" ", ndmin=2, ), 4,
                axis=1).tolist() if len(sr) != 0 else []

            targets = [torch.zeros(3, S, S, 5 + self.nc) for S in self.s]
            for box in bboxes:
                x1, y1, w, h, class_label = box
                dpa = torch.zeros(self.nc)

                dpa[int(class_label)] = 1
                class_label = dpa
                has_anchor = [False] * 3
                for anchor_idx in range(3):
                    scale_idx = torch.div((1e-16 + anchor_idx), other=3)
                    scale_idx = int(scale_idx)
                    anchor_on_scale = anchor_idx % 3
                    S = self.s[scale_idx]
                    i, j = int(S * y1), int(S * x1)

                    anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                    if not anchor_taken and not has_anchor[scale_idx]:
                        targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                        x_cell, y_cell = S * x1 - j, S * y1 - i

                        box_coordinates = torch.tensor(
                            [x1 * S, y1 * S, h * S, h * S]
                        )
                        targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                        targets[scale_idx][anchor_on_scale, i, j, 5:] = class_label

                        has_anchor[scale_idx] = True

            img = Image.open(f'{path}/{current[item][:-4]}.jpg')
            to_tensor = lambda ten: torch.from_numpy(ten)
            tt = lambda xf: xf.type(T.float64)
            tn = lambda xr: xr / 255
            ts = lambda xs: xs.reshape((self.img_shape, self.img_shape, 3))
            data = img.getdata()
            image_pixel = list(list(pixel) for pixel in data)
            image_rgb = np.array(image_pixel).reshape((self.img_shape, self.img_shape, 3))
            image_bgr = image_rgb[:, :, ::-1]
            x = to_tensor(image_rgb)
            x = ts(tn(tt(x))).permute(2, 1, 0).reshape(3, self.img_shape, self.img_shape)
            if DEVICE == 'cuda:0':
                x = x.type(T.cuda.FloatTensor)
            else:
                x = x.type(T.FloatTensor)
            xsl.append(x)
            ysl.append(tuple(targets))

            sys.stdout.write('\r Moving Data To Ram Or Gpu %{} remaining '.format(
                f"{((item / tm) * 100):.4f}"))
        sys.stdout.write('\n')
        return xsl, ysl

    def train_dataloader(self):
        data_train = DataReader(self.x_train, self.y_train)
        return DataLoader(data_train, batch_size=self.batch_size, num_workers=6)

    def val_dataloader(self):
        data_val = DataReader(self.x_val, self.y_val)
        return DataLoader(data_val, batch_size=self.batch_size, num_workers=6)
