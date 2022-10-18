import os
import sys

import numpy as np
import torch
import torch as T
import yaml
from PIL import Image
# from .anchor_predict import anchor_prediction
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from utils.utils import iou

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
                 img_shape: int = 640):
        super(DataLoaderLightning, self).__init__()
        with open(path, 'r') as r:
            iw = yaml.full_load(r)

        self.debug = debug
        self.nc = nc
        self.val_pers = val_pers
        self.batch_size = batch_size
        self.img_shape = img_shape
        self.val_perc = val_pers
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
        self.s = [20, 40, 80]

        np.seterr(all='ignore')
        if len(self.ti) == 0:
            raise Exception('NO Data Found')

        self.total = len(self.ti) if not self.debug else int(len(self.ti) / self.prc)
        self.x_train, self.y_train = self.__start__(current=self.ti)
        self.x_val, self.y_val = self.__start__(current=self.vi, is_val=True)

    def __start__(self, current, is_val: bool = False):
        xsl, ysl = [], []
        wax = []
        way = []
        path = self.path_valid if is_val else self.path_train
        tm = len(current) if not self.debug else int(len(current) * (self.prc if not is_val else self.val_perc))
        print(f"Loading {tm} Samples")
        for item in range(tm):
            with open(f'{path}/{current[item][:-4]}.txt', 'r') as r:
                sr = r.readline()
            bboxes = np.roll(
                np.loadtxt(f'{path}/{current[item][:-4]}.txt', delimiter=" ", ndmin=2, ), 4,
                axis=1).tolist() if len(sr) != 0 else []
            targets = torch.zeros((len(bboxes), 6))
            for i, box in enumerate(bboxes):
                x1, y1, w, h, class_index = box
                w += x1
                h += y1
                targets[i, :] = torch.tensor([item, class_index, x1, y1, w, h])
                # wax.append(int(w * self.img_shape))
                # way.append(int(h * self.img_shape))

            img = Image.open(f'{path}/{current[item][:-4]}.jpg')
            to_tensor = lambda ten: torch.from_numpy(ten)
            tt, tn, ts = [lambda xf: xf.type(T.float64), lambda xr: xr / 255, lambda xs: xs.reshape(
                (self.img_shape, self.img_shape, 3))]
            data = img.getdata()
            image_pixel = list(list(pixel) for pixel in data)
            image_rgb = np.array(image_pixel).reshape((self.img_shape, self.img_shape, 3))
            image_bgr = image_rgb[:, :, ::-1]
            x = ts(tn(tt(to_tensor(image_rgb)))).permute(2, 1, 0).reshape(3, self.img_shape, self.img_shape)
            # x = x.type(T.cuda.FloatTensor) if DEVICE == 'cuda:0' else x.type(T.FloatTensor)
            x = x.cpu()
            xsl.append(x)
            ysl.append(targets.cpu())
            sys.stdout.write('\r Moving Data To Ram Or Gpu %{} remaining '.format(
                f"{((item / tm) * 100):.4f}"))
        sys.stdout.write('\n')
        # anchors = anchor_prediction(wax, way, n_clusters=9, original_height=self.img_shape,
        #                             original_width=self.img_shape, c_number=self.img_shape)
        # if not is_val:
        #     with open('anchors.txt', 'a') as w:
        #         w.write(f"{anchors.tolist()}")
        return xsl, ysl

    def train_dataloader(self):
        del self.ti
        data_train = DataReader(self.x_train, self.y_train)
        return DataLoader(data_train, batch_size=self.batch_size, num_workers=1)

    def val_dataloader(self):
        del self.vi
        data_val = DataReader(self.x_val, self.y_val)
        return DataLoader(data_val, batch_size=self.batch_size, num_workers=1
                          )
