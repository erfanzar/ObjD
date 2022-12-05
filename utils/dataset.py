import math
import os
import sys
from abc import ABC

import numpy as np
import torch
import torch as T
import yaml
from PIL import Image
from torch.utils.data import Dataset

from utils.utils import Cp, fast_reader, printf, avg_iou, fast_normalize
from .anchor_predict import anchor_prediction

DEVICE = 'cuda:0' if T.cuda.is_available() else 'cpu'

global mark_try_block, anchors


class DataReader(Dataset):

    def __init__(self, x, y, batch_size):
        self.x = x
        self.y = y
        self.batch_size = batch_size

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item, train: bool = True):
        x, y = [self.x[item], self.y[item]]
        return x, y


class DataLoaderTorch(Dataset, ABC):
    def __init__(self, path, debug: bool = False, nc: int = 4,
                 val_pers=0.3, batch_size: int = 32, auto_anchor: bool = True,
                 prc: float = 0.3,
                 img_shape: int = 640):
        super(DataLoaderTorch, self).__init__()
        with open(path, 'r') as r:
            iw = yaml.full_load(r)

        self.debug = debug
        self.nc = nc
        self.val_pers = val_pers
        self.batch_size = batch_size
        self.img_shape = img_shape
        self.val_perc = val_pers
        self.auto_anchor = auto_anchor
        self.debug = debug
        self.prc = prc

        self.path_train = os.path.join(os.getcwd(), iw['train'])
        self.path_valid = os.path.join(os.getcwd(), iw['valid'])
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
        self.x_train, self.y_train, self.anchors = self.dynamical_batch(self.__start__(current=self.ti),
                                                                        self.batch_size)
        self.x_val, self.y_val, _ = self.dynamical_batch(self.__start__(current=self.vi, is_val=True), self.batch_size)

    def __start__(self, current, is_val: bool = False):
        name = "train" if not is_val else "Validation"
        xsl, ysl = [], []
        wax = []
        way = []
        path = self.path_valid if is_val else self.path_train

        tm = len(current) if not self.debug else int(len(current) * (self.prc if not is_val else self.val_perc))
        setattr(self, 'tm', tm)
        reader = fast_reader(path, tm, current)
        print(f"{Cp.BLUE}Loading {tm} Samples{Cp.WHITE}")

        for item in range(tm):
            bboxes = reader[item]
            targets = torch.zeros((len(bboxes), 6))
            for i, box in enumerate(bboxes):
                x1, y1, w, h, class_index = box
                wax.append(int(w * self.img_shape))
                way.append(int(h * self.img_shape))
                targets[i, :] = torch.tensor([item % self.batch_size, class_index, x1, y1, w, h])
            img = Image.open(f'{path}/{current[item][:-4]}.jpg')
            data = img.getdata()
            image_pixel = list(list(pixel) for pixel in data)
            image_rgb = np.array(image_pixel)
            x = torch.from_numpy(fast_normalize(image_rgb, self.img_shape))
            x = x.cpu()
            xsl.append(x)

            ysl.append(targets.cpu())
            sys.stdout.write('\r{}Moving Data To Ram Or Gpu %{} remaining '.format(
                f'{Cp.BLUE}{name} : {Cp.RESET}', f"{((item / tm) * 100):.4f}"))
        sys.stdout.write('\n')

        if not is_val:
            anchors = anchor_prediction(wax, way, n_clusters=9, original_height=self.img_shape,
                                        original_width=self.img_shape, c_number=self.img_shape)

            dt = np.asarray([wax, way]).T
            printf(f"{Cp.CYAN}Anchors To use Next Time :{Cp.RESET}\n{anchors}\n")
            printf(f"{Cp.CYAN} Accuracy:{Cp.RESET} {avg_iou(dt, anchors) * 100}")
        if not is_val:
            with open('anchors.yaml', 'w') as w:
                yaml.dump({"anchors": anchors.tolist()}, w)
        print(Cp.RESET)
        try:
            anc = anchors
        except:
            anc = []
        return xsl, ysl, anc

    def dynamical_batch(self, xy, batch):

        x, y, a = xy
        tx, ty, xl, item = [], [], len(x), 0

        loop_i_number = math.ceil(xl / batch) if len(x) < batch else math.ceil(xl / batch)

        for i in range(loop_i_number):

            loop_j_batch = (xl if xl < batch else (batch if i != loop_i_number - 1 else xl % batch)) if batch > 1 else 1

            ex, ey = [], []
            for j in range(loop_j_batch):
                ex.append(x[item].cpu().view(1, 3, self.img_shape, self.img_shape))

                ey.append(y[item].cpu().view(-1, 6))

                item += 1

            if len(ex) != 0:
                ex = torch.cat(ex, 0) if batch > 1 else ex[0]
                ey = torch.cat(ey, 0) if batch > 1 else ey[0]

                tx.append(ex.view(-1, 3, self.img_shape, self.img_shape))
                ty.append(ey.view(-1, 6))

        return tx, ty, a  # Done converting to list or cat torch list

    def train_datareader(self):
        if hasattr(self, 'ti'):
            del self.ti
        data_val = DataReader(self.x_train, self.y_train, self.batch_size)
        return data_val
    def val_datareader(self):
        if hasattr(self, 'vi'):
            del self.vi
        data_val = DataReader(self.x_val, self.y_val, self.batch_size)
        return data_val
