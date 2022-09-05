import os

import numpy as np
import torch
import torch as T
import yaml
from PIL import Image
from torch.utils.data import Dataset


class DataReader(Dataset):

    def __init__(self, yaml_path, nc: int = 4):
        with open(f'{yaml_path}', 'r') as r:
            iw = yaml.full_load(r)
        print(os.getcwd())
        self.path_train = os.path.join(os.getcwd(), iw['train'])
        self.path_valid = os.path.join(os.getcwd(), iw['valid'])
        self.nc = nc
        self.ti = [t for t in os.listdir(self.path_train) if os.path.exists(os.path.join(self.path_train, t)) and
                   t.endswith('.jpg')]
        self.vi = [v for v in os.listdir(self.path_valid) if os.path.exists(os.path.join(self.path_valid, v)) and
                   v.endswith('.jpg')]

    def __len__(self):
        return len(self.ti)

    def __getitem__(self, item, train: bool = True):
        with open(f'{self.path_train}/{self.ti[item][:-4]}.txt', 'r') as r:
            ic = r.readlines()
        y = T.zeros(len(ic), 4 + self.nc)
        img = Image.open(f'{self.path_train}/{self.ti[item][:-4]}.jpg')
        for idx, data in enumerate(ic):
            ici = ic[idx].replace(' ', '-')
            ha = [i for i, h in enumerate(ici) if h == '-']
            c, x1, x2, y1, y2 = [ici[0:ha[0]], ici[ha[0]:ha[1]], ici[ha[1]:ha[2]], ici[ha[2]:ha[3]], ici[ha[3]:]]
            c_array = np.zeros(self.nc)
            c = int(c)
            c_array[c] = 1
            nw = np.array([x1, x2, y1, y2]).astype(np.float64)
            y[idx, 0:4] = T.from_numpy(nw)
            y[idx, 4:] = T.from_numpy(c_array)
        to_tensor = lambda ten: torch.from_numpy(ten)
        tt = lambda xf: xf.type(T.float64)
        tn = lambda xr: xr / 255
        ts = lambda xs: xs.reshape((416, 416, 3))
        data = img.getdata()
        image_pixel = list(list(pixel) for pixel in data)
        image_rgb = np.array(image_pixel).reshape((416, 416, 3))
        image_bgr = image_rgb[:, :, ::-1]
        x = to_tensor(image_rgb)
        x = ts(tn(tt(x)))
        return x, y, image_rgb, image_bgr
