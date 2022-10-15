import time

import cv2 as cv
import torch
import torch.nn as nn
import torch.jit as jit
import argparse
import numpy as np
import onnxruntime as rt
from utils.utils import printf

pars = argparse.ArgumentParser()
pars.add_argument('--path', '--path')
pars.add_argument('--source', '--source', default=0)
pars.add_argument('--img-size', '--img_size', default=640)
pars.add_argument('--weights', '--weights', default='best.torchscript.pt')
args = pars.parse_args()
pars.print_usage()


def check_available_cameras():
    index = 0
    arr = []
    while True:
        cap = cv.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            arr.append(index)
        cap.release()
        index += 1
    return arr


def detect(opt):
    cam = cv.VideoCapture(opt.source)
    t1 = time.time()
    h_w = opt.img_size
    if opt.weights.endswith('.onnx'):
        printf('Onnx Model Deploying .... ')
        model = rt.InferenceSession(opt.weights, providers=['CPUExecutionProvider'])
        while True:
            status, fr = cam.read()
            # printf(f' \r Status : {status}')
            x = cv.cvtColor(fr, cv.COLOR_BGR2RGB)
            x = cv.resize(x / 255, (h_w, h_w))

            x = x.reshape(1, 3, h_w, h_w).astype(np.float32)

            x = model.run(['output'], {'images': x})
            # printf(f' \r {x[0].shape}')
            x = x[0]

            v = x[:, :, 4] > 0.60

            printf(f' \r {x[v]}')
            cv.imshow('windows', fr)
            cv.waitKey(1)
            if cv.waitKey(1) == ord('q'):
                printf(f' \n Total Estimated time : {time.time() - t1:.2f} sec')
                break

    if opt.weights.endswith('.torchscript.pt'):
        printf('Torch Script Model Deploying ....')


if __name__ == "__main__":
    printf(f'There is available cameras : {len(check_available_cameras())}')
    detect(args)
