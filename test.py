import time

import numpy as np
import torch
import yaml
from module.object_detector_module import ObjectDetectorModule
from utils.dataset import DataLoaderLightning
import pytorch_lightning as pl
import onnxruntime as rt
import onnx
from torch.utils.data import Dataset, DataLoader
from module.loss import ComputeLoss
from sklearn.cluster import KMeans
from utils.utils import printf
from utils.dataset import DataLoaderLightning
import cv2 as cv

if __name__ == "__main__":
    model = rt.InferenceSession('best.onnx', providers=['CPUExecutionProvider'])
    print([v.name for v in model.get_outputs()])
    print([v.name for v in model.get_inputs()])
    cam = cv.VideoCapture(0)
    t1 = time.time()
    while True:
        _, frame = cam.read()
        cv.imshow('windows', frame)
        cv.waitKey(1)
        if cv.waitKey(1) == ord('q'):
            printf(f'Total Estimated time : {time.time() - t1:.2f} sec')
            break
        # model.run('output', {'images':x})
