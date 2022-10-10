import cv2 as cv
import torch
import torch.nn as nn
import torch.jit as jit
import argparse
import numpy as np
import onnx
import onnxruntime
import keyboard

pars = argparse.ArgumentParser()
pars.add_argument('-p', '--path')
pars.add_argument('-s', '--source')

opt = pars.parse_args()

# model = jit.load('last.torchscript.pt', 'cuda:0')

cam = cv.VideoCapture(0)

if __name__ == "__main__":
    model = onnxruntime.InferenceSession('last.onnx')

    # model = onnx.load_model('last.onnx')
    # print(*(v.name for v in model.get_inputs()))
    # print(*(v.shape for v in model.get_inputs()))
    # print(*(v.name for v in model.get_outputs()))
    # print(*(v.shape for v in model.get_outputs()))
    while True:

        _, frame = cam.read()
        # inp = frame / 255
        inp = (frame / 255)
        inp = cv.resize(inp, (640, 640))

        inp[:, :, [0, 1, 2]] = inp[:, :, [2, 1, 0]]
        # inp = torch.cuda.FloatTensor(inp.float())
        inp = np.float32(inp).reshape(1, 3, 640, 640)

        res = model.run(['output'], {'images': inp})
        cv.imshow('windows', frame)
        res = res[0]
        print(res.shape)
        for i in range(res.shape[1]):
            if res[0, i, 4].tolist() > 0.3:
                print(res[0, i, 5:])
        if keyboard.is_pressed('q'):
            break
        cv.waitKey(1)
