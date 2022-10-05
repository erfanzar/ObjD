import torch

from module.object_detector_module import ObjectDetectorModule

if __name__ == "__main__":
    model = ObjectDetectorModule(cfg='cfg/tiny.yaml')
    x = torch.zeros((1, 3, 416, 416)).to('cuda:0')
    x = model.back_forward(x)
