import torch
import pytorch_lightning as pl
from module.object_detector_module import ObjectDetectorModule

if __name__ == "__main__":
    va = ObjectDetectorModule()
    model = ObjectDetectorModule.load_from_checkpoint('model/saves/epoch=148-step=201150.ckpt')
    inp = torch.rand((1, 3, 640, 640))
    model.to_onnx('model.onnx', inp, export_params=True)
    # script = model.to_torchscript()
    # torch.jit.save(script, "model.torchscript")
