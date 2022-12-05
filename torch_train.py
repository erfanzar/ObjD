import argparse
import math
import os
import time

import torch.optim

from module.loss import ComputeLoss
from module.object_detector_module import ObjectDetectorModule
from utils.dataset import DataLoaderTorch
from utils.logger import Logger
from utils.utils import printf, Cp

pars = argparse.ArgumentParser()
pars.add_argument("--epochs", "-epochs", default=300, type=int, help='Epochs For Model Train')
pars.add_argument("--data", "-data", default="data/path.yaml", type=str, help='Path to data yaml file')
pars.add_argument("--cfg", "-configs", default="cfg/objd-s.yaml", type=str, help='Path to config file')
pars.add_argument("--batch", "-batch size", default=3, type=int, help='Batch Size for train')
pars.add_argument("--eval", "-eval model", help='use validation for model',
                  action='store_true')
pars.add_argument("--debug", "-debug Mode", help='For Developing', action='store_true')
pars.add_argument("--auto-anchors", "-anchors", default=False, type=bool, help='automate the anchors for your dataset')
pars.add_argument("--device", '-Device', default='cuda:0' if torch.cuda.is_available() else 'cpu', type=str,
                  help="Device to train model Eg cpu cuda")

opt = pars.parse_args()


# pars.print_help()
# pars.print_usage()


def train(opt):
    logger = Logger()
    # printf(*(f'{v} : {eval(f"opt.{v}")} \n' for i, v in enumerate(pal)))
    printf(f"{Cp.CYAN}Device : {Cp.RESET}{opt.device} \n")
    printf(f"{Cp.CYAN}epochs : {Cp.RESET}{opt.epochs} \n")
    printf(f"{Cp.CYAN}batch  : {Cp.RESET}{opt.batch} \n")
    printf(f"{Cp.CYAN}debug  : {Cp.RESET}{opt.debug} \n")
    printf(f"{Cp.CYAN}autoAnchor  : {Cp.RESET}{opt.auto_anchors} \n")
    printf((f"{Cp.CYAN}Running on the Gpu : {Cp.RESET}{torch.cuda.get_device_name()}\n" if opt.device == 'cuda:0' else
            f"{Cp.RED}Gpu Is available But Running on Cpu RunTime {Cp.RESET} \n" +
            f"{Cp.CYAN}Running with cpu \n Cpu Counts : {Cp.RESET}{os.cpu_count()}\n")
           if torch.cuda.is_available() else f"{Cp.CYAN}Running with cpu \n Cpu Counts :{Cp.RESET} {os.cpu_count()}\n")
    dataloader = DataLoaderTorch('data/path.yaml', batch_size=opt.batch, debug=opt.debug, prc=0.05)
    model = ObjectDetectorModule(cfg=opt.cfg).to(opt.device)
    if opt.auto_anchors:
        model.anchors = dataloader.anchors
    model.init()
    train_data = dataloader.train_datareader()
    optimizer = torch.optim.SGD(model.parameters(), 0.0001)
    scalar = torch.cuda.amp.GradScaler()
    eval_data = dataloader.val_datareader()
    loss_function = ComputeLoss(model, last_layer=model.m[-1])
    ceil_train = math.ceil(train_data.__len__())
    ceil_eval = math.ceil(eval_data.__len__())

    for epoch in range(opt.epochs):
        printf(f"Epoch {epoch + 1} / {opt.epochs} \n\n")
        for i in range(ceil_train):

            s = time.time()
            optimizer.zero_grad()
            x, y = train_data.__getitem__(i)
            x, y = x.to(opt.device), y.to(opt.device)
            x_ = model.forward(x)
            loss = loss_function(x_, y)
            scalar.scale(loss[0]).backward()
            scalar.step(optimizer)
            logger.set_desc(
                '\r {:>20} {:>20} {:>20} {:>20}'.format(f'{i + 1}/{ceil_train}', f'Batch Loss : {loss[0].item()} ',
                                                        f'Loss : {[v.item() for v in loss[1]]} ',
                                                        f'MAP : {time.time() - s:.4f}'))
            scalar.update()
            del x, y, x_
            logger()
            if epoch % 15 == 0:
                ckpt = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch_data': [epoch, opt.epochs],
                    'anchors': model.anchors,
                    'cfg': opt.cfg
                }
                torch.save(ckpt, 'model.pt')

        logger.end()
        for i in range(ceil_eval):
            s = time.time()
            x, y = eval_data.__getitem__(item=i)
            x, y = x.to(opt.device), y.to(opt.device)
            x_ = model.forward(x)
            loss = loss_function(x_, y)
            logger.set_desc(
                '\r Validating {:>20} {:>20} {:>20} {:>20}'.format(f'{i + 1}/{ceil_eval}',
                                                                   f'Batch Loss : {loss[0].item()} ',
                                                                   f'Loss : {[v.item() for v in loss[1]]} ',
                                                                   f'MAP : {time.time() - s:.4f}'))
            logger()
        logger.end()


if __name__ == "__main__":
    train(opt=opt)
