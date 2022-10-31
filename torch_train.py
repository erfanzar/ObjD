import time

import torch.optim
from utils.utils import printf, Cp
from module.object_detector_module import ObjectDetectorModule
from utils.dataset import DataLoaderTorch
from utils.config import cfg as cfg_list
from module.loss import ComputeLoss
from utils.logger import Logger

import math
import argparse

pars = argparse.ArgumentParser()
pars.add_argument("--epochs", "-epochs", default=300, type=int, help='Epochs For Model Train')
pars.add_argument("--data", "-data", default="data/path.yaml", type=str, help='Path to data yaml file')
pars.add_argument("--cfg", "-configs", default="cfg/objd-s.yaml", type=str, help='Path to config file')
pars.add_argument("--batch", "-batch size", default=4, type=int, help='Batch Size for train')
pars.add_argument("--eval", "-eval model", default=False, type=bool, help='use validation for model')
pars.add_argument("--debug", "-debug Mode", default=True, type=bool, help='For Developing')
pars.add_argument("--device", '-Device', default='cuda:0', type=str,
                  help="Device to train model Eg cpu cuda")

opt = pars.parse_args()

# pars.print_help()
pars.print_usage()


def train(opt):
    logger = Logger()
    # printf(*(f'{v} : {eval(f"opt.{v}")} \n' for i, v in enumerate(pal)))
    printf(f"{Cp.CYAN}Device : {Cp.WHITE}{opt.device} \n")
    printf(f"{Cp.CYAN}epochs : {Cp.WHITE}{opt.epochs} \n")
    printf(f"{Cp.CYAN}batch  : {Cp.WHITE}{opt.batch} \n")
    printf(f"{Cp.CYAN}debug  : {Cp.WHITE}{opt.debug} \n")
    model = ObjectDetectorModule(cfg=opt.cfg).to(opt.device)
    dataloader = DataLoaderTorch('data/path.yaml', batch_size=opt.batch, debug=opt.debug, prc=0.05)
    train_data = dataloader.train_datareader()
    optimizer = torch.optim.SGD(model.parameters(), 0.0001)
    scalar = torch.cuda.amp.GradScaler()
    eval_data = dataloader.val_datareader()
    loss_function = ComputeLoss(model, last_layer=model.m[-1])
    ceil_train = math.ceil(train_data.__len__() / opt.batch) + 1
    ceil_eval = math.ceil(eval_data.__len__() / opt.batch) + 1
    for epoch in range(opt.epochs):
        for i in range(ceil_train):
            s = time.time()
            optimizer.zero_grad()
            x, y = train_data.__getitem__(item=i)
            x, y = x.to(opt.device), y.to(opt.device)

            x_ = model.forward(x)
            loss = loss_function(x_, y)
            scalar.scale(loss[0]).backward()
            scalar.step(optimizer)
            logger.set_desc('\r {:>20} {:>20} {:>20} {:>20}'.format(f'{i + 1}/{ceil_train}', f'Batch Loss : {loss[0]} ',
                                                                    f'Loss : {loss[1]} ',
                                                                    f'MAP : {time.time() - s:.4f}'))
            scalar.update()
            del x, y, x_
            logger()
        logger.end()
        for i in range(ceil_eval):
            s = time.time()
            x, y = eval_data.__getitem__(item=i)
            x.y = x.to(opt.device), y.to(opt.device)
            x_ = model.forward(x)
            loss = loss_function(x_, y)
            logger.set_desc(
                '\r Validating {:>20} {:>20} {:>20} {:>20}'.format(f'{i + 1}/{ceil_train}', f'Batch Loss : {loss[0]} ',
                                                                   f'Loss : {loss[1]} ',
                                                                   f'MAP : {time.time() - s:.4f}'))
            logger()
        logger.end()


if __name__ == "__main__":
    train(opt=opt)
