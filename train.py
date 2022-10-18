from module.object_detector_module import ObjectDetectorModule
from utils.dataset import DataLoaderLightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import BackboneFinetuning, Checkpoint, LearningRateMonitor, ModelCheckpoint, Timer, \
    EarlyStopping
from utils.config import cfg as cfg_list
from pytorch_lightning.callbacks import ModelSummary


class LightningTrain:
    def __init__(self, nc: int = 4, cfg: [str, list] = 'cfg/objd-n.yaml'):
        super(LightningTrain, self).__init__()
        self.nc = nc
        self.cfg = cfg
        self.net = ObjectDetectorModule(cfg=self.cfg)
        self.net.fr = False

    def train(self, time: str = None):
        self.net.prepare_data()

        checkpoint = Checkpoint()
        model_summery = ModelSummary(max_depth=10)
        model_checkpoint = ModelCheckpoint(dirpath='model/saves/', save_top_k=10, monitor='train_loss')
        lr_monitor = LearningRateMonitor(logging_interval='step')
        # early_stopping = EarlyStopping('Loss')
        timer = Timer(duration='01:00:00:00' if time is None else time)
        trainer = pl.Trainer(accelerator='cpu', min_epochs=50, max_epochs=50000, auto_lr_find=True,
                             callbacks=[model_summery, model_checkpoint, checkpoint, lr_monitor, timer])
        data_loader_lightning = DataLoaderLightning(path='data/path.yaml', debug=True, val_pers=0.01, nc=4, prc=0.01,
                                                    batch_size=1)
        dataloader_train = data_loader_lightning.train_dataloader()
        dataloader_validation = data_loader_lightning.val_dataloader()
        trainer.fit(self.net, train_dataloaders=dataloader_train, val_dataloaders=dataloader_validation)


if __name__ == "__main__":
    train_class = LightningTrain()
    train_class.train()
