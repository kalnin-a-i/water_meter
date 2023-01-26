from torch.optim import Adam
from pytorch_lightning import LightningModule
from torch import nn

class SegmetationModule(LightningModule):
    def __init__(self, model, lr:float=1e-4, **model_args) -> None:
        super().__init__()
        self.model = model(**model_args)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr

        self.save_hyperparameters(ignore=['criterion', 'model'])

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        inputs, masks = batch
        preds = self.model(inputs)
        preds, masks = preds.squeeze(1), masks.squeeze(1)
        loss = self.criterion(preds, masks)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch):
        inputs, masks = batch
        preds = self.model(inputs)
        preds, masks = preds.squeeze(1), masks.squeeze(1)
        loss = self.criterion(preds, masks)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return 