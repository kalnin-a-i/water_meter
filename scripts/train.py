from torch.optim import Adam
from pytorch_lightning import LightningModule
from torch import nn
from torchmetrics.functional import dice
import torch 


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
        # self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack(outputs).mean()
        self.log('epoch_train_loss', avg_loss)

    def validation_step(self, batch, batch_idx):
        # get preds 
        inputs, masks = batch
        preds = self.model(inputs)
        preds, masks = preds.squeeze(1), masks.squeeze(1)
        
        # get validation loss
        loss = self.criterion(preds, masks)

        # get validation dice score 
        dice_score = dice(preds, masks.int())

        # self.log('val_dice', dice_score, on_step=False, on_epoch=False, logger=True, prog_bar=True)
        # self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'loss' : loss, 'dice_score': dice_score}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_dice = torch.stack([x['dice_score'] for x in outputs]).mean()

        self.log('epoch_val_loss', avg_loss)
        self.log('epoch_val_dice', avg_dice)
        