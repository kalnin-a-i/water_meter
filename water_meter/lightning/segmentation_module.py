from torch.optim import Adam
from pytorch_lightning import LightningModule
from torch import nn
from torchmetrics.functional import dice
import torch 

class SegmetationModule(LightningModule):
    def __init__(self,
                 model,
                 criterion=nn.CrossEntropyLoss(), 
                 model_args={}, 
                 optimizer_args={},) -> None:
        super().__init__()
        
        # inititalize model
        self.model = model(**model_args)

        #initialize criterion
        self.criterion = criterion

        #initialoze optimizer args
        self.optimizer_args = optimizer_args

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return Adam(self.model.parameters(), **self.optimizer_args)

    def training_step(self, batch, batch_idx):
        # get preds and 
        inputs, masks = batch
        preds = self.model(inputs)
        preds, masks = preds.squeeze(1), masks.squeeze(1)

        # calc loss 
        loss = self.criterion(preds, masks)
        
        return {'preds': preds, 'loss' : loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
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

        return {'preds': preds, 'loss' : loss, 'dice_score': dice_score}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_dice = torch.stack([x['dice_score'] for x in outputs]).mean()

        self.log('epoch_val_loss', avg_loss)
        self.log('epoch_val_dice', avg_dice)



        