import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import OneCycleLR
from optim_lr_scheduler import WarmupLRScheduler


class PhraseSimilarityModelImpl(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.head = nn.Linear(768, 1, bias=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, text, mask):
        # calculate bert output
        feats = self.bert(text, mask)
        # calculate sum of all tokens then divide by the number of tokens
        feats = torch.sum(feats[0], 1) / feats[0].shape[1]
        feats = self.dropout(feats)
        output = self.head(feats)
        return output


class PhraseSimilarityModel(pl.LightningModule):
    def __init__(self, model, lr, num_warmups, num_decreases, criterion, metric):
        super(PhraseSimilarityModel, self).__init__()
        self.model = model
        self.lr = lr
        self.num_warmups = num_warmups
        self.num_decreases = num_decreases
        self.criterion = criterion
        self.metric = metric
        self.lrs = []

    def forward(self, text, mask):
        return self.model(text, mask)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.sch = WarmupLRScheduler(self.optimizer, int(self.num_warmups/8), int(self.num_decreases/8))
        print(f"Scheduler learning rate{self.sch.lr}")
        lr_scheduler = {
                "scheduler": self.sch,
                "interval": "step",
                "frequency": 1
            }
        return [self.optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        self.lrs.append(self.sch.optimizer.param_groups[0]["lr"])
        
        ids, mask = batch[0], batch[1]
        preds = self.model(ids, mask)
        loss = self.criterion(preds.squeeze(1), batch[2])
        rmse = self.metric(preds.squeeze(1), batch[2])
        logs = {"train_loss": loss, "train_error": rmse,
                "lr": self.optimizer.param_groups[0]['lr']}

        self.log_dict(logs, on_step=False, on_epoch=True,
                      prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        ids, mask = batch[0], batch[1]
        preds = self.model(ids, mask)
        loss = self.criterion(preds.squeeze(1), batch[2])
        rmse = self.metric(preds.squeeze(1), batch[2])
        logs = {"val_loss": loss, "val_error": rmse}
        self.log_dict(logs, on_step=False, on_epoch=True,
                      prog_bar=True, logger=True)
        return loss

    def predict_step(self, batch, batch_idx):
        ids, mask = batch[0], batch[1]
        preds = self.model(ids, mask)
        return preds
