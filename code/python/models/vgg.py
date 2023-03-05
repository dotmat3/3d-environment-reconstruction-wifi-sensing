from pytorch_lightning import LightningModule

import torch

from utils import iou

class CSI2EnvVGG(LightningModule):
  def __init__(self, learning_rate):
    super().__init__()

    self.save_hyperparameters()

    self.encoder = torch.nn.Sequential(
        torch.nn.Conv2d(1, 64, kernel_size=2, stride=1, padding="same"),
        torch.nn.ReLU(),
        torch.nn.Conv2d(64, 64, kernel_size=2, stride=1, padding="same"),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2),
        torch.nn.Conv2d(64, 128, kernel_size=2, stride=1, padding="same"),
        torch.nn.ReLU(),
        torch.nn.Conv2d(128, 128, kernel_size=2, stride=1, padding="same"),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2),
        torch.nn.Conv2d(128, 256, kernel_size=2, stride=1, padding="same"),
        torch.nn.ReLU(),
        torch.nn.Conv2d(256, 256, kernel_size=2, stride=1, padding="same"),
        torch.nn.ReLU(),
        torch.nn.Conv2d(256, 256, kernel_size=2, stride=1, padding="same"),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2),
        torch.nn.Conv2d(256, 512, kernel_size=2, stride=1, padding="same"),
        torch.nn.ReLU(),
        torch.nn.Conv2d(512, 512, kernel_size=2, stride=1, padding="same"),
        torch.nn.ReLU(),
        torch.nn.Conv2d(512, 512, kernel_size=2, stride=1, padding="same"),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2),
        torch.nn.Conv2d(512, 512, kernel_size=2, stride=1, padding="same"),
        torch.nn.ReLU(),
        torch.nn.Conv2d(512, 512, kernel_size=2, stride=1, padding="same"),
        torch.nn.ReLU(),
        torch.nn.Conv2d(512, 512, kernel_size=2, stride=1, padding="same"),
        torch.nn.ReLU(),
    )

    self.fusion = torch.nn.Sequential(
        torch.nn.ConvTranspose3d(96, 64, kernel_size=4, stride=2, padding=1),
        torch.nn.ReLU(),
        torch.nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
        torch.nn.ReLU(),
        torch.nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1),
        torch.nn.ReLU(),
        torch.nn.ConvTranspose3d(16, 1, kernel_size=4, stride=2, padding=1),
        torch.nn.Sigmoid(),
    )

  def forward(self, x):
    x = x.unsqueeze(1) # [batch_size, 1, 64, 50]

    code = self.encoder(x) # [batch_size, 512, 4, 3]

    code = code.view(-1, 96, 4, 4, 4) # [batch_size, 96, 4, 4, 4]
    
    return self.fusion(code) # [batch_size, 1, 64, 64, 64]

  def training_step(self, batch, batch_idx):
    x, y_true = batch
    y_pred = self(x).reshape(y_true.shape)

    iou_value = iou(y_pred.round().bool(), y_true.round().bool())
    bce_value = torch.nn.functional.binary_cross_entropy(y_pred, y_true)

    loss = bce_value

    self.log("loss", loss)
    self.log("iou", iou_value, prog_bar=True)

    return {"loss": loss, "pred": y_pred}

  def validation_step(self, batch, batch_idx):
    x, y_true = batch
    y_pred = self(x).reshape(y_true.shape)

    iou_value = iou(y_pred.round().bool(), y_true.round().bool())
    bce_value = torch.nn.functional.binary_cross_entropy(y_pred, y_true)

    loss = bce_value

    self.log("val_loss", loss, prog_bar=True)
    self.log("val_iou", iou_value, prog_bar=True)

    return {"val_loss": loss, "pred": y_pred}

  def test_step(self, batch, batch_idx):
    x, y_true = batch
    y_pred = self(x).reshape(y_true.shape)
    
    iou_value = iou(y_pred.round().bool(), y_true.round().bool())
    bce_value = torch.nn.functional.binary_cross_entropy(y_pred, y_true)

    loss = bce_value

    self.log("test_loss", loss)
    self.log("test_iou", iou_value)

    return {"test_loss": loss, "pred": y_pred}
  
  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)