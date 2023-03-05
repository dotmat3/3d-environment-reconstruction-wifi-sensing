import torch

from pytorch_lightning import LightningModule

from .positional_encoding import PositionalEncoding
from utils import iou

class CSI2EnvTransformerA(LightningModule):
  def __init__(self, learning_rate, num_heads, num_encoder_layers, dropout = 0.1):
    super().__init__()
    self.save_hyperparameters()

    self.positional_encoding = PositionalEncoding(d_model=50, dropout=dropout)

    encoder_layer = torch.nn.TransformerEncoderLayer(d_model=50, nhead=num_heads, dropout=dropout, batch_first=True)
    self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_encoder_layers)

    self.fusion = torch.nn.Sequential(
        torch.nn.ConvTranspose3d(50, 64, kernel_size=4, stride=2, padding=1),
        torch.nn.ReLU(),
        torch.nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
        torch.nn.ReLU(),
        torch.nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1),
        torch.nn.ReLU(),
        torch.nn.ConvTranspose3d(16, 1, kernel_size=4, stride=2, padding=1),
        torch.nn.Sigmoid(),
    )

  def forward(self, x):
    batch_size, num_packets, num_subcarriers = x.size()

    x = self.positional_encoding(x) # [batch_size, 64, 50]

    code = self.encoder(x) # [batch_size, 64, 50]

    # code = torch.einsum("bps -> bsp", code)
    # code = code.swapaxes(1, 2)

    code = code.view(batch_size, num_subcarriers, 4, 4, 4) # [batch_size, 50, 4, 4, 4]
    
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

  def predict_step(self, batch, batch_idx):
    x, y_true = batch
    y_pred = self(x).reshape(y_true.shape)

    return y_pred

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)