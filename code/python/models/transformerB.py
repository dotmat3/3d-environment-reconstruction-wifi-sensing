import torch

from pytorch_lightning import LightningModule
from .positional_encoding import PositionalEncoding

class CSI2EnvTransformerB(LightningModule):
  def __init__(self, learning_rate, num_heads, num_encoder_layers, dropout = 0.1):
    super().__init__()

    self.positional_encoding = PositionalEncoding(d_model=50, dropout=dropout)

    encoder_layer = torch.nn.TransformerEncoderLayer(d_model=50, nhead=num_heads, dropout=dropout, batch_first=True)
    self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_encoder_layers)

    self.fusion = torch.nn.Sequential(
        torch.nn.Conv3d(1, 16, kernel_size=4, stride=1, padding="same"),
        torch.nn.ReLU(),
        torch.nn.Conv3d(16, 32, kernel_size=4, stride=1, padding="same"),
        torch.nn.ReLU(),
        torch.nn.Conv3d(32, 64, kernel_size=4, stride=1, padding="same"),
        torch.nn.ReLU(),
        torch.nn.Conv3d(64, 32, kernel_size=4, stride=1, padding="same"),
        torch.nn.ReLU(),
        torch.nn.Conv3d(32, 1, kernel_size=4, stride=1, padding="same"),
        torch.nn.Sigmoid(),
    )

  def forward(self, x):
    x = self.positional_encoding(x)

    code = self.transformer_encoder(x) # [batch_size, 64, 50]

    code = code.view(-1, 1, 64, 50, 64) # [batch_size, 1, 64, 50, 64]

    return self.fusion(code) # [batch_size, 1, 64, 50, 64]

  def training_step(self, batch, batch_idx):
    x, y_true = batch
    y_pred = self(x).squeeze()
    loss = torch.nn.functional.binary_cross_entropy(y_pred, y_true)
    self.log("loss", loss)
    return loss

  def validation_step(self, batch, batch_idx):
    x, y_true = batch
    y_pred = self(x).squeeze()
    loss = torch.nn.functional.binary_cross_entropy(y_pred, y_true)
    self.log("val_loss", loss)
    return loss

  def test_step(self, batch, batch_idx):
    x, y_true = batch
    y_pred = self(x).squeeze()
    loss = torch.nn.functional.binary_cross_entropy(y_pred, y_true)
    self.log("test_loss", loss)
    return loss
  
  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)