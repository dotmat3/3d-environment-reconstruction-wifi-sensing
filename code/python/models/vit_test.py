import torch

from pytorch_lightning import LightningModule

from utils import iou, dice
from .positional_encoding import PositionalEncoding

class Res3dBlock(torch.nn.Module):

  def __init__(self, channels):
    super().__init__()

    self.model = torch.nn.Sequential(
        torch.nn.Conv3d(channels, channels, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv3d(channels, channels, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv3d(channels, channels, kernel_size=1)
    )

  def forward(self, x):
    return self.model(x) + x

class CSI2EnvViTC(LightningModule):
  def __init__(self, learning_rate, num_heads, num_encoder_layers, num_decoder_layers, dropout=0.1, num_packets=64, d_model=50, embed_dim=128, representation_size=512, decoder_embed_dim=512, patch_num=4):
    super().__init__()
    self.save_hyperparameters()

    # --- Encoder ---
    self.pos_enc = PositionalEncoding(embed_dim, dropout=dropout)

    encoder_layer = torch.nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True)
    self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_encoder_layers)

    self.proj = torch.nn.Linear(d_model, embed_dim)

    self.pre_logits = torch.nn.Linear(embed_dim, representation_size)

    # --- Decoder ---
    self.decoder_embed = torch.nn.Embedding(patch_num ** 3, decoder_embed_dim)

    decoder_layer = torch.nn.TransformerDecoderLayer(d_model=decoder_embed_dim, nhead=num_heads, batch_first=True)
    self.transformer_decoder = torch.nn.TransformerDecoder(decoder_layer, num_decoder_layers)

    self.voxel_upscaler = torch.nn.Sequential(
        torch.nn.Conv3d(decoder_embed_dim, 64, kernel_size=1, stride=1),
        Res3dBlock(64),
        Res3dBlock(64),
        Res3dBlock(64),
        torch.nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
        torch.nn.ReLU(),
        torch.nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1),
        torch.nn.ReLU(),
        torch.nn.ConvTranspose3d(16, 8, kernel_size=4, stride=2, padding=1),
        torch.nn.ReLU(),
        torch.nn.ConvTranspose3d(8, 1, kernel_size=4, stride=2, padding=1),
        torch.nn.Sigmoid(),
    )

  def forward(self, x):
    batch_size, num_packets, num_subcarriers = x.size() # [batch_size, 64, 50]

    # Extract patch embedding
    x = self.proj(x) # [batch_size, 64, embed_dim]

    # Apply transformer encoder
    x = self.encoder(x) # [batch_size, 64, embed_dim]

    # Final linear transformation
    context = self.pre_logits(x) # [batch_size, 64, representation_size]

    # Create decoder embedding
    init = torch.arange(self.hparams.patch_num ** 3, device=context.device)
    x = self.decoder_embed(init) # [patch_num ** 3, decoder_embed_dim]
    x = x.unsqueeze(0).repeat(batch_size, 1, 1) # [batch_size, patch_num ** 3, decoder_embed_dim]

    out = self.transformer_decoder(tgt=x, memory=context) # [batch_size, patch_num ** 3, decoder_embed_dim]

    out = out.reshape(batch_size, self.hparams.patch_num, self.hparams.patch_num, self.hparams.patch_num, self.hparams.decoder_embed_dim) # [batch_size, patch_num, patch_num, patch_num, decoder_embed_dim]
    out = out.permute(0, 4, 1, 2, 3)  # [batch_size, decoder_embed_dim, patch_num, patch_num, patch_num]

    out = self.voxel_upscaler(out) # [batch_size, 1, 64, 64, 64]

    return out

  def training_step(self, batch, batch_idx):
    x, y_true = batch
    y_pred = self(x).view(y_true.shape)

    loss = torch.nn.functional.binary_cross_entropy(y_pred, y_true)

    self.log("loss", loss)

    return {"loss": loss, "pred": y_pred}

#   def validation_step(self, batch, batch_idx):
#     x, y_true = batch
#     y_pred = self(x)

#     print(y_true.shape)
#     # t = y_true[:, :4, :4, :4]

#     loss = torch.nn.functional.binary_cross_entropy(y_pred, y_true)

#     self.log("val_loss", loss, prog_bar=True)

#     return {"val_loss": loss, "pred": y_pred}

  def test_step(self, batch, batch_idx):
    x, y_true = batch
    y_pred = self(x).reshape(y_true.shape)
    
    iou_value = iou(y_pred.round().bool(), y_true.round().bool())
    bce_value = torch.nn.functional.binary_cross_entropy(y_pred, y_true)
    # dice_value = dice(y_pred.round().bool(), y_true.round().bool())

    loss = bce_value

    self.log("test_loss", loss, prog_bar=True)
    self.log("test_iou", iou_value, prog_bar=True)
    # self.log("test_bce", bce_value, prog_bar=True)

    return {"test_loss": loss, "pred": y_pred}

  def predict_step(self, batch, batch_idx):
    x, y_true = batch
    y_pred = self(x).reshape(y_true.shape)

    return y_pred

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)