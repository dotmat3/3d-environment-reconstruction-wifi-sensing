import torch

from pytorch_lightning import LightningModule

from utils import iou, dice
from collections import OrderedDict
from .positional_encoding import PositionalEncoding

class PatchEmbed(torch.nn.Module):
    """From https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/layers/patch_embed.py"""

    def __init__(self, image_size, in_channels, patch_size, embed_dim):
        super().__init__()

        self.grid_size = (image_size[0] // patch_size, image_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = torch.nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

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

class CSI2EnvViTB(LightningModule):
  def __init__(self, learning_rate, num_heads, num_encoder_layers, num_decoder_layers, dropout = 0.1, num_packets=64, d_model=50, embed_dim=128, representation_size=512, decoder_embed_dim=512, cnn_hidden_dim=64, patch_num=4, patch_size=16):
    super().__init__()
    self.save_hyperparameters()

    # --- Encoder ---
    self.embed = PatchEmbed((num_packets, d_model), 1, patch_size, embed_dim)
    
    self.pos_enc = PositionalEncoding(embed_dim, dropout=dropout)

    encoder_layer = torch.nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True)
    self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_encoder_layers)

    self.pre_logits = torch.nn.Sequential(OrderedDict([
                ('fc', torch.nn.Linear(embed_dim, representation_size)),
                ('act', torch.nn.Tanh())
            ]))

    # --- Decoder ---
    self.decoder_embed = torch.nn.Embedding(patch_num ** 3, decoder_embed_dim)

    decoder_layer = torch.nn.TransformerDecoderLayer(d_model=decoder_embed_dim, nhead=num_heads, batch_first=True)
    norm = torch.nn.LayerNorm(decoder_embed_dim)
    self.transformer_decoder = torch.nn.TransformerDecoder(decoder_layer, num_decoder_layers, norm=norm)

    self.voxel_upscaler = torch.nn.Sequential(
        torch.nn.ConvTranspose3d(decoder_embed_dim, cnn_hidden_dim, kernel_size=1),
        Res3dBlock(cnn_hidden_dim),
        Res3dBlock(cnn_hidden_dim),
        Res3dBlock(cnn_hidden_dim),
        torch.nn.ConvTranspose3d(cnn_hidden_dim, cnn_hidden_dim, kernel_size=4, stride=2, padding=1),
        torch.nn.ReLU(),
        torch.nn.ConvTranspose3d(cnn_hidden_dim, cnn_hidden_dim, kernel_size=4, stride=2, padding=1),
        torch.nn.ReLU(),
        torch.nn.ConvTranspose3d(cnn_hidden_dim, cnn_hidden_dim, kernel_size=4, stride=2, padding=1),
        torch.nn.ReLU(),
        torch.nn.ConvTranspose3d(cnn_hidden_dim, cnn_hidden_dim, kernel_size=4, stride=2, padding=1),
        torch.nn.ReLU(),
        torch.nn.ConvTranspose3d(cnn_hidden_dim, 1, kernel_size=1, stride=1),
        torch.nn.Sigmoid(),
    )

  def forward(self, x):
    batch_size, num_packets, num_subcarriers = x.size() # [batch_size, 64, 50]

    # Add channel dimension
    x = x.unsqueeze(1) # [batch_size, 1, 64, 50]

    # Extract patch embedding
    x = self.embed(x) # [batch_size, num_patches, embed_dim]
    
    # Add positional enconding
    x = self.pos_enc(x) # [batch_size, num_patches, embed_dim]

    # Apply transformer encoder
    x = self.encoder(x) # [batch_size, num_patches, embed_dim]

    # Final linear transformation
    context = self.pre_logits(x) # [batch_size, num_patches, representation_size]

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
    y_pred = self(x).reshape(y_true.shape)
    
    iou_value = iou(y_pred.round().bool(), y_true.round().bool())
    dice_value = dice(y_pred, y_true)
    bce_value = torch.nn.functional.binary_cross_entropy(y_pred, y_true)

    loss = dice_value

    self.log("loss", loss)
    self.log("iou", iou_value, prog_bar=True)
    self.log("bce", bce_value, prog_bar=True)

    return {"loss": loss, "pred": y_pred}

  def validation_step(self, batch, batch_idx):
    x, y_true = batch
    y_pred = self(x).reshape(y_true.shape)

    iou_value = iou(y_pred.round().bool(), y_true.round().bool())
    bce_value = torch.nn.functional.binary_cross_entropy(y_pred, y_true)
    # dice_value = dice(y_pred.round().bool(), y_true.round().bool())

    loss = bce_value

    self.log("val_loss", loss, prog_bar=True)
    self.log("val_iou", iou_value, prog_bar=True)
    # self.log("val_bce", bce_value, prog_bar=True)

    return {"val_loss": loss, "pred": y_pred}

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