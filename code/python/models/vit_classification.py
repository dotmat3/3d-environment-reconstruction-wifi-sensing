import torch

from pytorch_lightning import LightningModule

from torchmetrics import Accuracy

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


class CSI2EnvViTClassification(LightningModule):
  def __init__(self, learning_rate, num_heads, num_encoder_layers, dropout = 0.1, num_packets=64, d_model=50, embed_dim=128, num_classes=7, patch_size=16):
    super().__init__()
    self.save_hyperparameters()

    # --- Encoder ---
    self.embed = PatchEmbed((num_packets, d_model), 1, patch_size, embed_dim)
    
    num_patches = self.embed.num_patches
    self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, embed_dim))
    self.pos_embed = torch.nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
    self.pos_drop = torch.nn.Dropout(p=dropout)

    encoder_layer = torch.nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True)
    self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_encoder_layers)

    self.linear_head = torch.nn.Sequential(torch.nn.LayerNorm(embed_dim), torch.nn.Linear(embed_dim, num_classes))

    self.train_acc = Accuracy(num_classes=num_classes)
    self.valid_acc = Accuracy(num_classes=num_classes)
    self.test_acc = Accuracy(num_classes=num_classes)

  def forward(self, x):
    batch_size, num_packets, num_subcarriers = x.size() # [batch_size, 64, 50]

    # Add channel dimension
    x = x.unsqueeze(1) # [batch_size, 1, 64, 50]

    # Extract patch embedding
    x = self.embed(x) # [batch_size, num_patches, embed_dim]

    # Add class token
    cls_tokens = self.cls_token.expand(batch_size, -1, -1) # [batch_size, 1, embed_dim]
    x = torch.cat((cls_tokens, x), dim=1) # [batch_size, num_patches + 1, embed_dim]
    
    # Add positional embedding
    x = x + self.pos_embed # [batch_size, num_patches + 1, embed_dim]
    x = self.pos_drop(x) # [batch_size, num_patches + 1, embed_dim]

    # Apply transformer encoder
    x = self.encoder(x) # [batch_size, num_patches + 1, embed_dim]

    # Take class token output
    x = x[:, 0, :]

    # Final linear transformation
    logits = self.linear_head(x) # [batch_size, num_classes]

    return logits

  def training_step(self, batch, batch_idx):
    x, y_true = batch
    y_pred = self(x)
    
    loss = torch.nn.functional.cross_entropy(y_pred, y_true)
    self.train_acc(torch.argmax(y_pred, dim=1), torch.argmax(y_true, dim=1))

    self.log("loss", loss)
    self.log("train_accuracy", self.train_acc, prog_bar=True)

    return {"loss": loss, "pred": y_pred}

  def validation_step(self, batch, batch_idx):
    x, y_true = batch
    y_pred = self(x)

    loss = torch.nn.functional.cross_entropy(y_pred, y_true)
    self.valid_acc(torch.argmax(y_pred, dim=1), torch.argmax(y_true, dim=1))
    
    self.log("val_loss", loss, prog_bar=True)
    self.log("val_accuracy", self.valid_acc, prog_bar=True)

    return {"val_loss": loss, "pred": y_pred}

  def test_step(self, batch, batch_idx):
    x, y_true = batch
    y_pred = self(x)
    
    loss = torch.nn.functional.cross_entropy(y_pred, y_true)
    self.test_acc(torch.argmax(y_pred, dim=1), torch.argmax(y_true, dim=1))
    
    self.log("test_loss", loss, prog_bar=True)
    self.log("test_accuracy", self.test_acc, prog_bar=True)

    return {"test_loss": loss, "pred": y_pred}

  def predict_step(self, batch, batch_idx):
    x, _ = batch
    y_pred = self(x)

    return y_pred

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)