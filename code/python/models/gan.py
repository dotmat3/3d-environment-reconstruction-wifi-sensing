import torch

from pytorch_lightning import LightningModule

from .transformerA import CSI2EnvTransformerA
from .vgg import CSI2EnvVGG
from utils import iou

class Generator(torch.nn.Module):
    def __init__(self, learning_rate, num_heads, num_encoder_layers, dropout):
    # def __init__(self, learning_rate):
      super().__init__()

      self.model = CSI2EnvTransformerA(learning_rate, num_heads, num_encoder_layers, dropout)
      # self.model = CSI2EnvVGG(learning_rate)

    def forward(self, z):
      return self.model(z)

class Discriminator(torch.nn.Module):
    def __init__(self):
      super().__init__()

      self.model = torch.nn.Sequential(
          torch.nn.Conv3d(1, 8, kernel_size=2, stride=2),
          torch.nn.LeakyReLU(0.2, inplace=True),
          torch.nn.Conv3d(8, 16, kernel_size=2, stride=2),
          torch.nn.LeakyReLU(0.2, inplace=True),
          torch.nn.Conv3d(16, 32, kernel_size=2, stride=2),
          torch.nn.LeakyReLU(0.2, inplace=True),
          torch.nn.Conv3d(32, 64, kernel_size=2, stride=2),
          torch.nn.LeakyReLU(0.2, inplace=True),
          torch.nn.Conv3d(64, 64, kernel_size=2, stride=2),
          torch.nn.LeakyReLU(0.2, inplace=True),
          torch.nn.Flatten(),
          torch.nn.Linear(512, 256),
          torch.nn.ReLU(),
          torch.nn.Linear(256, 128),
          torch.nn.ReLU(),
          torch.nn.Linear(128, 64),
          torch.nn.ReLU(),
          torch.nn.Linear(64, 32),
          torch.nn.ReLU(),
          torch.nn.Linear(32, 16),
          torch.nn.ReLU(),
          torch.nn.Linear(16, 1),
          torch.nn.Sigmoid(),
      )

    def forward(self, x):
      return self.model(x)

class CSI2EnvGAN(LightningModule):
  def __init__(self, learning_rate, num_heads, num_encoder_layers, dropout):
  # def __init__(self, learning_rate):
    super().__init__()
    self.save_hyperparameters()

    # networks
    self.generator = Generator(self.hparams.learning_rate, self.hparams.num_heads, self.hparams.num_encoder_layers, self.hparams.dropout)
    # self.generator = Generator(self.hparams.learning_rate)
    self.discriminator = Discriminator()

  def forward(self, z):
    return self.generator(z)

  def adversarial_loss(self, y_pred, y_true):
    return torch.nn.functional.binary_cross_entropy(y_pred, y_true)

  def content_loss(self, y_pred, y_true):
    return torch.nn.functional.binary_cross_entropy(y_pred, y_true)

  def generator_step(self, x, y_true):
    gen_output = self(x)
    disc_output = self.discriminator(gen_output)

    valid = torch.ones(x.size(0), 1)
    valid = valid.type_as(x)

    adv_loss = self.adversarial_loss(disc_output, valid)
    content_loss = self.content_loss(gen_output.squeeze(), y_true)
    iou_value = iou(gen_output.squeeze().round().bool(), y_true.round().bool())
    
    gen_loss = adv_loss + content_loss

    self.log("adv_loss", adv_loss, prog_bar=True)
    self.log("content_loss", content_loss, prog_bar=True)
    self.log("iou", iou_value, prog_bar=True)
    self.log("gen_loss", gen_loss, prog_bar=True)

    return {"loss": gen_loss, "pred": gen_output}

  def discriminator_step(self, x, y_true):
    valid = torch.ones(x.size(0), 1)
    valid = valid.type_as(x)

    gen_output = self(x)

    fake_disc_output = self.discriminator(gen_output)
    real_disc_output = self.discriminator(y_true.unsqueeze(1))

    fake = torch.zeros(x.size(0), 1)
    fake = fake.type_as(x)

    real_loss = self.adversarial_loss(real_disc_output, valid)
    fake_loss = self.adversarial_loss(fake_disc_output, fake)

    disc_loss = (real_loss + fake_loss) / 2

    self.log("disc_loss", disc_loss, prog_bar=True)

    return {"loss": disc_loss, "pred": gen_output}

  def training_step(self, batch, batch_idx, optimizer_idx):
    x, y_true = batch

    # Train generator
    if optimizer_idx == 0:
        return self.generator_step(x, y_true)

    # Train discriminator
    if optimizer_idx == 1:
        return self.discriminator_step(x, y_true)

  def validation_step(self, batch, batch_idx):
    x, y_true = batch

    gen_output = self(x)

    real_output = self.discriminator(y_true.unsqueeze(1))
    fake_output = self.discriminator(gen_output)

    valid = torch.ones(x.size(0), 1)
    valid = valid.type_as(x)

    adv_loss = self.adversarial_loss(fake_output, valid)
    content_loss = self.content_loss(gen_output.squeeze(), y_true)
    iou_value = iou(gen_output.squeeze().round().bool(), y_true.round().bool())
    
    gen_loss = adv_loss + content_loss

    self.log("val_adv_loss", adv_loss, prog_bar=True)
    self.log("val_content_loss", content_loss, prog_bar=True)
    self.log("val_iou", iou_value, prog_bar=True)
    self.log("val_gen_loss", gen_loss, prog_bar=True)

    fake = torch.zeros(x.size(0), 1)
    fake = fake.type_as(x)

    real_loss = self.adversarial_loss(real_output, valid)
    fake_loss = self.adversarial_loss(fake_output, fake)

    disc_loss = (real_loss + fake_loss) / 2

    self.log("val_disc_loss", disc_loss, prog_bar=True)

    return {"val_gen_loss": gen_loss, "val_disc_loss": disc_loss, "pred": gen_output}
  
  def test_step(self, batch, batch_idx):
    x, y_true = batch

    gen_output = self(x)

    real_output = self.discriminator(y_true.unsqueeze(1))
    fake_output = self.discriminator(gen_output)

    valid = torch.ones(x.size(0), 1)
    valid = valid.type_as(x)

    adv_loss = self.adversarial_loss(fake_output, valid)
    content_loss = self.content_loss(gen_output.squeeze(), y_true)
    
    gen_loss = adv_loss + content_loss

    self.log("test_adv_loss", adv_loss, prog_bar=True)
    self.log("test_content_loss", content_loss, prog_bar=True)
    self.log("test_gen_loss", gen_loss, prog_bar=True)

    fake = torch.zeros(x.size(0), 1)
    fake = fake.type_as(x)

    real_loss = self.adversarial_loss(real_output, valid)
    fake_loss = self.adversarial_loss(fake_output, fake)

    disc_loss = (real_loss + fake_loss) / 2

    self.log("test_disc_loss", disc_loss, prog_bar=True)

  def predict_step(self, batch, batch_idx):
    x, _ = batch

    gen_output = self(x)

    return gen_output

  def configure_optimizers(self):
    opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.learning_rate)
    opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.learning_rate)
    return [opt_g, opt_d], []