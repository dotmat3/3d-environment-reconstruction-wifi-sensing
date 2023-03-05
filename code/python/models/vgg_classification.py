from pytorch_lightning import LightningModule

import torch


class CSI2EnvVGGClassification(LightningModule):
  def __init__(self, learning_rate, num_classes=7):
    super().__init__()

    self.save_hyperparameters()

    self.vgg = torch.nn.Sequential(
        torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding="same"),
        torch.nn.ReLU(),
        torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding="same"),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2, stride=2),
        torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding="same"),
        torch.nn.ReLU(),
        torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding="same"),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2, stride=2),
        torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding="same"),
        torch.nn.ReLU(),
        torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding="same"),
        torch.nn.ReLU(),
        torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding="same"),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2, stride=2),
        torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding="same"),
        torch.nn.ReLU(),
        torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding="same"),
        torch.nn.ReLU(),
        torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding="same"),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2, stride=2),
        torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding="same"),
        torch.nn.ReLU(),
        torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding="same"),
        torch.nn.ReLU(),
        torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding="same"),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2, stride=2),
        torch.nn.Flatten(),
        torch.nn.Linear(1024, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, num_classes),
    )

  def forward(self, x):
    x = x.unsqueeze(1) # [batch_size, 1, 64, 50]

    return self.vgg(x) # [batch_size, num_classes]

  def training_step(self, batch, batch_idx):
    x, y_true = batch
    y_pred = self(x)

    loss = torch.nn.functional.cross_entropy(y_pred, y_true)

    self.log("loss", loss)

    return {"loss": loss, "pred": y_pred}

  def validation_step(self, batch, batch_idx):
    x, y_true = batch
    y_pred = self(x)

    loss = torch.nn.functional.cross_entropy(y_pred, y_true)

    self.log("val_loss", loss, prog_bar=True)

    return {"val_loss": loss, "pred": y_pred}

  def test_step(self, batch, batch_idx):
    x, y_true = batch
    y_pred = self(x)

    loss = torch.nn.functional.cross_entropy(y_pred, y_true)

    self.log("test_loss", loss)

    return {"test_loss": loss, "pred": y_pred}
  
  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)