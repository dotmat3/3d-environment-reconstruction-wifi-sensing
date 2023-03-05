from pytorch_lightning.callbacks.progress.base import ProgressBarBase
from pytorch_lightning.callbacks import Callback

from visualizer import export_data

import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb

import csv
import json
import time
import os

def load_csi(path):
    """Load the CSI data contained in a CSV file into a numpy array."""

    # Open file
    with open(path, "r") as f:
        data = []
        # Read content as csv
        reader = csv.reader(f, delimiter=",")

        for row in reader:
            # Get column corresponding to CSI_DATA
            array = row[-1].replace("[", "").replace("]", "").strip().split(" ")

            data.append(np.array(array, dtype=np.int32))

    return np.array(data)


def load_vox(path):
    """Load VOX object created using https://voxelator.com/."""

    # Open file
    with open(path, "r") as f:
        # Read content as json
        mesh_data = json.load(f)

    palette = mesh_data["palette"]
    voxels = mesh_data["layers"][0]["voxels"]

    return voxels, palette


def load_environment(path, shape):
    """Load a 3D environment saved as VOX object. The result is a numpy array with the provided shape containing 1 where a voxel is present or 0 otherwise."""

    voxels, _ = load_vox(path)

    # Create empty grid
    grid = np.zeros(shape)

    # Fill the grid with the voxels
    for voxel in voxels:
        x, y, z, _ = voxel
        grid[z, y, x] = 1

    return grid



class KerasProgressBar(ProgressBarBase):
    """Custom progress bar inspired by the Keras progress bar."""

    def __init__(self, hide_loss=False, hide_v_num=True):
      super().__init__()

      self.hide_v_num = hide_v_num
      self.hide_loss = hide_loss

      self.enable = True

    def _get_bar(self, value, total, len=20):
      progress = value * len // total
      remaining = len - progress
      progress_str = "=" * progress
      remaining_str = " " * remaining
      return "[" + progress_str + remaining_str + "]"

    def disable(self):
      self.enable = False

    def get_metrics(self, trainer, pl_model):
      items = super().get_metrics(trainer, pl_model)
      
      if self.hide_v_num:
        items.pop("v_num", None)

      if self.hide_loss:
        items.pop("loss", None)

      return items

    def on_train_epoch_start(self, trainer, pl_module):
      super().on_train_epoch_start(trainer, pl_module)

      self._start = time.time()

      print(f"Epoch {trainer.current_epoch + 1}/{trainer.max_epochs}")

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
      super().on_train_batch_start(trainer, pl_module, batch, batch_idx)

      if self.train_batch_idx == 0:
        self._time_after_first_step = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
      super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

      now = time.time()
      progress = self._get_bar(self.train_batch_idx, self.total_train_batches)

      time_per_unit = (now - self._time_after_first_step) / self.train_batch_idx
      eta = time_per_unit * (self.total_train_batches - self.train_batch_idx)

      if eta > 3600:
        eta_format = "%d:%02d:%02d" % (eta // 3600, (eta % 3600) // 60, eta % 60)
      elif eta > 60:
        eta_format = "%d:%02d" % (eta // 60, eta % 60)
      else:
        eta_format = "%ds" % eta

      metrics = " - ".join(f"{key}: {float(value):.4f}" for key, value in self.get_metrics(trainer, pl_module).items())

      print(f"\r{self.train_batch_idx}/{self.total_train_batches} {progress} - ETA: {eta_format} - {metrics}", end="")
    
    def on_train_epoch_end(self, trainer, pl_module):
      super().on_train_epoch_end(trainer, pl_module)

      now = time.time()
      progress = self._get_bar(self.train_batch_idx, self.total_train_batches)
      elapsed_time = now - self._start
      time_per_unit = (now - self._time_after_first_step) / self.train_batch_idx

      if time_per_unit >= 1 or time_per_unit == 0:
        formatted = ' %.0fs/step' % time_per_unit
      elif time_per_unit >= 1e-3:
        formatted = ' %.0fms/step' % (time_per_unit * 1e3)
      else:
        formatted = ' %.0fus/step' % (time_per_unit * 1e6)

      metrics = " - ".join(f"{key}: {float(value):.4f}" for key, value in self.get_metrics(trainer, pl_module).items())

      print(f"\r{self.train_batch_idx}/{self.total_train_batches} {progress} - {elapsed_time:.0f}s {formatted} - {metrics}")

class Accumulator():

  def __init__(self, max_size: int = None):
    
    self.data = []

    self.max_size = max_size

  def append(self, element):
    if len(self.data) >= self.max_size:
      return False

    self.data.append(element)
    
    return True

  def clear(self):
    self.data.clear()

  def __getitem__(self, idx):
    return self.data[idx]

  def __iter__(self):
    return iter(self.data)

  def __len__(self):
    return len(self.data)



class LogPredictionsCallback(Callback):
    """Custom callback used to save the predictions, and associated ground truth, to Weights and Biases during the trainig."""

    def __init__(self, save_glb_every_n_epochs, save_images_every_n_epochs, num_glb_predictions, num_images_predictions, on_multiple_outputs_take=0):
      super().__init__()

      self.save_glb_every_n_epochs = save_glb_every_n_epochs
      self.save_images_every_n_epochs = save_images_every_n_epochs

      self.num_glb_predictions = num_glb_predictions
      self.num_images_predictions = num_images_predictions

      self.on_multiple_outputs_take = on_multiple_outputs_take

      self.train_glb_epoch_outputs = Accumulator(num_glb_predictions)
      self.train_images_epoch_outputs = Accumulator(num_images_predictions)

      self.validate_glb_epoch_outputs = Accumulator(num_glb_predictions)
      self.validate_images_epoch_outputs = Accumulator(num_images_predictions)

    def _postprocess_output(self, output):
      return output.squeeze().round().detach().cpu().numpy().astype(bool)

    def _accumulate_batches(self, accumulator, batch, outputs):
      batch_x, batch_y = batch

      # Postprocess and collect other predictions of the current batch
      for x, y, output in zip(batch_x, batch_y, outputs["pred"]):
        pred = self._postprocess_output(output)
        true = self._postprocess_output(y)
        
        accumulator.append((x, true, pred))

    def _save_glb_on_wandb(self, accumulator, step):
      ground_truth = []
      predictions = []

      # For each prediction collected
      for index, data in enumerate(accumulator):
        _, true, pred = data
        ground_truth.append(self.make_wandb_object_3d(true, "true", index))
        predictions.append(self.make_wandb_object_3d(pred, "pred", index))

      wandb.log({f"{step}-3d-ground-truth": ground_truth, f"{step}-3d-predictions": predictions }, commit=False)

      # Clear temp files
      for index in range(len(accumulator)):
        if os.path.exists(f"temp_true_{index}.glb"):
          os.remove(f"temp_true_{index}.glb")
        if os.path.exists(f"temp_pred_{index}.glb"):
          os.remove(f"temp_pred_{index}.glb")
    
    def _save_images_on_wandb(self, accumulator, step):
      image_ground_truth = []
      image_predictions = []

      # For each prediction collected
      for _, true, pred in accumulator:
        image_ground_truth.append(self.make_wandb_top_view_image(true))
        image_predictions.append(self.make_wandb_top_view_image(pred))

      wandb.log({f"{step}-image-ground-truth": image_ground_truth, f"{step}-image-predictions": image_predictions}, commit=False)


    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
      batch_x, batch_y = batch

      # Handle multple outputs models
      if type(outputs) == list:
        outputs = outputs[self.on_multiple_outputs_take]

      # Save npz files of the first batch each epoch
      if batch_idx == 0:
        self.save_npz(trainer.current_epoch, (batch_x, batch_y, outputs["pred"]), train=True)

      # Accumulate glb samples
      if (trainer.current_epoch + 1) % self.save_glb_every_n_epochs == 0:
        self._accumulate_batches(self.train_glb_epoch_outputs, batch, outputs)

      # Accumulate images samples
      if (trainer.current_epoch + 1) % self.save_images_every_n_epochs == 0:
        self._accumulate_batches(self.train_images_epoch_outputs, batch, outputs)
      
    def on_train_epoch_end(self, trainer, pl_module):
      # Save glb
      if (trainer.current_epoch + 1) % self.save_glb_every_n_epochs == 0:
        self._save_glb_on_wandb(self.train_glb_epoch_outputs, "train")

      # Save images
      if (trainer.current_epoch + 1) % self.save_images_every_n_epochs == 0:
        self._save_images_on_wandb(self.train_images_epoch_outputs, "train")

      self.train_glb_epoch_outputs.clear()
      self.train_images_epoch_outputs.clear()


    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
      batch_x, batch_y = batch

      # Handle multple outputs models
      if type(outputs) == list:
        outputs = outputs[self.on_multiple_outputs_take]

      # Save npz files of the first batch each epoch
      if batch_idx == 0:
        self.save_npz(trainer.current_epoch, (batch_x, batch_y, outputs["pred"]), train=True)

      # Accumulate glb samples
      if (trainer.current_epoch + 1) % self.save_glb_every_n_epochs == 0:
        self._accumulate_batches(self.validate_glb_epoch_outputs, batch, outputs)

      # Accumulate images samples
      if (trainer.current_epoch + 1) % self.save_images_every_n_epochs == 0:
        self._accumulate_batches(self.validate_images_epoch_outputs, batch, outputs)
      
    def on_validation_epoch_end(self, trainer, pl_module):
      # Save glb
      if (trainer.current_epoch + 1) % self.save_glb_every_n_epochs == 0:
        self._save_glb_on_wandb(self.validate_glb_epoch_outputs, "validate")

      # Save images
      if (trainer.current_epoch + 1) % self.save_images_every_n_epochs == 0:
        self._save_images_on_wandb(self.validate_images_epoch_outputs, "validate")

      self.validate_glb_epoch_outputs.clear()
      self.validate_images_epoch_outputs.clear()


    def make_wandb_object_3d(self, data, type, index):
      # If the data is empty, return an empty 3d object
      if data.sum() == 0:
        return wandb.Object3D("visualizer/empty.glb", file_type="glb")

      # Export the data in the glb format
      glb = export_data(data, file_type="glb")
      filename = f"temp_{type}_{index}.glb"

      # Store the data in a temp file
      with open(filename, "wb") as f:
        f.write(glb)

      # Create the 3D object from the file
      return wandb.Object3D(filename, file_type="glb")
    
    def make_wandb_top_view_image(self, data):
      image = data.squeeze().sum(axis=1)

      fig = plt.figure()
      plt.imshow(image)
      plt.axis("off")
      plt.tight_layout()

      matplotlib_image = matplotlib_figure_to_image(fig)

      plt.close()

      return wandb.Image(matplotlib_image)

    def save_npz(self, epoch, data, train=True):
      phase_name = "train" if train else "valid"

      # Create the dir path of the npz file
      dir_path = os.path.join(wandb.run.dir, "outputs")
      os.makedirs(dir_path, exist_ok=True)

      filename = f"{phase_name}_{epoch}.npz"
      file_path = os.path.join(dir_path, filename)

      # Process data
      x, y, pred = data
      x = x.detach().cpu()
      y = self._postprocess_output(y)
      pred = self._postprocess_output(pred)
      
      # Compress data and save it to wandb
      np.savez_compressed(file_path, x=x, y=y, pred=pred)
      wandb.save(file_path, base_path=wandb.run.dir, policy="now")


def iou(y_pred, y_true, smooth=1e-6):
  intersection = (y_pred & y_true).sum((1, 2, 3))
  union = (y_pred | y_true).sum((1, 2, 3))

  result = (intersection + smooth) / (union + smooth)

  return result.mean()

def dice(y_pred, y_true, smooth=1, p=2):
  # flatten
  y_pred = y_pred.contiguous().view(y_pred.shape[0], -1)
  y_true = y_true.contiguous().view(y_true.shape[0], -1)

  num = torch.sum(torch.mul(y_pred, y_true), dim=1) + smooth
  den = torch.sum(y_pred.pow(p) + y_true.pow(p), dim=1) + smooth

  loss = 1 - num / den

  return loss.mean()

def matplotlib_figure_to_image(figure):
  from matplotlib.backends.backend_agg import FigureCanvasAgg

  canvas = FigureCanvasAgg(figure)
  canvas.draw()
  rgba = np.asarray(canvas.buffer_rgba())

  return rgba