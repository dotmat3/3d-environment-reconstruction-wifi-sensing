from pytorch_lightning.core.datamodule import LightningDataModule

import numpy as np
import torch

import shutil
import os

import gdown
from zipfile import ZipFile

from preprocessing import fillmissing, preprocess_sample
from utils import load_csi, load_environment

class CSIDataset(torch.utils.data.Dataset):
  """Class in charge of downloading, loading and processing the CSI data and the 3D environments"""

  MODELS_GDRIVE_ID = "1kU7ZRPkKyL0SxW7934J3TpH9cUCywoTg"
  CSI_GDRIVE_ID = "1O6TaTrLBNFqpT_sS3Vn5CzPZpGhJ9veT"

  def __init__(self, meters, num_packets, ntx, nrx, num_subcarriers,
               output_shape, classification=False, max_samples=None, max_envs=None, normalize=False, data_dir="dataset", x="both", download=False, load=True,
               quiet=True):
    super().__init__()

    self.meters = meters
    self.num_packets = num_packets
    self.ntx = ntx
    self.nrx = nrx
    self.num_subcarriers = num_subcarriers
    self.output_shape = output_shape
    self.normalize = normalize
    self.x = x

    self.classification = classification

    self.max_samples = max_samples
    self.max_envs = max_envs

    self.data_dir = data_dir
    self.quiet = quiet

    self.complete_dataset = None
    self.samples_per_class = {}
    self.class_names = None

    if download:
      self.__download()

    if load:
      self.__load_and_process()

  def __download(self):
    if not self.quiet:
      print("Downloading CSIDataset...")

    complete_file_path = os.path.join(self.data_dir, f".complete")

    # Check if the download was already completed
    if os.path.exists(complete_file_path):
      if not self.quiet:
        print("Already downloaded")
      return

    # If the dataset folder doesn't exist make it
    if not os.path.exists(self.data_dir):
      os.makedirs(self.data_dir)

    # Download models data
    self.__download_from_gdrive(os.path.join(self.data_dir, "models"), self.MODELS_GDRIVE_ID)

    # Download csi data
    self.__download_from_gdrive(os.path.join(self.data_dir, "csi"), self.CSI_GDRIVE_ID)

    # Create complete file to avoid other downloads
    open(complete_file_path, "a").close()

    if not self.quiet:
      print("Dataset downloaded!")

  def __download_from_gdrive(self, target_dir, id, unzip=True):
    # Download using gdown
    url = "https://drive.google.com/uc?id={}&confirm=t".format(id)
    output = gdown.download(url, quiet=self.quiet, use_cookies=False, resume=True)

    if not self.quiet:
      print(f"Downloaded {output}")

    # If targe directory doesn't exists make it
    if not os.path.exists(target_dir):
      os.makedirs(target_dir)

    # If requested to do so and the file is a ZIP file
    if unzip and os.path.splitext(output)[-1] == ".zip":
      # Unzip the file
      with ZipFile(output, "r") as f:
        f.extractall(target_dir)

      # Remove the original zip file after unzipping
      os.remove(output)

      if not self.quiet:
        print(f"{output} unzipped in {target_dir}")
    else:
      # Otherwise simply move the file to the target dir
      shutil.move(output, target_dir)

      if not self.quiet:
        print(f"{output} moved to {target_dir}")

  def __load_and_process(self):
    if not self.quiet:
      print("Loading and processing dataset...")

    if self.complete_dataset is not None:
      if not self.quiet:
        print("Dataset already loaded and processed!")
      return

    data = []
    min_value = np.inf
    max_value = -np.inf

    csi_folder = os.path.join(self.data_dir, "csi")
    models_folder = os.path.join(self.data_dir, "models")
    rooms_folder = os.path.join(csi_folder, str(self.meters) + "mt")

    rooms = sorted(os.listdir(rooms_folder))
    num_rooms = self.max_envs if self.max_envs else len(rooms)

    if self.classification:
      self.class_names = {}

    # For each room
    for room_name in rooms:
      # Check if we reached the maximum number of environments
      if self.max_envs and len(self.samples_per_class) == self.max_envs:
        break

      if not self.quiet:
        current_room = len(self.samples_per_class)
        print(f"\rLoading...{room_name} ({current_room}/{num_rooms})", end="")
      
      room_path = os.path.join(rooms_folder, room_name)
      model_path = os.path.join(models_folder, room_name)
      env_path = os.path.join(model_path, room_name + ".vox")

      if not self.classification:
        # Load the environment
        env = load_environment(env_path, self.output_shape)
      else:
        index = len(self.samples_per_class)
        env = np.zeros(num_rooms, dtype=np.float32)
        env[index] = 1.

        self.class_names[index] = room_name
      
      self.samples_per_class[room_name] = 0

      # For each experiment session
      for session in os.listdir(room_path):
        session_path = os.path.join(room_path, session)

        # For experiment of that session
        for csv_file in os.listdir(session_path):
          # Check if we reached the maximum number of samples per environment
          if self.max_samples and self.samples_per_class[room_name] == self.max_samples:
            break

          csv_path = os.path.join(session_path, csv_file)

          # Load the CSI_DATA
          csi = load_csi(csv_path)

          # Preprocess the data using the specified arguments
          _, _, final_phase, _, filtered_median_amplitude = preprocess_sample(csi, self.num_packets, self.ntx, self.nrx, self.num_subcarriers)

          # Create amplitude and phase tensors for pytorch
          if self.x == "amplitude" or self.x == "both":
            amplitude = fillmissing(filtered_median_amplitude)
            x_amplitude = torch.from_numpy(amplitude.astype(np.float32))

          if self.x == "phase" or self.x == "both":
            final_phase = np.median(final_phase, [1, 2])
            final_phase = final_phase[np.any(final_phase, 1), :]
            phase = fillmissing(final_phase)
            x_phase = torch.from_numpy(phase.astype(np.float32))
          
          if self.x == "amplitude":
            x = x_amplitude
          elif self.x == "phase":
            x = x_phase
          else:
            x = torch.stack([x_amplitude, x_phase])
          
          y = torch.tensor(env, dtype=torch.float32)

          data.append((x, y))
          self.samples_per_class[room_name] += 1

          # Compute min and max for later normalization
          if x.min() < min_value:
            min_value = x.min()

          if x.max() > max_value:
            max_value = x.max()

    # Normalization
    if self.normalize:
      if not self.quiet:
        print(f"\rNormalizing in range 0-1")
      
      def normalize(x):
        return (x - min_value) / (max_value - min_value)

      self.complete_dataset = [(normalize(x), y) for x, y in data]
    else:
      self.complete_dataset = data
    
    if not self.quiet:
      print("Dataset loaded and processed!")

  def __getitem__(self, idx):
    return self.complete_dataset[idx]

  def __len__(self):
    return len(self.complete_dataset)

class CSIDataModule(LightningDataModule):
  """Handle the download, loading and processing of :class:`~dataset.CSIDataset`. It also provides the train, val and test data loaders."""
  
  def __init__(self, meters, num_packets, ntx, nrx, num_subcarriers,
               output_shape, train_size, valid_size, classification=False, max_samples=None, max_envs=None, normalize=False, data_dir="dataset", x="both", batch_size=8, quiet=True):
    super().__init__()
    self.save_hyperparameters(ignore="quiet")

    self.meters = meters
    self.num_packets = num_packets
    self.ntx = ntx
    self.nrx = nrx
    self.num_subcarriers = num_subcarriers
    self.output_shape = output_shape
    self.data_dir = data_dir
    self.normalize = normalize
    self.x = x

    self.classification = classification

    self.max_samples = max_samples
    self.max_envs = max_envs

    self.train_size = train_size
    self.valid_size = valid_size
    self.batch_size = batch_size
    self.quiet = quiet

    self.dataset = None

    self.train_dataset = None
    self.valid_dataset = None
    self.test_dataset = None

  def prepare_data(self):
    # Dataset already loaded and splitted
    if self.train_dataset is not None:
      return
    
    # Download dataset
    CSIDataset(self.meters, self.num_packets, self.ntx, self.nrx, self.num_subcarriers, self.output_shape, self.data_dir, download=True, load=False, quiet=self.quiet)

  def setup(self, stage):
    # Dataset already loaded and splitted
    if self.train_dataset is not None:
      return

    # Load dataset
    args = {
      "meters": self.meters,
      "num_packets": self.num_packets,
      "ntx": self.ntx,
      "nrx": self.nrx,
      "num_subcarriers": self.num_subcarriers,
      "output_shape": self.output_shape,
      "classification": self.classification,
      "max_samples": self.max_samples,
      "max_envs": self.max_envs,
      "normalize": self.normalize,
      "data_dir": self.data_dir,
      "x": self.x,
      "download": False,
      "load": True,
      "quiet": self.quiet
    }
    self.dataset = CSIDataset(**args)
    
    # Split dataset
    length = len(self.dataset)
    train_size = int(self.train_size * length)
    valid_size = int(self.valid_size * length)
    test_size = length - train_size - valid_size

    generator = torch.Generator()
    generator.manual_seed(42)

    self.train_dataset, self.valid_dataset, self.test_dataset = torch.utils.data.random_split(self.dataset, [train_size, valid_size, test_size], generator=generator)

  def train_dataloader(self):
    return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)

  def val_dataloader(self):
    return torch.utils.data.DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)
  
  def test_dataloader(self):
    return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)