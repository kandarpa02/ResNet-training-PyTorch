import torch
from torch.utils.data import Dataset
import os
import gdown
import numpy as np
import torch.nn.functional as F
from torchvision.transforms import functional as TF


URL = "https://drive.google.com/uc?id=1vqX5FKh7bmvxWL0CYjdo2xJFJ4mx3aqd"

CACHE_DIR = os.path.expanduser("~/.cache/ResNet-training-PyTorch")
FILENAME = "cifar10.npz"
FILEPATH = os.path.join(CACHE_DIR, FILENAME)


def cifar_download():
  os.makedirs(CACHE_DIR, exist_ok=True)

  if not os.path.exists(FILEPATH):
    gdown.download(URL, output=FILEPATH, quiet=False)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

class cifar10(Dataset):
  """
  split:
    (80,)       -> single split, use 80% of data, part ignored
    (80, 20)    -> two splits
                   part=0 -> first 80%
                   part=1 -> remaining 20%
  """
  def __init__(self, split: tuple, part: int = 0, train=True):
    self.train = train
    assert len(split) in (1, 2), "split must be (x,) or (x, y)"
    assert sum(split) == 100, "split percentages must sum to 100"

    if len(split) == 2:
      assert part in (0, 1), "part must be 0 or 1 for two-way split"

    cifar_download()

    with np.load(FILEPATH) as c10:
      images = c10["x"]   # (N, 3, 32, 32), float32
      labels = c10["y"]   # (N,), int64

    N = images.shape[0]

    n0 = int(split[0] / 100 * N)

    if len(split) == 1:
      # single split
      self.images = images[:n0]
      self.labels = labels[:n0]

    else:
      # two-way split
      if part == 0:
        self.images = images[:n0]
        self.labels = labels[:n0]
      else:
        self.images = images[n0:]
        self.labels = labels[n0:]

  def __len__(self):
    return self.images.shape[0]

  def __getitem__(self, idx):

    img = torch.from_numpy(self.images[idx]).float()

    if self.train:

        # Random horizontal flip
        if torch.rand(1).item() < 0.5:
            img = torch.flip(img, dims=[2])

        # Pad then random crop
        img = F.pad(img.unsqueeze(0), (4, 4, 4, 4), mode="reflect").squeeze(0)

        top = torch.randint(0, 9, ()).item()
        left = torch.randint(0, 9, ()).item()

        img = img[:, top:top + 32, left:left + 32]

        # Random rotation
        angle = float(torch.empty(1).uniform_(-15, 15))
        img = TF.rotate(img, angle)

        # Random brightness
        brightness = float(torch.empty(1).uniform_(0.8, 1.2))
        img = torch.clamp(img * brightness, 0.0, 1.0)

        # Random contrast
        contrast = float(torch.empty(1).uniform_(0.8, 1.2))
        mean = img.mean(dim=(1, 2), keepdim=True)
        img = torch.clamp((img - mean) * contrast + mean, 0.0, 1.0)

    img = F.interpolate(
        img.unsqueeze(0),
        size=(224, 224),
        mode="bilinear",
        align_corners=False
    ).squeeze(0)

    img = (img - IMAGENET_MEAN) / IMAGENET_STD

    label = torch.tensor(self.labels[idx], dtype=torch.long)

    return img, label