from typing import Any

from .base import Cell
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import resnet18

class ResNet18(nn.Module):
  def __init__(self, *args: Any, **kwargs: Any) -> None:
    super().__init__(*args, **kwargs)
    net = resnet18()
    net.fc = nn.Linear(net.fc.in_features, 10)
    self.model = net

    def forward(self, x):
        return self.model(x)