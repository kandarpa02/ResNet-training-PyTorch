import timm
import torch
import torch.nn as nn
from typing import Any
import torchvision.models as models

class DeiT(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model = timm.create_model(
        "deit_small_patch16_224",
        pretrained=True,
        num_classes=10
    )
        
    def forward(self, x):
        return self.model(x)