import torch
from typing import Optional, Callable

class Cell(torch.nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, *args, **kwargs):
    raise NotImplementedError

  def freeze(self):
    for p in self.parameters():
      p.requires_grad = False

  def unfreeze(self):
    for p in self.parameters():
      p.requires_grad = True

def param_count(model):
  c = 0
  for i in model.parameters(True):
    c+=i.numel()
  print(f"{model.__class__.__name__}: {c}")