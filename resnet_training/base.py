import torch
from typing import Optional, Callable

class Module_(torch.nn.Module):
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

def compact(obj:Module_):
  def run(*args):
    out = None
    for i, f in enumerate(obj.__dict__['_modules'].values()):
      if i==0:
        out = f(*args)
      else:
        out = f(out)
    return out
  return run
