import torch
from dataclasses import dataclass
from collections import OrderedDict
from typing import Any

@dataclass
class Checkpoint:
  last_epoch:int|None
  param_state:OrderedDict|None
  opt_state:dict|None
  scalar_state:dict|None

  def update(self, **kwargs):
    for k, v in kwargs.items():
      self.__setattr__(k, v)
