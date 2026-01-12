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

  def state_dict(self):
    return {
      "last_epoch": self.last_epoch,
      "param_state": self.param_state,
      "opt_state": self.opt_state,
      "scalar_state": self.scalar_state,
    }

  @classmethod
  def from_state_dict(cls, d):
    return cls(**d)