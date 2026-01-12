import torch
from dataclasses import dataclass
from collections import OrderedDict
from typing import Any

@dataclass
class Checkpoint:
  """
  Checkpoint container for storing and restoring training state.

  This object is used to resume training by keeping track of model parameters,
  optimizer state, AMP scaler state, and the last completed epoch.

  Arguments:
    last_epoch (Optional[int]): Index of the last completed training epoch,
      or None if training starts from scratch.
    param_state (Optional[OrderedDict]): Model parameter state dictionary
      (as returned by model.state_dict()).
    opt_state (Optional[dict]): Optimizer state dictionary
      (as returned by optimizer.state_dict()).
    scalar_state (Optional[dict]): GradScaler state dictionary used for
      automatic mixed precision (AMP), if enabled.

  Example:
    ```python
    ckpt = Checkpoint(
        last_epoch=None,
        param_state=None,
        opt_state=None,
        scalar_state=None,
    )
    ```
  """
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