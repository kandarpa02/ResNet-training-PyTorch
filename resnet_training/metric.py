import torch

class Metric:
  def __init__(self, device=None, dtype=torch.float32) -> None:
    self.device = device
    self.dtype = dtype
    self.reset()

  def reset(self):
    self._sum = torch.zeros((), device=self.device, dtype=self.dtype)
    self._count = torch.zeros((), device=self.device, dtype=torch.long)

  @torch.no_grad()
  def update(self, *args, **kwargs):
    self.update_rule(*args, **kwargs)

  def update_rule(self, *args, **kwargs):
    raise NotImplementedError
  
  def compute(self):
    return self._sum / self._count.clamp_min(1)
  

class Mean(Metric):
  def __init__(self, *, device=None):
    super().__init__(device=device, dtype=torch.float32)

  @torch.no_grad()
  def update_rule(self, value, n=1):
    """
    value: Tensor or float (already reduced)
    n: number of samples contributing to value
    """
    value = torch.as_tensor(value, device=self.device, dtype=self.dtype)
    self._sum += value * n
    self._count += n

  def compute(self):
    return self._sum / self._count.clamp_min(1)
  

class Accuracy(Metric):
  def __init__(self, *, device=None):
    super().__init__(device=device, dtype=torch.float32)

  @torch.no_grad()
  def update_rule(self, preds: torch.Tensor, targets: torch.Tensor):
    """
    preds: logits or probabilities, shape [N, C]
    targets: int64 labels, shape [N]
    """
    if preds.ndim > 1:
        preds = preds.argmax(dim=1)

    correct = (preds == targets).sum().to(self._sum.dtype)
    n = targets.numel()

    self._sum += correct
    self._count += n
