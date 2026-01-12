import torch

class Metric(object):
  """
  
  """
  def __init__(self) -> None:
    self.value = 0.0
    self.count = 0
  
  def update(self, value:float|torch.Tensor):
    self.count += 1
    self.value += value 
  
  def mean(self):
    m = self.value/max(self.count, 1)
    return m

def accuracy(pred, y):
  return (pred.argmax(dim=1) == y).float().mean()
