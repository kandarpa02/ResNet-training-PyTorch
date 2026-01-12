import torch

class Metric(object):
  def __init__(self) -> None:
    self.value = torch.tensor(0.)
    self.count = 0
  
  def update(self, value:float|torch.Tensor):
    self.count += 1
    self.value += value 
  
  def mean(self):
    m = self.value/(self.count if self.count != 0 else 1)
    return m.item()

def accuracy(pred, y):
  return (pred.argmax(dim=1) == y).float().mean()
