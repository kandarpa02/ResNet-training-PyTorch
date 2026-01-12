import torch

class Metric(object):
  def __init__(self) -> None:
    self.value = torch.tensor(0.)
    self.count = 1
  
  def update(self, value:float|torch.Tensor):
    self.count += 1
    self.value += value 
  
  def mean(self):
    m = self.value/self.count
    return m.item()

def accuracy(pred, y):
  return torch.mean(torch.argmax(pred, dim=0)==y)