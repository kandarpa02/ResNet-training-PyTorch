import torch

# class Metric:
#   def __init__(self):
#     self.value = 0.0
#     self.count = 0

#   def update(self, value, n=1):
#     self.value += value
#     self.count += n

#   def mean(self):
#     return self.value / max(self.count, 1)

# def accuracy(pred, y):
#   return (pred.argmax(dim=1) == y).float().mean()


class Metric(object):
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
