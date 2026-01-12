import torch

class Metric:
  def __init__(self):
    self.value = 0.0
    self.count = 0

  def update(self, value, n=1):
    self.value += value
    self.count += n

  def mean(self):
    return self.value / max(self.count, 1)

def accuracy(pred, y):
  return (pred.argmax(dim=1) == y).float().mean()
