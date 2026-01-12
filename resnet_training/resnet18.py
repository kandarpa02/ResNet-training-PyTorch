from .base import Cell
import torch.nn as nn
import torch.nn.functional as F
import torch

class BasicBlock(Cell):
  def __init__(self, planes: int, stride: int = 1):
    super().__init__()

    self.conv1 = nn.LazyConv2d(
      planes, kernel_size=3, stride=stride, padding=1, bias=False
    )
    self.bn1 = nn.LazyBatchNorm2d()

    self.conv2 = nn.LazyConv2d(
      planes, kernel_size=3, stride=1, padding=1, bias=False
    )
    self.bn2 = nn.LazyBatchNorm2d()

    self.relu = nn.ReLU(inplace=True)

    self.downsample = None
    if stride != 1:
      self.downsample = nn.Sequential(
        nn.LazyConv2d(
          planes, kernel_size=1, stride=stride, bias=False
        ),
        nn.LazyBatchNorm2d(),
      )

  def forward(self, x):
    identity = x

    out = self.relu(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))

    if self.downsample is not None:
      identity = self.downsample(x)

    out += identity
    return self.relu(out)


class ResNet18(Cell):
  def __init__(self, num_classes):
    super().__init__()

    # Stem
    self.conv1 = nn.LazyConv2d(
      64, kernel_size=7, stride=2, padding=3, bias=False
    )
    self.bn1 = nn.LazyBatchNorm2d()
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(
      kernel_size=3, stride=2, padding=1
    )

    # Res layers
    self.layer1 = self._make_layer(64, blocks=2, stride=1)
    self.layer2 = self._make_layer(128, blocks=2, stride=2)
    self.layer3 = self._make_layer(256, blocks=2, stride=2)
    self.layer4 = self._make_layer(512, blocks=2, stride=2)

    # Head
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.LazyLinear(num_classes)

  def _make_layer(self, planes: int, blocks: int, stride: int):
    layers = []

    # First block may downsample
    layers.append(BasicBlock(planes, stride=stride))

    # Remaining blocks keep same shape
    for _ in range(1, blocks):
      layers.append(BasicBlock(planes, stride=1))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.relu(self.bn1(self.conv1(x)))
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.fc(x)

    return x

def init(model, shape):
  _ = torch.ones(shape)
  model.eval()
  model(_)
