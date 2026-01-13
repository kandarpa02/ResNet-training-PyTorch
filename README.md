# ResNet-training-PyTorch

This repository contains a PyTorch training pipleline of ResNet18 on CIFAR-10. It a custom `Checkpoint` object to store checkpoints efficiently

## Usage:

### Imports:
```python
from resnet_training import ResNet18, trainer, cifar10, Checkpoint, init
from torch.utils.data import DataLoader
import torch
# the model is defined with PyTorch's lazy layers for error-free coding,
# because of that 'init' function is there

```

### Data:
```python 
train_ds = cifar10(split=(80,20), part=0)
val_ds = cifar10(split=(80,20), part=1)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=2)

```

### Initializing the factory function:
```python
model = ResNet18(10)
init(model, [1, 3, 32, 32])
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

opt = torch.optim.Adam(model.parameters())

loss_fn = torch.nn.CrossEntropyLoss()

ckpt = Checkpoint(
    None,
    param_state=None,
    opt_state=None,
    scalar_state=None
)
train_func = trainer(50, ckpt=ckpt, model=model, optimizer=opt, loss_fn=loss_fn, device=device, ckpt_path="/content/checkpoint", auto_cast=True)
```

### Train:
```python
train_func(train_loader, val_loader)

# Epoch 1/50 Train_Loss:1.7193, Val_Loss:1.5661 Train_Acc:0.3772, Val_Acc:0.4317 Took:7.92 
# Epoch 2/50 Train_Loss:1.4636, Val_Loss:1.5017 Train_Acc:0.4683, Val_Acc:0.4517 Took:7.71 
# Epoch 3/50 Train_Loss:1.3390, Val_Loss:1.4681 Train_Acc:0.5159, Val_Acc:0.4612 Took:7.89 
# Epoch 4/50 Train_Loss:1.2582, Val_Loss:1.4182 Train_Acc:0.5425, Val_Acc:0.5083 Took:7.43 
# Epoch 5/50 Train_Loss:1.1814, Val_Loss:1.3995 Train_Acc:0.5754, Val_Acc:0.5082 Took:7.38 
# Epoch 6/50 Train_Loss:1.1208, Val_Loss:1.3720 Train_Acc:0.5960, Val_Acc:0.5215 Took:7.83 
# Epoch 7/50 Train_Loss:1.0621, Val_Loss:1.2044 Train_Acc:0.6218, Val_Acc:0.5702 Took:7.36 
# Epoch 8/50 Train_Loss:0.9988, Val_Loss:1.5843 Train_Acc:0.6406, Val_Acc:0.4697 Took:7.46 
# Epoch 9/50 Train_Loss:0.9301, Val_Loss:1.4589 Train_Acc:0.6657, Val_Acc:0.5219 Took:7.72 
```
### Load Checkpoint:
```python
model = ResNet18(10)
c50 = torch.load("/home/kandarpa-sarkar/Downloads/ckpt50", weights_only=False, map_location=torch.device('cpu'))
model.load_state_dict(c50['param_state'])
```
See that we have flexible checkpoint management for training resumption and the whole system is greate for single GPU training pipeline