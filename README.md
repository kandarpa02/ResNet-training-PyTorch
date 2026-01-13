# ResNet Training (PyTorch)

A **minimal, production-oriented PyTorch training core** focused on **correctness, resume-safe checkpointing, and performance**, demonstrated using **ResNet-18 on CIFAR-10**.

This project is designed for **small research labs and early-stage startups** that want:
- full control over the training loop
- explicit, debuggable code
- AMP performance gains
- reliable training resumption

It intentionally avoids high-level abstractions and distributed complexity.

---

## Key Features

- **Trainer factory pattern** for clean, configurable training
- **Resume-safe checkpointing** (model, optimizer, AMP scaler, epoch)
- **Automatic Mixed Precision (AMP)** via `torch.amp.autocast` + `GradScaler`
- Explicit training & validation loops (no hidden magic)
- Designed for **single-GPU production workflows**

---

## Usage

### Imports
```python
from resnet_training import ResNet18, trainer, cifar10, Checkpoint, init
from torch.utils.data import DataLoader
import torch

# The model uses PyTorch lazy layers.
# `init` performs a shape-aware initialization to avoid silent shape errors.
```


### Data and DataLoader
```python 
train_ds = cifar10(split=(80, 20), part=0)
val_ds   = cifar10(split=(80, 20), part=1)

train_loader = DataLoader(
    train_ds, batch_size=128, shuffle=True, num_workers=2
)
val_loader = DataLoader(
    val_ds, batch_size=512, shuffle=False, num_workers=2)
```

### Trainer Initialization:
```python
model = ResNet18(10)
init(model, [1, 3, 32, 32])

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

optimizer = torch.optim.Adam(model.parameters())
loss_fn = torch.nn.CrossEntropyLoss()

ckpt = Checkpoint(
    last_epoch=None,
    param_state=None,
    opt_state=None,
    scalar_state=None,
)

train_fn = trainer(
    epochs=50,
    ckpt=ckpt,
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    device=device,
    ckpt_path="./checkpoints",
    auto_cast=True,  # Enable AMP
)

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
### Loading a Checkpoint (Resume Training):
```python
model = ResNet18(10)

ckpt_state = torch.load(
    "./checkpoints/ckpt50",
    map_location=torch.device("cpu"),
)

model.load_state_dict(ckpt_state["param_state"])
```

---

*The checkpoint system supports clean training resumption, including optimizer and AMP scaler state when enabled.*

**Design Philosophy**

- Explicit over implicit: no hidden training behavior
- Resume safety first: checkpoints capture full training state
- Performance-aware: AMP is a first-class feature
- Single-GPU focused: optimized for simplicity and reliability

**Non-Goals**

- Distributed multi-node training
- High-level training abstractions (Lightning, callbacks, YAML configs)
- Automatic hyperparameter orchestration
- The training loop is intentionally written in plain PyTorch and can be extended (e.g., with DDP or schedulers) if required.