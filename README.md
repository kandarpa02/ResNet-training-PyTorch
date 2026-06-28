# ResNet Training Core (PyTorch)

A lightweight, production-oriented PyTorch training framework
demonstrating **correct, reproducible, and resume-safe deep learning
training** using **ResNet-18 on CIFAR-10**.

The project emphasizes **explicit training logic**, **performance**, and
**maintainability** over framework abstractions, making it suitable for
research prototypes, internal tooling, and small-scale production
workloads.

```{=html}
<p align="center">
```
`<img src="media/output.png" width="900">`{=html}
```{=html}
</p>
```

------------------------------------------------------------------------

## Highlights

-   Production-style training loop written in plain PyTorch
-   Automatic Mixed Precision (AMP) with `torch.amp.autocast`
-   Resume-safe checkpointing
-   Modular trainer factory
-   Explicit validation pipeline
-   Minimal codebase with no hidden framework logic
-   Easily extensible to schedulers, DDP, or custom metrics

------------------------------------------------------------------------

# Training Results

**Model:** ResNet-18

**Dataset:** CIFAR-10

  Metric                                     Value
  --------------------- --------------------------
  Epochs                                       200
  Train Accuracy                        **98.05%**
  Validation Accuracy                   **79.84%**
  Train Loss                            **0.0608**
  Validation Loss                       **0.8029**
  Average Epoch Time      **≈22.7 s (Tesla P100)**

The included plots visualize:

-   Training vs validation accuracy
-   Training vs validation loss
-   Epoch execution time
-   Generalization gap throughout training

------------------------------------------------------------------------

# Features

### Trainer Factory

Creates configurable training functions while keeping the training loop
explicit and easy to debug.

### Resume-safe Checkpointing

Every checkpoint stores:

-   Model parameters
-   Optimizer state
-   AMP GradScaler state
-   Current epoch

allowing training to resume without losing optimizer momentum or
mixed-precision state.

### Automatic Mixed Precision (AMP)

Native PyTorch AMP support reduces GPU memory usage and improves
throughput with minimal code changes.

### Explicit Training Pipeline

No callbacks.

No hidden state.

No training framework magic.

Every optimization step is visible and easy to modify.

------------------------------------------------------------------------

# Installation

``` bash
git clone https://github.com/<username>/<repo>.git
cd <repo>
pip install -r requirements.txt
```

------------------------------------------------------------------------

# Usage

## Imports

``` python
from resnet_training import (
    ResNet18,
    trainer,
    cifar10,
    Checkpoint,
    init,
)

from torch.utils.data import DataLoader
import torch
```

## Dataset

``` python
train_ds = cifar10(split=(80, 20), part=0)
val_ds   = cifar10(split=(80, 20), part=1)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=2)
```

## Model

``` python
model = ResNet18(10)
init(model, [1, 3, 32, 32])
```

## Trainer

``` python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

optimizer = torch.optim.Adam(model.parameters())
loss_fn = torch.nn.CrossEntropyLoss()

ckpt = Checkpoint(
    last_epoch=None,
    param_state=None,
    opt_state=None,
    scalar_state=None,
)

train_fn = trainer(
    epochs=200,
    ckpt=ckpt,
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    device=device,
    ckpt_path="./checkpoints",
    auto_cast=True,
)
```

## Train

``` python
train_fn(train_loader, val_loader)
```

Example output:

``` text
Epoch 199/200
Train Loss : 0.0605
Val Loss   : 0.8024
Train Acc  : 98.09%
Val Acc    : 79.84%
```

## Resume Training

``` python
model = ResNet18(10)

checkpoint = torch.load(
    "./checkpoints/ckpt200",
    map_location="cpu",
)

model.load_state_dict(checkpoint["param_state"])
```

The checkpoint also contains the optimizer state, AMP scaler state, and
last completed epoch, enabling seamless training resumption.

------------------------------------------------------------------------

# Design Philosophy

-   Explicit over implicit
-   Correctness before abstraction
-   Resume safety as a core feature
-   Performance through native PyTorch
-   Easy to extend without rewriting the training loop

------------------------------------------------------------------------

# Project Scope

## Included

-   ResNet-18 training
-   CIFAR-10 pipeline
-   Mixed precision training
-   Checkpoint management
-   Modular trainer
-   Validation metrics

## Not Included

-   Distributed Data Parallel (DDP)
-   Multi-node training
-   Callback systems
-   Configuration frameworks
-   Hyperparameter orchestration

These features can be added incrementally without changing the overall
project architecture.