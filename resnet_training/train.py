import torch
import time
import os

from .checkpoint import Checkpoint
from .base import Cell
from .metric import Mean, Accuracy
from typing import Callable

from torch import autocast
from torch import GradScaler

Opt = torch.optim.Optimizer

def trainer(
    epochs:int,
    *,
    ckpt:Checkpoint, 
    model:Cell, 
    optimizer:Opt,
    loss_fn:Callable,
    device:torch.device,
    ckpt_path:str,
    save_every_nth:int=1,
    auto_cast:bool=False
    
    ) -> Callable:

  """
  Trainer factory function with configurable training behavior.

  Arguments:
    epochs (int): Number of epochs to train the model.
    ckpt (Checkpoint): Checkpoint object containing last_epoch, param_state,
      opt_state, and scalar_state for resume-safe training.
    model (torch.nn.Module): Model instance to be trained.
    optimizer (torch.optim.Optimizer): Optimizer instance.
    loss_fn (Callable): Loss function (e.g., torch.nn.CrossEntropyLoss).
    device (torch.device): Target device, e.g., torch.device("cpu") or
      torch.device("cuda").
    ckpt_path (str): Directory path where checkpoints will be saved.
    save_every_nth (int): Frequency (in epochs) for saving checkpoints.
    auto_cast (bool): Enable automatic mixed precision using
      torch.amp.autocast and GradScaler.

  Returns:
    Callable: A training function that accepts (train_loader, val_loader).

  Example:
    ```python
    from resnet_training import ResNet18, trainer, cifar10, Checkpoint, init
    from torch.utils.data import DataLoader
    import torch

    # dataset & loaders
    train_loader = ...
    val_loader = ...

    model = ResNet18(10)
    init(model, [1, 3, 32, 32])

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    optimizer = torch.optim.SGD(model.parameters(), 0.05, momentum=0.9, nesterov=True)
    loss_fn = torch.nn.CrossEntropyLoss()

    ckpt = Checkpoint(
        last_epoch=None,
        param_state=None,
        opt_state=None,
        scalar_state=None,
    )

    train_fn = trainer(
        epochs=5,
        ckpt=ckpt,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        ckpt_path="./checkpoints",
        auto_cast=True,
    )

    train_fn(train_loader, val_loader)

    # Epoch 1/5 Train_Loss:1.98, Val_Loss:1.87 Train_Acc:0.31, Val_Acc:0.33 Took:13.66s
  """
  
  last_epoch = ckpt.last_epoch
  opt_state = ckpt.opt_state
  params_state = ckpt.param_state
  scalar_state = ckpt.scalar_state

  if params_state:
    model.load_state_dict(params_state)
  model = model.to(device)
  
  if opt_state:
    optimizer.load_state_dict(opt_state)

  if auto_cast:
    scalar = GradScaler(device.type)
    if scalar_state:
      scalar.load_state_dict(scalar_state)

  START_EPOCHS = last_epoch if last_epoch is not None else 0
  EPOCHS = epochs + START_EPOCHS

  os.makedirs(ckpt_path, exist_ok=True)

  def fit(
      train_loader, val_loader
      ):
    
    for e in range(START_EPOCHS, EPOCHS):
      t0 = time.time()
      t_loss = Mean(device=device)
      v_loss = Mean(device=device)
      t_acc  = Accuracy(device=device)
      v_acc  = Accuracy(device=device)

      def batched_train_step(x, y, auto_cast):
        model.train()
        if auto_cast:
          with autocast(device.type):
            pred = model(x)
            loss = loss_fn(pred, y)

        else:
          pred = model(x)
          loss = loss_fn(pred, y)
        return loss, pred
      
      def batched_val_step(x, y, auto_cast):
        model.eval()
        with torch.no_grad():
          if auto_cast:
            with autocast(device.type):
              pred = model(x)
              loss = loss_fn(pred, y)

          else:
            pred = model(x)
            loss = loss_fn(pred, y)
          return loss, pred
      
      for x, y in train_loader:
          x = x.to(device)
          y = y.to(device)
          optimizer.zero_grad(set_to_none=True)

          train_loss, train_pred = batched_train_step(x, y, auto_cast=auto_cast)

          t_loss.update(train_loss.item())
          t_acc.update(train_pred, y)

          if auto_cast:
            scalar.scale(train_loss).backward()
            scalar.step(optimizer)
            scalar.update()
          else:
            train_loss.backward()
            optimizer.step()

      for x, y in val_loader:
          x = x.to(device)
          y = y.to(device)
          val_loss, val_pred = batched_val_step(x, y, auto_cast=auto_cast)

          v_loss.update(val_loss.item())
          v_acc.update(val_pred, y)


      t1 = time.time()

      _time = t1-t0
      tr_loss = t_loss.compute().item()
      tr_acc = t_acc.compute().item()
      val_loss = v_loss.compute().item()
      val_acc = v_acc.compute().item()

      print(
        f"Epoch {e+1}/{EPOCHS} "
        f"Train_Loss:{tr_loss:.4f}, Val_Loss:{val_loss:.4f} "
        f"Train_Acc:{tr_acc:.4f}, Val_Acc:{val_acc:.4f} "
        f"Took:{_time:.2f} "
      )

      if (e+1)%save_every_nth!=1:
        ckpt.update(
          last_epoch = e+1,
          params_state = model.state_dict(),
          opt_state = optimizer.state_dict(),
          scalar_state = scalar.state_dict() if auto_cast else None,
        )
        torch.save(ckpt.state_dict(), f"{ckpt_path}/final_ckpt")
        torch.save(ckpt.state_dict(), f"{ckpt_path}/ckpt{e+1}")

  return fit