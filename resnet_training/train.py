import torch
import time

from .checkpoint import Checkpoint
from .base import Module_
from .metric import Metric, accuracy
from typing import Callable

from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

Opt = torch.optim.Optimizer

def trainer(
    epochs:int,
    *,
    ckpt:Checkpoint, 
    model:Module_, 
    optimizer:Opt,
    loss_fn:Callable,
    device:torch.device,
    ckpt_path:str,
    save_every_nth:int=1,
    auto_cast:bool=False
    
    ):
  
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

  def innerfun(
      train_loader, val_loader
      ):
    
    for e in range(START_EPOCHS, EPOCHS):
      t0 = time.time()
      t_loss = Metric()
      v_loss = Metric()
      t_acc  = Metric()
      v_acc  = Metric()

      def batched_train_step(x, y, auto_cast):
        model.train()
        if auto_cast:
          with autocast(device.type):
            pred = model(x)
            acc = accuracy(pred, y)
            loss = loss_fn(pred, y)

        else:
          pred = model(x)
          acc = accuracy(pred, y)
          loss = loss_fn(pred, y)
        return loss, acc
      
      def batched_val_step(x, y, auto_cast):
        model.eval()
        if auto_cast:
          with autocast(device.type):
            pred = model(x)
            acc = accuracy(pred, y)
            loss = loss_fn(pred, y)

        else:
          pred = model(x)
          acc = accuracy(pred, y)
          loss = loss_fn(pred, y)
        return loss, acc
      
      for x, y in train_loader:
          x = x.to(device)
          y = y.to(device)
          optimizer.zero_grad()

          train_loss, train_acc = batched_train_step(x, y, auto_cast=auto_cast)
          t_loss.update(train_loss.item())
          t_acc.update(train_acc.item())

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
          val_loss, val_acc = batched_val_step(x, y, auto_cast=auto_cast)
          v_loss.update(val_loss.item())
          v_acc.update(val_acc.item())


      t1 = time.time()

      _time = t1-t0
      tr_loss = t_loss.mean()
      tr_acc = t_acc.mean()
      val_loss = v_loss.mean()
      val_acc = v_acc.mean()

      print(
        f"Epoch {e+1}/{EPOCHS+1} "
        f"Train_Loss:{tr_loss:.4f}, Val_Loss:{val_loss:.4f}"
        f"Train_Acc:{tr_acc:.4f}, Val_Acc:{val_acc:.4f}"
      )

      if e%save_every_nth!=1:
        ckpt.update(
          last_epoch = e,
          params_state = model.state_dict(),
          opt_state = optimizer.state_dict(),
          scalar_state = scalar.state_dict() if auto_cast else None,
        )

    torch.save(ckpt, ckpt_path)

  return innerfun