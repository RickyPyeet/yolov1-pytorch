import torch
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import SequentialLR, LambdaLR, CosineAnnealingLR

def setup_scheduler(optimizer,
                    epochs,
                    min_lr = 0,
                    warmup_epochs = 0):
  if warmup_epochs > 0:
    def warmup_lambda(epoch):
      return (1+epoch)/warmup_epochs

    warmup_scheduler = LambdaLR(optimizer, lr_lambda = warmup_lambda)

    cosine_scheduler = CosineAnnealingLR(optimizer, eta_min = min_lr, T_max = epochs - warmup_epochs)

    scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones = [warmup_epochs])

  else:
    scheduler = CosineAnnealingLR(optimizer, T_max = epochs, eta_min = min_lr)

  return scheduler