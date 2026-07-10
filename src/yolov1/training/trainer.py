import torch
from tqdm.auto import tqdm
from typing import Dict

from src.yolov1.evaluation.map import compute_map
from src.yolov1.evaluation.targets import decopde_gt, mask_gt
from src.yolov1.utils.checkpoint import save_checkpoint
from src.yolov1.utils.nms import compute_nms

# TRAINING STEP
def training_step(model: torch.nn.Module,
                  train_dataloader: torch.utils.data.DataLoader,
                  optimizer: torch.optim.Optimizer,
                  loss_fn: torch.nn.Module,
                  max_clip_norm: float = 10,
                  device: torch.device = "cpu"):
  model.train()

  train_loss = 0

  for batch, (X, y) in enumerate(train_dataloader):
    X, y = X.to(device), y.to(device)
    y_pred = model(X)
    loss = loss_fn(y_pred, y)
    train_loss += loss.item()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_clip_norm)
    optimizer.step()

    if batch % 50 == 0:
      print(f"Looked at: {batch*len(X)}/{len(train_dataloader.dataset)} samples")

  train_loss /= len(train_dataloader)

  return train_loss

# VALIDATION STEP
def validation_step(model: torch.nn.Module,
                    val_dataloader: torch.utils.data.DataLoader,
                    loss_fn: torch.nn.Module,
                    nms_conf_threshold: float = 0.01,
                    device: torch.device = "cpu"):
  model.eval()

  val_loss = 0
  map_score = 0

  pred_idx = 0
  target_idx = 0

  preds_list = []
  targets_list= []

  with torch.inference_mode():
    for batch, (X, y) in enumerate(val_dataloader):
      X, y = X.to(device), y.to(device)
      y_pred = model(X)
      loss = loss_fn(y_pred, y)
      val_loss += loss.item()

      # Compute nms
      nms_list = compute_nms(pred_batches = y_pred,
                             conf_threshold = nms_conf_threshold,
                             iou_threshold = 0.5)

      # Add img idx to nms_list and target
      for preds in nms_list:
        for box in preds:
          preds_list.append([pred_idx]+box)
        pred_idx += 1

      # Add image idx to decoded target
      decoded_targets = decode_gt(y.detach().to("cpu"))
      masked_targets = mask_gt(decoded_targets)

      for targets in masked_targets:
        for box in targets:
          targets_list.append([target_idx]+box)
        target_idx += 1

    # Compute map
    print(f"Starting mAP... :O")
    map_score = compute_map(preds_list, targets_list, iou_threshold = 0.5, num_classes = 20)

    val_loss /= len(val_dataloader)

  return val_loss, map_score

# COMPLETE TRAINING FUNCTION
def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          val_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          nms_conf_threshold: float = 0.01,
          scheduler = None,
          checkpoint_path: str | None = None,
          model_name: str = 'yolov1.pt',
          max_clip_norm: float = 10,
          device: torch.device | str = 'cpu',
          logger = None) -> Dict:

  # Results dict
  results = {'train_loss_history': [],
             'val_loss_history': [],
             'map_history': [],
             'epochs': []}
  # Move to device
  model.to(device)

  for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch} | {epochs}\n-------------")

    # Training step
    train_loss = training_step(model = model,
                                train_dataloader = train_dataloader,
                                optimizer = optimizer,
                                loss_fn = loss_fn,
                                device = device,
                                max_clip_norm = max_clip_norm)

    # Validation step with map
    print("Starting inference mode!!! :)")

    val_loss, map_score = validation_step(model = model,
                                          val_dataloader = val_dataloader,
                                          loss_fn = loss_fn,
                                          nms_conf_threshold = nms_conf_threshold,
                                          device = device)
    # Stepping the scheduler
    if scheduler is not None:
     scheduler.step()

    # Logging results
    if logger is not None:
      metrics = {'train_loss': train_loss,
                  'val_loss': val_loss,
                  'map_score': map_score,
                  'epoch': epoch}
      logger(metrics)


    # Appending results
    results["map_history"].append(map_score)
    results["train_loss_history"].append(train_loss)
    results['val_loss_history'].append(val_loss)
    results["epochs"].append(epoch)

    # Printing out results
    print(f"Epoch: {epoch} | Train loss: {train_loss} | Val_loss: {val_loss} | mAP: {map_score}\n")

    if (epoch+1) == epochs and (checkpoint_path is not None):
      save_checkpoint(checkpoint_path = checkpoint_path,
                      model = model,
                      model_name = model_name,
                      optimizer = optimizer,
                      results_dict = results,
                      scheduler = scheduler)

  return results
