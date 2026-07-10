import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from src.yolov1.data.transforms import get_voc_train_transforms, get_voc_val_transforms
from src.yolov1.data.voc import create_dataloaders
from src.yolov1.models.yolov1 import create_yolo_detection
from src.yolov1.training.loss import YOLOLoss
from src.yolov1.training.trainer import train
from src.yolov1.utils.scheduler import setup_scheduler
from src.yolov1.utils.seed import set_seed
from src.yolov1.utils.config import load_config

def create_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
  """
  Instantiate an optimizer based on the name passed [sgd, adam, adamw]
  """

  optimizer_name = config["name"]
  lr = config["lr"]
  weight_decay = config.get("weight_decay", 0.0)

  if optimizer_name == "sgd":
      return torch.optim.SGD(
          model.parameters(),
          lr=lr,
          momentum=config.get("momentum", 0.0),
          weight_decay=weight_decay)

  if optimizer_name == "adam":
      return torch.optim.Adam(
          model.parameters(),
          lr=lr,
          weight_decay=weight_decay)

  if optimizer_name == "adamw":
      return torch.optim.AdamW(
          model.parameters(),
          lr=lr,
          weight_decay=weight_decay)

  raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def main(config_path: str | Path) -> None:
  config = load_config(config_path)

  seed = config['seed'] 
  set_seed(seed)

  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  # Instantiate model with backbone weights
  model = create_yolo_detection(backbone_weights_path = config['model']['backbone_weights_path'], 
                                seed = seed).to(device)
  
  # Compile model if required
  if config['model'].get('compile', False):
    torch.set_float32_matmul_precision('high')
    model.compile()

  # Instantiate Yolo loss
  loss_fn = YOLOLoss(S = config['model']['S'],
                     B = config['model']['B'],
                     C = config['model']['C'],
                     L_coord = config['loss']['L_coord'],
                     L_noobj = config['loss']['L_noobj']).to(device)
  
  # Instantiate optimizer
  optimizer = create_optimizer(model = model, config = config['optimizer'])

  # Create Train and Val Dataloader
  train_dataloader, val_dataloader = create_dataloaders(training_csv = config['data']['train_csv'],
                                                        val_csv = config['data']['val_csv'],
                                                        voc_train_transforms = get_voc_train_transforms(),
                                                        voc_val_transforms = get_voc_val_transforms(),
                                                        batch_size = config['data']['batch_size'],
                                                        num_workers = config['data'].get('num_workers', os.cpu_count() or 1))
  # Scheduler
  scheduler = None
  if config['scheduler']['enabled']:
    scheduler = setup_scheduler(optimizer = optimizer,
                                epochs = config['training']['epochs'],
                                min_lr = config['scheduler']['min_lr'],
                                warmup_epochs = config['scheduler']['warmup_epochs'])
    
  train(model = model,
        train_dataloader = train_dataloader,
        val_dataloader = val_dataloader,
        optimizer = optimizer,
        loss_fn = loss_fn,
        epochs = config['training']['epochs'],
        nms_conf_threshold = config['training']['nms_conf_threshold'],
        scheduler = scheduler,
        checkpoint_path = config['checkpoint']['path'],
        model_name = config['checkpoint']['model_name'],
        max_clip_norm = config['training']['max_clip_norm'],
        device = device)
  
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description = 'Train YOLOv1 on Pascal VOC')

  parser.add_argument('--config', type = str, default = 'configs/voc.yaml', help = 'Path to the training YAML configuration')

  args = parser.parse_args()

  main(config_path = args.config)
