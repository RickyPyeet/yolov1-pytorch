import datetime

from pathlib import Path

def save_checkpoint(checkpoint_path: str,
                    model: nn.Module,
                    model_name:str,
                    optimizer: torch.optim.Optimizer,
                    results_dict: Dict,
                    scheduler: torch.optim.lr_scheduler.LRScheduler = None) -> None:

  checkpoint_path = Path(checkpoint_path)

  if not checkpoint_path.exists():
    checkpoint_path.mkdir(parents = True, exist_ok = True)

  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with .pth or .pt (.pt is preferred)"

  date = datetime.datetime.now()
  date = "_".join(date.strftime("%c").strip().split())

  model_save_path = checkpoint_path / f"{date}_{model_name}"

  checkpoint = {'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'results_dict': results_dict}

  if scheduler:
    checkpoint['lr_scheduler_state_dict'] = scheduler.state_dict()

  print(f"[INFO] Saving {model_save_path}...")

  torch.save(obj = checkpoint, f = model_save_path)
  
  return checkpoint
