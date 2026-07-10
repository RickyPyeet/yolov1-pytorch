import torch

from typing import List

from src.yolov1.utils.boxes import bbox_centers_to_corners

def decode_gt(batched_gt, S: int = 7) -> List[List]:
  """
  Here all bboxes are to keep since they are all good bboxes
  args:
    batched_gt (torch.Tensor): tensor of batches extracted from dataloader with shape (batch, S, S, 30)
    S (int): grid size, used to split second bbox that contains 0s
  output:
    target_decoded (list): contains a list of lists. Each list will have the img corresponding bboxes with format (batch, S*S, 6) -> [xmin, ymin, xmax, ymax, conf, class]
  """
  bboxes = bbox_centers_to_corners(batched_gt[...,:4], S)
  target_classes = batched_gt[..., 10:]
  target_conf = batched_gt[..., 4].unsqueeze(-1)

  _, max_class_idx = torch.max(target_classes, dim = -1)

  target_decoded = torch.cat([bboxes, target_conf, max_class_idx.unsqueeze(-1)], dim = -1)

  target_decoded = target_decoded.reshape(target_decoded.shape[0], S*S, target_decoded.shape[3])

  return target_decoded

def mask_gt(target_decoded):
  """
  After decoding a batched gt, loops through the pictures and keeps only the bboxes that have a conf == 1.
  It returns a List of lists and each list will correspond to the gt bboxes for each image.
  """
  batches = target_decoded.shape[0]
  mask = target_decoded[..., 4] == 1 # Mask only bboxes with conf == 1 -> correct boxes
  keep_bboxes = []

  for b in range(batches):
    masked_gt = target_decoded[b][mask[b]]
    keep_bboxes.append(masked_gt.tolist())

  return keep_bboxes
