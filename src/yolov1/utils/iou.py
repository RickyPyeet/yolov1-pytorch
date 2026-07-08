import torch

def get_iou(corners_one, corners_target):
  # Get intersection corners
  xmin = torch.max(corners_one[..., 0], corners_target[..., 0])
  ymin = torch.max(corners_one[..., 1], corners_target[..., 1])
  xmax = torch.min(corners_one[..., 2], corners_target[..., 2])
  ymax = torch.min(corners_one[..., 3], corners_target[..., 3])

  # Get bbox areas to caculate union
  area_bbox_one = abs((corners_one[..., 2] - corners_one[..., 0]) * (corners_one[..., 3] - corners_one[..., 1]))
  area_bbox_two = abs((corners_target[..., 2] - corners_target[..., 0]) * (corners_target[..., 3] - corners_target[..., 1]))

  # Calculate intersection and union
  intersection = torch.clamp(xmax - xmin, min = 0) * torch.clamp(ymax - ymin, min = 0) # Clamps values to min = 0 so that it doesn't go negative

  union = area_bbox_one + area_bbox_two - intersection + 1e-9 # Sum a very small number to avoid dividing by 0 -> this will allow iou = 0 if bboxes don't overlap

  return intersection / union
