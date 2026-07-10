import torch

from typing import List

from src.yolov1.utils.iou import get_iou

def compute_map(pred_boxes: List, target_boxes: List, iou_threshold: float = 0.5, num_classes: int = 20) -> float:
  """
  Compute the mAP for the whole dataset.
  args:
    pred_boxes (list): list of predicted boxes after nms. Each prediction is structured this way [img_idx, xmin, ymin, xmax, ymax, confidence, class]
    target_boxes (list): list of target boxes after being decoded for mAP, each box is structured this way [img_dix, xmin, ymin, xmax, ymax, conf, class]
    iou_threshold (float - 0.5): threshold for iou between predictions and ground truths (gt)
    num_classes (int - 20): number of classes to loop through. Default 20 for PascalVOC
  output:
    map (float): mean average precision
  """

  average_precision = []

  for c in range(num_classes):
    detections = []
    gts = []

    # Add predictions and targets that correspond to class "c" to a list
    for detection in pred_boxes:
      if detection[6] == c:
        detections.append(detection)

    for target in target_boxes:
      if target[6] == c:
        gts.append(target)

    # Count amount of boxes for each example of class c -> {0: 2, 1: 4} -> img0 has 2 bbox, img1 has 4 bbox
    boxes_amount = {}
    for g in gts:
      if g[0] not in boxes_amount.keys():
        boxes_amount[g[0]] = 1
      else:
        boxes_amount[g[0]] += 1
    # Turn dictionary into dictionary with 0s for each # of boxes
    for key, value in boxes_amount.items():
      boxes_amount[key] = torch.zeros(value) # {0: tensor[0., 0.], 1: tensor[0., 0., 0., 0.]}

    # Sort detections by confidence scores
    detections.sort(key = lambda x: x[5], reverse = True) # Sort box by more confident to less confident

    # Create TP and FP tensors
    TP = torch.zeros(len(detections))
    FP = torch.zeros(len(detections))
    total_true_boxes = len(gts)

    if total_true_boxes == 0:
      continue # Skip class if there are no boxes in ground truth

    # Loop through the detections and count how many are TP or FP
    for detect_idx, detection in enumerate(detections):
      # Only consider gts that belong to the same image and have the same class c
      comparing_gt = [gt for gt in gts if gt[0] == detection[0]]

      best_iou = 0
      best_gt_idx = -1

      for idx, gt in enumerate(comparing_gt):
        # Compute iou between detection and ground truth
        iou = get_iou(torch.tensor(detection[1:5], dtype = torch.float32), torch.tensor(gt[1:5], dtype = torch.float32))

        # If that box has the best iou, keep track of it and the ground truth idx
        if iou > best_iou:
          best_iou = iou
          best_gt_idx = idx

      # If best_iou is greater than the threshold
      if best_iou > iou_threshold:
        if boxes_amount[detection[0]][best_gt_idx] == 0:
          TP[detect_idx] = 1
          boxes_amount[detection[0]][best_gt_idx] = 1
        else:
          FP[detect_idx] = 1
      else:
        FP[detect_idx] = 1

    TP_cumsum = torch.cumsum(TP, dim = 0)
    FP_cumsum = torch.cumsum(FP, dim = 0)
    recall = TP_cumsum / (total_true_boxes + 1e-6)
    precision = TP_cumsum / (TP_cumsum + FP_cumsum + 1e-6)
    precision = torch.cat((torch.tensor([1]), precision))
    recall = torch.cat((torch.tensor([0]), recall))

    average_precision.append(torch.trapezoid(y = precision, x = recall, dim = -1))

  if len(average_precision) == 0:
    return 0.0

  map_result = sum(average_precision) / len(average_precision)

  return map_result.item()
