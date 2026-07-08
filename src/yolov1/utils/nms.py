import torch

from typing import List

from src.yolov1.utils.boxes import bbox_centers_to_corners
from src.yolov1.utils.iou import get_iou

def compute_corners_and_concat(pred_tensor):
  """
  From one tensor, split it into two bboxes per section, compute corners, extract classes and boxes conf scores and concatenate together
  to get an output of shape (batch, S*S*B, [xmin,ymin, xmax, ymax, confidence, class])
  """
  # Extract bbox data
  bbox_one = pred_tensor[..., 0:4]
  bbox_two = pred_tensor[..., 5:9]

  batch_size = pred_tensor.shape[0]
  S1, S2 = pred_tensor.shape[1], pred_tensor.shape[2]

  bbox_one_conf = pred_tensor[..., 4].unsqueeze(-1).reshape(pred_tensor.shape[0], S1*S2, 1) # reshaping everything to be (batch, S*S, conf)
  bbox_two_conf = pred_tensor[..., 9].unsqueeze(-1).reshape(pred_tensor.shape[0], S1*S2, 1)

  box_class = pred_tensor[..., 10:30].reshape(pred_tensor.shape[0], S1*S2, 20)

  # Extract xmin, ymin, xmax, ymax
  corners_one = bbox_centers_to_corners(bbox_one, S = 7).reshape(pred_tensor.shape[0], S1*S2, 4) # reshape to be (batch, S*S, [xmin, ymin, xmax, ymax])
  corners_two = bbox_centers_to_corners(bbox_two, S = 7).reshape(pred_tensor.shape[0], S1*S2, 4)

  # Stack everything together
  bboxes_one = torch.cat([corners_one, bbox_one_conf, box_class], dim = 2) # Add data to last dimension so we have tensor [xmin, ymin, xmax, ymax, conf, classes]
  bboxes_two = torch.cat([corners_two, bbox_two_conf, box_class], dim = 2)
  tensors_concat = torch.cat((bboxes_one, bboxes_two), dim = 1) # Add data to area (S*S) dimension so that we can compare them.

  return tensors_concat


def compute_nms(pred_batches, conf_threshold: float = 0.25, iou_threshold: float = 0.5) -> List[List]:
  """
  This version computes the nms but instead of returning a tensor, it returns a list of lists, each list will contain
  bboxes with the following bbox coordinates [xmin, ymin, xmax, ymax, box_conf, class_idx]
  - box conf = conf_score * class_prob

  args:
    pred_batches (torch.Tensor): a tensor of batched predictions with shape (batches, S, S, 30)
    conf_threshold (float): first conf threshold to quickly remove bboxes with low confidence
    iou_threshold (float): threshold to remove boxes that overlap with the one with higher confidence

  output:
    keep_list (list): a list of lists containing good bboxes for each image [[xmin, ymin, xmax, ymax, conf_score, class_idx], ...]
  """
  num_batches = pred_batches.shape[0]

  pred_batches = pred_batches.reshape(num_batches, 7, 7, 30)

  bbox = compute_corners_and_concat(pred_batches.to("cpu")) # extract corners from predictions
  batch_size = bbox.shape[0] # Extract batch size for looping
  keep_list = [] # Create a list for output

  for b in range(batch_size):
    boxes = bbox[b] # Extract bboxes from corresponding img
    class_probs = boxes[..., 5:] # Extracting classes
    conf = boxes[..., 4].unsqueeze(-1) # Extract confidence and add dimension at the end

    scores = conf * class_probs

    best_scores, best_classes = torch.max(scores, dim = 1)

    # Create a mask and filter by it
    mask = best_scores > conf_threshold

    boxes = boxes[mask]
    best_scores = best_scores[mask]
    best_classes = best_classes[mask]

    # Sort idx in descending order and apply it to masked tensors
    sorted_idx = torch.argsort(best_scores, descending = True)

    boxes = boxes[sorted_idx]
    best_scores = best_scores[sorted_idx]
    best_classes = best_classes[sorted_idx]

    keep = []

    while len(boxes) > 0:
      best_box = boxes[0]
      best_class = best_classes[0]
      xmin, ymin, xmax, ymax = best_box[:4].tolist()

      keep.append([xmin, ymin, xmax, ymax, best_scores[0].item(), best_class.item()]) # [xmin, ymin, xmax, ymax, conf_score, class_idx]

      if len(boxes) == 1:
        break

      rest_boxes = boxes[1:]
      rest_classes = best_classes[1:]
      # print(f"Remaining boxes are: {rest_boxes}") # DEBUG
      ious= get_iou(best_box[:4], rest_boxes[:, :4])

      keep_mask = (ious < iou_threshold) | (rest_classes != best_class) # Mask if the iou is less than the threshold or if the classes are different

      boxes = rest_boxes[keep_mask] # Change the boxes with the ones that are left out
      best_scores = best_scores[1:][keep_mask] # Change the classes
      best_classes = rest_classes[keep_mask]

    keep_list.append(keep) # Append list of bboxes

  return keep_list
