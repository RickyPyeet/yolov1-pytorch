import torch
from torch import nn

from src.yolov1.utils.boxes import bbox_centers_to_corners
from src.yolov1.utils.iou import get_iou

class YOLOLoss(nn.Module):
  def __init__(self, S, B, C, L_coord = 5, L_noobj = 0.5):
    super().__init__()

    # Define Lambdas
    self.L_coord = L_coord
    self.L_noobj = L_noobj

    self.S = S
    self.B = B
    self.C = C

    # Define MSE
    self.mse = nn.MSELoss(reduction = "sum") # Sum since losses aren't averaged

  def forward(self, y_pred, y_target):
    # Reshape tensors from flat to (same, 7, 7, 30)
    y_pred = y_pred.reshape(-1, self.S, self.S, self.C+self.B*5)
    y_target = y_target.reshape(-1, self.S, self.S, self.C+self.B*5)

    # Extract bboxes
    target_bbox = y_target[..., :4] # [..., [x_c, y_c, w, h]]
    bbox_one = y_pred[..., :4] # 1st bbox data
    bbox_two = y_pred[..., 5:9] # 2nd bbox data

    # Extract object existence from target
    obj = y_target[..., 4].unsqueeze(-1) # Mask if section contains bbox with obj

    # Get IoU
    bbox_centers_one = bbox_centers_to_corners(bbox_one, S = self.S)
    bbox_centers_two = bbox_centers_to_corners(bbox_two, S = self.S)
    target_corners = bbox_centers_to_corners(target_bbox, S = self.S)

    iou_bbox_one = get_iou(bbox_centers_one, target_corners).unsqueeze(-1) # unsqueeze dim to concatenate them
    iou_bbox_two = get_iou(bbox_centers_two, target_corners).unsqueeze(-1) # unsqueeze dim to concatenate them


    # iou_bbox_concat = torch.cat((masked_iou_bbox_one, masked_iou_bbox_two), dim = -1) # new cell 24/03/2026
    iou_bbox_concat = torch.cat((iou_bbox_one, iou_bbox_two), dim = -1) # Removed because we use the masked iou instead 24/03/2026

    # Get idx of max of the two bboxes
    best_iou, best_iou_idx= torch.max(iou_bbox_concat, dim = -1)
    best_iou = best_iou.unsqueeze(-1)
    responsible_mask = best_iou_idx.unsqueeze(-1) # Boolean mask to specify what bbox of the two is best

    # Calculate 1_{i,j}^{obj}*x -> is bbox responsible for detecting obj and mask all those responsibles
    responsible_bbox = (1 - responsible_mask) * bbox_one + responsible_mask * bbox_two # if bbox one -> idx = 0 -> (1*bbox_one)+ 0*bbox_two

    target_bbox = target_bbox * obj
    masked_bbox = obj*responsible_bbox # masks the best box IF cell has object in it - otherwise 0

    # Extract bbox height and width and compute sqrt
    target_hw = torch.sqrt(target_bbox[..., 2:4]) # target_bbox already has mask to identify Cell with object in it
    responsible_bbox_hw = torch.sign(masked_bbox[..., 2:4])*torch.sqrt(torch.abs(masked_bbox[...,2:4]) + 1e-6) # If starting values are negative, keep sign and compute absolute value before sqrt. Add 1e-6 to avoid derivative(sqrt(0)) -> infinity during gradient

    # 1. Compute center loss
    bbox_center_loss = self.mse(target_bbox[...,:2], masked_bbox[...,:2])

    # 2. Compute height and width loss ###
    bbox_hw_loss = self.mse(target_hw, responsible_bbox_hw)

    # 3. Box confidence loss if object exists
    target_conf = y_target[..., 4].unsqueeze(-1)
    bbox_one_conf = y_pred[..., 4].unsqueeze(-1)
    bbox_two_conf = y_pred[..., 9].unsqueeze(-1)

    responsible_bbox_conf = (1 - responsible_mask) * bbox_one_conf + responsible_mask * bbox_two_conf
    masked_responsible = obj * responsible_bbox_conf # Show responsible bbox confidence ONLY IF cell has object

    exist_target_conf = obj * target_conf # Mask target confidence ONLY IF cell has obj

    obj_conf_loss = self.mse(exist_target_conf * best_iou.detach(), masked_responsible) # ADDED '.DETACH()' - 24/03/2026 # Multiply the target objectness by IoU to get correct confidence score

    # 4. Box conf if object does NOT exist
    no_obj = 1 - obj # Mask if section does not contain bbox with obj
    # Make no-responsible mask
    no_obj_conf_loss = self.mse(no_obj * target_conf, no_obj * bbox_one_conf)
    no_obj_conf_loss += self.mse(no_obj * target_conf, no_obj * bbox_two_conf)

    # 5. Class prediction loss
    target_class = obj * y_target[..., self.B*5:]
    pred_class = obj * y_pred[..., self.B*5:]

    class_loss = self.mse(target_class, pred_class)

    # Compute total loss
    total_loss = self.L_coord * (bbox_center_loss + bbox_hw_loss) + obj_conf_loss + self.L_noobj*no_obj_conf_loss + class_loss
    total_loss /= y_pred.shape[0] # Divide by batch size

    return total_loss
