import torch
import math
from torch import nn

from src.yolov1.models.backbone import YOLOv1BackboneBlock
from src.yolov1.models.heads import YOLOv1ClassificationBlock, YOLOv1DetectionBlock

# Create a YOLOv1 class
class YOLOv1(nn.Module):

  def __init__(self,
               S: int = 7,
               B: int = 2,
               C: int = 20,
               mode:str = "detection",
               classific_num_classes: int = 200):
    """
    Creates a YOLOv1 model.
    Args:
      - S (int): the grid size, default (7,7)
      - B (int): the number of bboxes per grid section, default 2
      - C (int): the number of classes for classification in detection, default 20 - PascalVOC
      - mode (str): ("detection / classification") whether model needs to be used for detection (train / inference) or classification (pre-train)
      - classific_num_classes (int): number of classes used for pre-training in classificaton mode - default 200, Tiny-ImageNet 200
    """
    super().__init__()
    self.S = S
    self.B = B
    self.C = C
    self.mode = mode
    self.num_classes = classific_num_classes
    # Create a backbone structure
    self.backbone = YOLOv1BackboneBlock()

    # Check what mode we are using YOLOv1 - classification / detection
    if self.mode == "classification":
      self.classifier = YOLOv1ClassificationBlock(num_classes = self.num_classes)

    elif self.mode == "detection":
      self.detect = YOLOv1DetectionBlock(S = self.S,
                                         B = self.B,
                                         C = self.C)

  def forward(self, x):
    if self.mode == "detection":
      return self.detect(self.backbone(x))
    else:
      return self.classifier(self.backbone(x))
