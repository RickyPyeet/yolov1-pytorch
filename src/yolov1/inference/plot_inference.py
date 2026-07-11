import cv2
import numpy as np

import torch
import torchvision
from torchvision.transforms import v2

from PIL import Image
from pathlib import Path

from src.yolov1.utils.nms import compute_nms

def inference_transform():
  inf_transforms = v2.Compose([v2.Resize(448),
                              v2.CenterCrop(448),
                              v2.ToDtype(torch.float32, scale = True),
                              v2.Normalize(mean = [0.485, 0.456, 0.406],
                                            std = [0.229, 0.224, 0.225])
                              ])
  return inf_transforms

def draw_img_bbox(img_path: str | Path, pred: torch.Tensor, nms_conf_threshold: float = 0.14):
  classes = ['aeroplane','bicycle','bird','boat','bottle',
            'bus','car','cat','chair','cow',
            'diningtable','dog','horse','motorbike','person',
            'pottedplant','sheep','sofa','train','tvmonitor']

  colors = [(255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
            (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
            (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
            (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255),
            (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199)]

  img = Image.open(img_path)
  width, height = img.size
  img = np.array(img)
  good_bbox = compute_nms(pred.reshape(1,7,7,30), conf_threshold = nms_conf_threshold)

  base_width = 640
  font_scale = max(width, height) / base_width * 0.5
  thickness = max(1, int(font_scale * 2))

  for img_bbox in good_bbox:
    for b in img_bbox:
      xmin, ymin, xmax, ymax, conf, class_idx = b
      xmin = int(xmin * width)
      xmax = int(xmax * width)
      ymin = int(ymin * height)
      ymax = int(ymax * height)
      label = classes[class_idx]
      color = colors[class_idx]

      text = f"{label}: {conf:.2f}"
      (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
      x_text = xmin
      y_text = max(ymin-10, text_h+5)

      cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)

      cv2.rectangle(img, (x_text, y_text - text_h - baseline-3), (x_text + text_w, y_text + baseline), color, -1)

      cv2.putText(img, text, (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness)
  return img
