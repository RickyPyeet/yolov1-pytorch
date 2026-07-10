import torch
import torchvision
import pandas as pd

from torchvision import tv_tensors
from torchvision.transforms import v2
from torch.utils.data import Dataset
from PIL import Image

from typing import List, Tuple

class VOCDataset(Dataset):
  def __init__(self,
               csv_file_path: str,
               transforms: v2.Transform,
               S: int = 7,
               B: int = 2,
               C: int = 20):

    self.csv_file_path = csv_file_path
    self.transforms = transforms
    self.S = S
    self.B = B
    self.C = C

    self.df = pd.read_csv(csv_file_path)

    self.classes = ['aeroplane','bicycle','bird','boat','bottle',
                    'bus','car','cat','chair','cow',
                    'diningtable','dog','horse','motorbike','person',
                    'pottedplant','sheep','sofa','train','tvmonitor']

    self.label_to_idx = {clas: i for i, clas in enumerate(self.classes)} # dictionary of labels for one-hot encoding
    self.unique_items = self.df['img_path'].unique() # list of files for getitem idx



  def __len__(self):
    return len(self.unique_items)

  # Create a transform function
  def _img_bbox_transform(self,
                          img,
                          bbox_data: List[List[float]],
                          labels,
                          transforms: v2.Transform) -> Tuple:
    """
    Function created to transform images and their bounding boxes as well.
    It uses torchvision.transforms.v2 module to transform bboxes too.
    args:
      img (C, H, W): torch.Tensor image to transform
      bbox_data: list of bboxes inside the image with format [[x1,y1,x2,y2],...]. They must be in format: (x_topleft, y_topleft, x_botright, y_botright)
      transforms: Compose of transforms to be used
    output:
      out_img: transformed img
      out_boxes: transformed bboxes
    """
    # Create v2 BoundingBoxes from coordinates x1,y1,x2,y2
    boxes = tv_tensors.BoundingBoxes(bbox_data,
                                    format = 'XYXY', # Data received will be formated as x_topl, y_topl, x_bottomr, y_bottomr
                                    canvas_size = img.shape[-2:]) # Original image size
    target = {"boxes": boxes,
              "labels": labels}

    out_img, out_target= transforms(img, target)
    out_boxes = out_target["boxes"]
    out_labels = out_target["labels"]

    return out_img, out_boxes, out_labels

  def _get_centers_and_size(self, bbox: List) -> List:
    new_bbox = []
    for box in bbox:
      xmin, ymin, xmax, ymax = box
      width = xmax-xmin
      height = ymax-ymin
      x_c = (xmax + xmin)/2
      y_c = (ymax + ymin)/2
      new_bbox.append([x_c, y_c, width, height])
    return new_bbox

  def _normalize_bbox(self, transformed_img: torch.tensor, bbox_extracted: List):
    _, new_height, new_width = transformed_img.shape
    normalized_bbox = []
    for box in bbox_extracted:
      x_c, y_c, width, height = box
      x_c = x_c/new_width
      y_c = y_c/new_height
      width = width/new_width
      height = height/new_height
      normalized_bbox.append([x_c, y_c, width, height])
    return normalized_bbox

  def _extract_bbox_section_adapt_centers(self, normalized_bbox: List):
    """Adapt centers relative to the section and extract sections with centers"""
    sections = []
    for box in normalized_bbox:
      x_c, y_c, _, _ = box
      j_section = min(int(x_c * self.S), self.S-1)
      i_section = min(int(y_c * self.S), self.S-1)
      x_c_section = (x_c * self.S) - j_section
      y_c_section = (y_c * self.S) - i_section

      box[0] = x_c_section
      box[1] = y_c_section

      sections.append([i_section, j_section])

    return sections

  def __getitem__(self, idx):
    img_path = self.unique_items[idx]
    # Get bboxes data
    target_data = self.df[self.df['img_path'] == img_path]
    # Open image
    img = Image.open(img_path).convert("RGB")
    img_tensor = torchvision.transforms.v2.ToImage()(img)
    # Extract bboxes xyxy coordinates as list of lists
    bbox_data = [data[-4:] for data in target_data.values.tolist()]
    # Get labels for each bbox
    labels = torch.tensor([self.label_to_idx[data[1]] for data in target_data.values.tolist()], dtype = torch.int64)
    # Transform bbox
    target_img, transformed_bbox, transformed_labels = self._img_bbox_transform(img = img_tensor,
                                                            bbox_data = bbox_data,
                                                            labels = labels,
                                                            transforms = self.transforms)
    # Turn them into centers, width and height
    bbox_extracted = self._get_centers_and_size(transformed_bbox.tolist())
    # Normalize them and adapt height and width to transformed img
    normalized_bbox = self._normalize_bbox(transformed_img = target_img,
                                          bbox_extracted = bbox_extracted)
    # Turn centers relative to section and extract coordinates of sections with centers
    sections = self._extract_bbox_section_adapt_centers(normalized_bbox = normalized_bbox)
    # Create empty tensor with shape (1, S, S, B*5+C)
    target_label = torch.zeros((self.S, self.S, self.B*5+self.C), dtype = torch.float32)
    # Fill target
    for n, section in enumerate(sections):
      i, j = section
      # Add only one object/bbox per section
      if target_label[i,j,4] == 0:
        x_c, y_c, width, height = normalized_bbox[n]
        target_label[i,j,:4] = torch.tensor([x_c, y_c, width, height], dtype = target_label.dtype) # Add bbox coordinates to section
        target_label[i,j,4] = 1 # Set bbox confidence score to 1
        target_label[i,j,(self.B*5)+transformed_labels[n]] = 1 # set correct class label to 1

    return target_img, target_label
