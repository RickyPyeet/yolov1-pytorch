import torch
from torchvision.transforms import v2

def get_voc_train_transforms(image_size: int = 448):
 return v2.Compose([
      v2.RandomResizedCrop(image_size), # Randomly resize it and crop it again to 448 to match architecture input size
      v2.RandomHorizontalFlip(p = 0.5),
      v2.ClampBoundingBoxes(), # Clamp bboxes if they lay outside the image
      v2.SanitizeBoundingBoxes(), # Remove bounding boxes that are falling out of the img
      v2.RandomPhotometricDistort(brightness = (0.875, 1.125), # These are all standard values
                                  contrast = (0.5, 1.5),
                                  saturation = (0.5, 1.5),
                                  hue = (-0.5, 0.5),
                                  p = 0.5),
      v2.ToDtype(torch.float32, scale = True), # torch.ToTensor() got deprecated in v2 and replaced with `torch.ToDtype()`
      v2.Normalize(mean = [0.485, 0.456, 0.406],
                   std = [0.229, 0.224, 0.225])
      ])
 
def get_voc_val_transforms(image_size: int = 448):
 return v2.Compose([
      v2.Resize(image_size), # Resize to increase img details
      v2.CenterCrop(448), # to keep 448,448 size, otherwise it will resize and keep the aspect ratio of pictures, erroring out because dataset class expect img of size 448x448
      v2.ToDtype(torch.float32, scale = True),
      v2.Normalize(mean = [0.485, 0.456, 0.406],
                   std = [0.229, 0.224, 0.225])
      ])
