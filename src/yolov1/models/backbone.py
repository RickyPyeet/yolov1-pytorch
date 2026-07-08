import torch
import math
from torch import nn
from typing import List

class YOLOv1BackboneBlock(nn.Module):
  def __init__(self) -> List[nn.Module]:
    """
    Creates the Backbone for YOLOv1
    """
    super().__init__()
    self.backbone_structure = [('c', 7, 64, 2), ('mp', 2, 2),
                               ('c', 3, 192, 1), ('mp', 2, 2),
                               ('c', 1, 128, 1), ('c', 3, 256, 1), ('c', 1, 256, 1), ('c', 3, 512, 1), ('mp', 2, 2),
                               [('c', 1, 256, 1), ('c', 3, 512, 1), 4], ('c', 1, 512, 1), ('c', 3, 1024, 1), ('mp', 2, 2),
                               [('c', 1, 512, 1), ('c', 3, 1024, 1), 2]]




    self.backbone = self._parse_create_structure(self.backbone_structure, in_channels = 3)
    self.backbone = nn.Sequential(*self.backbone)

  def _parse_create_structure(self, structure, in_channels):

    _structure = []

    for substruct in structure:
      if isinstance(substruct, tuple):
        if substruct[0] == 'c':
          kernel_size, out_channels, stride = substruct[1:]
          padding = math.ceil((kernel_size - stride)/2)
          _structure.append(self._create_block(in_channels = in_channels,
                                               out_channels = out_channels,
                                               kernel_size = kernel_size,
                                               stride= stride,
                                               padding = padding))
          in_channels = out_channels

        elif substruct[0] == 'mp':
          kernel_size, stride = substruct[1:]
          block = nn.MaxPool2d(kernel_size = kernel_size, stride = stride)
          _structure.append(block)

      elif isinstance(substruct, list):
        n_repeats = substruct[-1]
        for _ in range(n_repeats): # how many times the block needs to be repeated
          for i in range(len(substruct[:-1])): # parse the list for cnn tuples
            kernel_size, out_channels, stride = substruct[i][1:]
            padding = math.ceil((kernel_size - stride)/2)
            _structure.append(self._create_block(in_channels = in_channels,
                                                 out_channels = out_channels,
                                                 kernel_size = kernel_size,
                                                 stride = stride,
                                                 padding = padding))
            in_channels = out_channels

    return _structure

  def _create_block(self, in_channels, out_channels, kernel_size, stride, padding, leaky_rate = 0.1):
    block = nn.Sequential(nn.Conv2d(in_channels = in_channels,
                                    out_channels = out_channels,
                                    kernel_size = kernel_size,
                                    stride = stride,
                                    padding = padding,
                                    bias = False),
                          nn.BatchNorm2d(out_channels),
                          nn.LeakyReLU(negative_slope = leaky_rate))
    nn.init.kaiming_normal_(tensor = block[0].weight, a = 0.1, mode = 'fan_out', nonlinearity = 'leaky_relu')

    return block

  def forward(self, x):
    return self.backbone(x)
