import torch
import math
from torch import nn

class YOLOv1ClassificationBlock(nn.Module):
  def __init__(self, num_classes: int = 200):
    super().__init__()
    self.num_classes = num_classes
    self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
                                    nn.Flatten(),
                                    nn.Linear(in_features = 1024, out_features = self.num_classes))
    nn.init.kaiming_normal_(self.classifier[2].weight)
    nn.init.constant_(self.classifier[2].bias, 0)

  def forward(self, x):
    return self.classifier(x)

class YOLOv1DetectionBlock(nn.Module):
  def __init__(self, S, B, C):
    super().__init__()
    self.detection_structure = [('c', 3, 1024, 1), ('c', 3, 1024, 2),
                                ('c', 3, 1024, 1), ('c', 3, 1024, 1)]
    self.S = S
    self.B = B
    self.C = C

    self.detection_start = self._parse_create_structure(self.detection_structure, in_channels = 1024)
    self.detection_start = nn.Sequential(*self.detection_start)

    self.detection_end = nn.Sequential(nn.Flatten(),
                                       nn.Linear(in_features = 7*7*1024, out_features = 4096),
                                       nn.Dropout(0.5),
                                       nn.LeakyReLU(0.1),
                                       nn.Linear(in_features = 4096, out_features = (self.S*self.S*(self.B*5+self.C))))

    nn.init.kaiming_normal_(self.detection_end[1].weight)
    nn.init.constant_(self.detection_end[1].bias, 0)
    nn.init.kaiming_normal_(self.detection_end[4].weight)
    nn.init.constant_(self.detection_end[4].bias, 0)

  def _parse_create_structure(self, structure, in_channels):

    _structure = []

    for substruct in structure:
      kernel_size, out_channels, stride = substruct[1:]
      padding = math.ceil((kernel_size - stride)/2)
      _structure.append(self._create_block(in_channels = in_channels,
                                            out_channels = out_channels,
                                            kernel_size = kernel_size,
                                            stride= stride,
                                            padding = padding))
      in_channels = out_channels

    return _structure

  def _create_block(self, in_channels, out_channels, kernel_size, stride, padding, leaky_rate = 0.1):
    block = nn.Sequential(nn.Conv2d(in_channels = in_channels,
                                    out_channels = out_channels,
                                    kernel_size = kernel_size,
                                    stride = stride,
                                    padding = padding,
                                    bias = False), # Turned off for batchnorm
                          nn.GroupNorm(num_groups = 32, num_channels = out_channels),
                          #nn.BatchNorm2d(out_channels), # Adding BatchNorm against normal paper adaption
                          nn.LeakyReLU(negative_slope = leaky_rate))
    nn.init.kaiming_normal_(tensor = block[0].weight, a = 0.1, mode = 'fan_out', nonlinearity = 'leaky_relu')

    return block

  def forward(self, x):
    return self.detection_end(self.detection_start(x)) # output (n, S*S*(B*5+C)) -> (n, 7*7*30) -> (n, 1470)
