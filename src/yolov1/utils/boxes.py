import torch

def bbox_centers_to_corners(bbox_data: torch.Tensor, S: int) -> torch.Tensor:
  """
  Extrapolates the bbox corners xmin, ymin, xmax, ymax from the output tensor.
  args:
    bbox_data (torch.tensor): torch tensor with size (n, S, S, 4)
  outputs:
    xmin, ymin, xmax, ymax
  """
  x_c = bbox_data[..., 0].unsqueeze(-1)
  y_c = bbox_data[..., 1].unsqueeze(-1)
  bbox_w = bbox_data[..., 2].unsqueeze(-1)
  bbox_h = bbox_data[..., 3].unsqueeze(-1)

  # Make centers relative to img
  grid = torch.arange(S, device = bbox_data.device)
  i_grid = grid.view((1,S,1,1))
  j_grid = grid.view((1,1,S,1))
  x_c_img = (x_c + j_grid) / S
  y_c_img = (y_c + i_grid) / S

  # Compute corners
  xmin = x_c_img - (bbox_w / 2)
  ymin = y_c_img - (bbox_h / 2)
  xmax = x_c_img + (bbox_w / 2)
  ymax = y_c_img + (bbox_h / 2)

  corners = torch.cat((xmin, ymin, xmax, ymax), dim = -1)

  return corners
