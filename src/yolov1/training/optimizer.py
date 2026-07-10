def create_optimizer(model, optim_type, lr):
  optim_list = ['sgd', 'adam', 'adamw']

  if optim_type not in optim_list:
    raise ValueError(f"{optim_type} is not a valid optimizer, pick {optim_list}")
  
  if optim_type == 'adam':
    optimizer = torch.optim.Adam(params = model.parameters(),
                                 lr = lr)
  elif optim_type == 'adamw':
    optimizer = torch.optim.AdamW(params = model.parameters(),
                                  lr = lr,
                                  betas = (0.9, 0.999),
                                  eps = 1e-8,
                                  weight_decay = 0.01)
   elif optim_type == 'sgd':
    optimizer = torch.optim.SGD(params = model.parameters(),
                                lr = lr)
                                
  return optimizer