import torch.nn as nn


class Driver(nn.Module):  
  def __init__(self, backbone: nn.Module, early: str, late: str):
    super().__init__()
    self.backbone = backbone
    self.early = backbone.get_submodule(early)
    self.late = backbone.get_submodule(late)
    # freeze the body, but keep the final classification head trainable
    for name, p in self.backbone.named_parameters():
      if name.startswith("fc."):
        continue
      p.requires_grad_(False)
