import torch

DEFAULT_OPERATIONS = [
    (torch, "add"),
    (torch.Tensor, "add"),
    (torch, "cos"),
    (torch.nn.modules.conv.Conv2d, "forward"),
    (torch.nn.modules.Linear, "forward"),
    (torch.nn.modules.MaxPool2d, "forward"),
    (torch.nn.modules.Flatten, "forward"),
    (torch.nn.modules.BatchNorm2d, "forward"),
    (torch.nn.functional, "relu"),
    (torch.Tensor, "view"),
    (torch.Tensor, "size")
]

