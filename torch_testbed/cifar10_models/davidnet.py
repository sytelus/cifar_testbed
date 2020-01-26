import torch
from torch import nn

class DavidNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._layers = nn.Sequential(
            nn.Conv2d
        )