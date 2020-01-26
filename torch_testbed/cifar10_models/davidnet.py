import torch
from torch import nn

class DavidNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._init = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            ReLU(inplace=True)
        )
        self._init = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            ReLU(inplace=True)
        )