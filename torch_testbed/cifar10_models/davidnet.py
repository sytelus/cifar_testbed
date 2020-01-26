import torch
from torch import nn

def davidnet():
    return DavidNet()

class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, weight_freeze=False, bias_freeze=False, weight_init=1.0, bias_init=0.0, **kwargs):
        super().__init__(num_features, eps=eps, momentum=momentum, **kwargs)
        if weight_init is not None: self.weight.data.fill_(weight_init)
        if bias_init is not None: self.bias.data.fill_(bias_init)
        self.weight.requires_grad = not weight_freeze
        self.bias.requires_grad = not bias_freeze

class ConvBn(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, padding=1, stride=1) -> None:
        super().__init__()
        self._op = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            BatchNorm(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self._op(x)

class ConvBnPool(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, padding=1, stride=1) -> None:
        super().__init__()
        self._op = nn.Sequential(
            ConvBn(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self._op(x)

class ConvRes(nn.Module):
    def __init__(self, ch_in, ch_out) -> None:
        super().__init__()
        self._stem = ConvBnPool(ch_in, ch_out)
        self._op =  nn.Sequential(
            ConvBn(ch_out, ch_out),
            ConvBn(ch_out, ch_out)
        )

    def forward(self, x):
        h = self._stem(x)
        return h + self._op(h)

class DavidNet(nn.Module):
    def __init__(self, pretrained=False) -> None:
        super().__init__()
        assert not pretrained
        self._op =  nn.Sequential(
            ConvBn(3, 64),
            ConvRes(64, 64),
            ConvBnPool(64, 64*2),
            ConvRes(64*2, 64*4),
            nn.AdaptiveMaxPool2d(128),
            nn.Linear(in_features=128, out_features=10, bias=True)
        )

    def forward(self, x):
        return self._op(x)
