from torch import nn


#####  WORK IN PROGRESS #############


class DartsNormalCell(nn.Module):
    def __init__(self, reduction_p, ch_in, ch_out, kernel_size, stride, padding, affine) -> None:
        if not reduction_p:
            self.s0 = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(ch_in, ch_out, kernel_size, stride=stride,
                        padding=padding, bias=False),
                nn.BatchNorm2d(ch_out, affine=affine)
            )
        else:
            self.s0 = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(ch_in, ch_out//2, 1, stride=2,padding=0, bias=False),
                nn.Conv2d(ch_in, ch_out//2, 1, stride=2,padding=0, bias=False),
                nn.BatchNorm2d(ch_out, affine=affine)
            )

class DartsNet(nn.Module):
    def __init__(self, init_ch_out=16, affine=True):
        super().__init__()
        self._stem0 = nn.Sequential(
            nn.Conv2d(3, init_ch_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(init_ch_out, affine=affine)
        )
        self._stem1 = nn.Sequential(
            nn.Conv2d(3, init_ch_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(init_ch_out, affine=affine)
        )

    def forward(self, x):
