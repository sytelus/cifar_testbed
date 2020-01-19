import torch
import numpy as np

from torch_testbed import utils, cifar10_models
from torch_testbed.timing import MeasureTime, print_all_timings, print_timing, get_timing

utils.setup_logging()

#utils.setup_cuda(42)

batch_size = 128
half = False
model = cifar10_models.resnet18().cuda()
lr, momentum, weight_decay = 0.025, 0.9, 3.0e-4
optim = torch.optim.SGD(model.parameters(),
                        lr, momentum=momentum, weight_decay=weight_decay)
crit = torch.nn.CrossEntropyLoss().cuda()

if half:
    model = model.half()
    crit = crit.half()

@MeasureTime
def iter_dl(ts):
    dummy = 0.0
    for x, l in ts:
        y = model(x)
        dummy += len(y)
        loss = crit(y, l)
        optim.zero_grad()
        loss.backward()
        optim.step()
    return dummy

dummy = 0.0
for _ in range(5):
    data = [(torch.rand(batch_size, 3, 12, 12).cuda() \
                if not half else torch.rand(batch_size, 3, 12, 12).cuda().half(), \
            torch.LongTensor(batch_size).random_(0, 10).cuda()) \
            for _ in range(round(50000/batch_size))]
    dummy += iter_dl(data)

print(dummy)

print_all_timings()

exit(0)