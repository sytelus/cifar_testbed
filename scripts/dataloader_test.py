import logging

from torch_testbed.dataloader import cifar10_dataloaders
from torch_testbed import utils
from torch_testbed.timing import MeasureTime, print_all_timings, print_timing, get_timing


utils.setup_logging()

datadir = utils.full_path('~/torchvision_data_dir')
train_dl, test_dl = cifar10_dataloaders(datadir,
    train_batch_size=128, test_batch_size=1024,
    cutout=0)

@MeasureTime
def iter_dl(dl):
    dummy = 0.0
    for x,y in train_dl:
        x = x.cuda()
        y = y.cuda()
        dummy += len(x)
        dummy += len(y)
    return dummy

logging.info(f'batch_cout={len(train_dl)}')

dummy = 0.0
for _ in range(5):
    dummy += iter_dl(train_dl)

print(dummy)

print_all_timings()

exit(0)