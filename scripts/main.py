from typing import List, Tuple, Any
import argparse
import logging
import csv
from collections import OrderedDict
import os
import time

from torch_testbed.train_test import train_test
from torch_testbed.timing import MeasureTime, print_all_timings, print_timing, get_timing
from torch_testbed.utils import cuda_device_names

def update_results_file(results:List[Tuple[str, Any]]):
    fieldnames, rows = [], []
    if os.path.exists('./results.tsv'):
        with open('./results.tsv', 'r') as f:
            dr = csv.DictReader(f, delimiter='\t')
            fieldnames = dr.fieldnames
            rows = [row for row in dr.reader]
    if fieldnames is None:
        fieldnames = []

    new_fieldnames = OrderedDict([(fn, None) for fn, v in results])
    for fn in fieldnames:
        new_fieldnames[fn]=None

    with open('./results.tsv', 'w') as f:
        dr = csv.DictWriter(f, fieldnames=new_fieldnames.keys(), delimiter='\t')
        dr.writeheader()
        for row in rows:
            dr.writerow(dict((k,v) for k,v in zip(fieldnames, row)))
        dr.writerow(OrderedDict(results))


@MeasureTime
def main():
    parser = argparse.ArgumentParser(description='Pytorch cifasr testbed')
    parser.add_argument('--experiment-name', '-n', default='throwaway')
    parser.add_argument('--experiment-description', '-d', default='pinmemory=true, 0 workers')
    parser.add_argument('--epochs', '-e', type=int, default=5)
    parser.add_argument('--model-name', '-m', default='resnet18')
    parser.add_argument('--train-batch', '-b', type=int, default=512)
    parser.add_argument('--test-batch', type=int, default=4096)
    parser.add_argument('--seed', '-s', type=int, default=42)
    parser.add_argument('--half', action='store_true', default=False)
    parser.add_argument('--cutout', type=int, default=0)
    parser.add_argument('--loader', default='auto', help='auto, torch, dali')
    parser.add_argument('--loader-workers', type=int, default=4, help='number of thread/workers for data loader')
    parser.add_argument('--sched-type', default='',
                        help='LR scheduler: darts (cosine) or '
                             'resnet (multi-step)'
                             'sc (super convergence)')
    parser.add_argument('--optim-type', default='',
                        help='Optimizer: darts(default) or resnet')
    parser.add_argument('--optim-sched', '-os', default='darts',
                        help='Optimizer and scheduler: darts or resnet')

    args = parser.parse_args()

    args.sched_type = args.sched_type or args.optim_sched
    args.optim_type = args.optim_type or args.optim_sched

    train_acc, test_acc = train_test(args.experiment_name, args.experiment_description,
                     args.epochs, args.model_name, args.train_batch, args.loader_workers,
                     args.seed, args.half, args.test_batch, args.loader,
                     args.cutout, args.sched_type, args.optim_type)
    print_all_timings()
    logging.info(f'test_accuracy={train_acc}')
    print_timing('train_epoch')

    results = [
        ('test_acc', test_acc),
        ('train_epoch_time', get_timing('train_epoch').mean()),
        ('test_epoch_time', get_timing('test').mean()),
        ('epochs', args.epochs),
        ('train_batch_size', args.train_batch),
        ('test_batch_size', args.test_batch),
        ('model_name', args.model_name),
        ('exp_name', args.experiment_name),
        ('exp_desc', args.experiment_description),
        ('seed', args.seed),
        ('devices', cuda_device_names()),
        ('half', args.half),
        ('cutout', args.cutout),
        ('sched_type', args.sched_type),
        ('optim_type', args.optim_type),
        ('train_acc', train_acc),
        ('loader', args.loader),
        ('loader_workers', args.loader_workers),
        ('date', str(time.time())),
    ]

    update_results_file(results)

if __name__ == '__main__':
    main()