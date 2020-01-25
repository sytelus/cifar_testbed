from typing import List, Tuple, Any
import argparse
import time


from torch_testbed.train_test import train_test
from torch_testbed.timing import MeasureTime, print_all_timings, print_timing, get_timing
from torch_testbed import utils


@MeasureTime
def main():
    parser = argparse.ArgumentParser(description='Pytorch cifasr testbed')
    parser.add_argument('--experiment-name', '-n', default='throwaway')
    parser.add_argument('--experiment-description', '-d', default='throwaway')
    parser.add_argument('--epochs', '-e', type=int, default=35)
    parser.add_argument('--model-name', '-m', default='resnet34')
    parser.add_argument('--train-batch', '-b', type=int, default=128)
    parser.add_argument('--test-batch', type=int, default=4096)
    parser.add_argument('--seed', '-s', type=int, default=42)
    parser.add_argument('--half', action='store_true', default=False)
    parser.add_argument('--cutout', type=int, default=0)
    parser.add_argument('--loader', default='torch', help='torch or dali')
    parser.add_argument('--loader-workers', type=int, default=-1, help='number of thread/workers for data loader (-1 means auto)')
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

    metrics = train_test(args.experiment_name, args.experiment_description,
                     args.epochs, args.model_name, args.train_batch, args.loader_workers,
                     args.seed, args.half, args.test_batch, args.loader,
                     args.cutout, args.sched_type, args.optim_type)

    print_all_timings()
    print_timing('train_epoch')

    results = [
        ('test_acc', metrics[-1]['test_top1']),
        ('train_epoch_time', get_timing('train_epoch').mean()),
        ('test_epoch_time', get_timing('test').mean()),
        ('epochs', args.epochs),
        ('train_batch_size', args.train_batch),
        ('test_batch_size', args.test_batch),
        ('model_name', args.model_name),
        ('exp_name', args.experiment_name),
        ('exp_desc', args.experiment_description),
        ('seed', args.seed),
        ('devices', utils.cuda_device_names()),
        ('half', args.half),
        ('cutout', args.cutout),
        ('sched_type', args.sched_type),
        ('optim_type', args.optim_type),
        ('train_acc', metrics[-1]['train_top1']),
        ('loader', args.loader),
        ('loader_workers', args.loader_workers),
        ('date', str(time.time())),
    ]

    utils.append_csv_file('./results.tsv', results)

if __name__ == '__main__':
    main()