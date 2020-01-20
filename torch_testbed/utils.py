from typing import List, Optional, Tuple, Any
import os
import sys
import logging
import csv
from collections import OrderedDict

import torch
import torch.backends.cudnn as cudnn
import numpy as np

def is_debugging()->bool:
    return 'pydevd' in sys.modules # works for vscode

def full_path(path:str)->str:
    path = os.path.expandvars(path)
    path = os.path.expanduser(path)
    return os.path.abspath(path)

def setup_logging(filepath:Optional[str]=None,
                  name:Optional[str]=None, level=logging.INFO) -> None:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.propagate = False # otherwise root logger prints things again

    if filepath:
        fh = logging.FileHandler(filename=full_path(filepath))
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

def setup_cuda(seed):
    # setup cuda
    cudnn.enabled = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True

def cuda_device_names()->str:
    return ', '.join([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

def append_csv_file(filepath:str, new_row:List[Tuple[str, Any]], delimiter='\t'):
    fieldnames, rows = [], []
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            dr = csv.DictReader(f, delimiter=delimiter)
            fieldnames = dr.fieldnames
            rows = [row for row in dr.reader]
    if fieldnames is None:
        fieldnames = []

    new_fieldnames = OrderedDict([(fn, None) for fn, v in new_row])
    for fn in fieldnames:
        new_fieldnames[fn]=None

    with open(filepath, 'w') as f:
        dr = csv.DictWriter(f, fieldnames=new_fieldnames.keys(), delimiter=delimiter)
        dr.writeheader()
        for row in rows:
            dr.writerow(dict((k,v) for k,v in zip(fieldnames, row)))
        dr.writerow(OrderedDict(new_row))
