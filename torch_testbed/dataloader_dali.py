import os
import sys
import time
import pickle
import numpy as np
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from sklearn.utils import shuffle
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, DALIGenericIterator

import torch

class HybridTrainPipe_CIFAR(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, seed,
                 crop=32, dali_cpu=False, local_rank=0, world_size=1,
                 cutout=0, dali_device = "gpu"):
        super(HybridTrainPipe_CIFAR, self).__init__(batch_size, num_threads, device_id, seed=seed + device_id)
        self.external_data = CIFAR_INPUT_ITER(batch_size, 'train', root=data_dir)
        self.iterator = iter(self.external_data)
        self.input = ops.ExternalSource()
        self.input_label = ops.ExternalSource()
        self.pad = ops.Paste(device=dali_device, ratio=1.25, fill_value=0)
        self.uniform = ops.Uniform(range=(0., 1.))
        self.crop = ops.Crop(device=dali_device, crop_h=crop, crop_w=crop)
        self.cmnp = ops.CropMirrorNormalize(device=dali_device,
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            image_type=types.RGB,
                                            mean=[0.49139968 * 255., 0.48215827 * 255., 0.44653124 * 255.],
                                            std=[0.24703233 * 255., 0.24348505 * 255., 0.26158768 * 255.]
                                            )
        self.coin = ops.CoinFlip(probability=0.5)

    def iter_setup(self):
        try:
            (images, labels) = self.iterator.next()
            self.feed_input(self.jpegs, images, layout="HWC")
            self.feed_input(self.labels, labels)
        except StopIteration:
            self.iterator = iter(self.external_data)
            raise StopIteration

    def define_graph(self):
        rng = self.coin()
        self.jpegs = self.input()
        self.labels = self.input_label()
        output = self.jpegs
        output = self.pad(output.gpu())
        output = self.crop(output, crop_pos_x=self.uniform(), crop_pos_y=self.uniform())
        output = self.cmnp(output, mirror=rng)
        return [output, self.labels]


class HybridValPipe_CIFAR(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, seed,
                 crop, local_rank=0, world_size=1, dali_device = "gpu"):
        super(HybridValPipe_CIFAR, self).__init__(batch_size, num_threads, device_id, seed=seed + device_id)
        self.external_data = CIFAR_INPUT_ITER(batch_size, 'val', root=data_dir)
        self.iterator = iter(self.external_data)
        self.input = ops.ExternalSource()
        self.input_label = ops.ExternalSource()
        self.cmnp = ops.CropMirrorNormalize(device=dali_device,
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            image_type=types.RGB,
                                            mean=[0.49139968 * 255., 0.48215827 * 255., 0.44653124 * 255.],
                                            std=[0.24703233 * 255., 0.24348505 * 255., 0.26158768 * 255.]
                                            )

    def iter_setup(self):
        try:
            (images, labels) = self.iterator.next()
            self.feed_input(self.jpegs, images, layout="HWC")
            self.feed_input(self.labels, labels)
        except StopIteration:
            self.iterator = iter(self.external_data)
            raise

    def define_graph(self):
        self.jpegs = self.input()
        self.labels = self.input_label()
        output = self.jpegs
        output = self.cmnp(output.gpu())
        return [output, self.labels]

class DaliLoaderWrapper:
    def __init__(self, dali_loader):
        self._dali_loader = dali_loader
        self._iter = None

    def __iter__(self):
        self._iter = iter(self._dali_loader)
        return self

    def __len__(self):
        return self._dali_loader._size

    def __next__(self):
        assert self._iter
        try:
            data = next(self._iter)
            return data[0]["data"], data[0]["label"].squeeze().long()
        except StopIteration:
            self._dali_loader.reset()
            raise

class CIFAR_INPUT_ITER():
    base_folder = 'cifar-10-batches-py'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    def __init__(self, batch_size, type='train', root='/userhome/memory_data/cifar10'):
        self.root = root
        self.batch_size = batch_size
        self.train = (type == 'train')
        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.targets = np.vstack(self.targets)
        self.data = self.data.transpose((0, 2, 3, 1)).copy()  # convert to HWC

        # TODO: measure perf for below
        # np.save("cifar.npy", self.data)
        # self.data = np.load('cifar.npy')  # to serialize, increase locality

    def __iter__(self):
        # TODO: shuffle?
        self.i = 0
        self.n = len(self.data)
        self.ended = self.i >= self.n
        if self.train:
            self.data, self.targets = shuffle(self.data, self.targets, random_state=0)
        return self

    def __next__(self):
        if self.ended:
            raise StopIteration()
        batch = []
        labels = []
        for _ in range(self.batch_size):
            img, label = self.data[self.i], self.targets[self.i]
            batch.append(img)
            labels.append(label)
            self.i += 1
            if self.i >= self.n:
                self.ended = True
                self.i = self.n - 1
        return (batch, labels)

    next = __next__

def cifar10_dataloaders(datadir:str, train_batch_size=128, test_batch_size=4096,
                        train_num_workers=-1, test_num_workers=-1,
                        cutout=0, seed=42, local_rank=0, world_size=1,
                        train=True, test=True, dali_device='gpu'):

    if train_batch_size <= -1:
        train_num_workers = torch.cuda.device_count()*4
    if test_num_workers <= -1:
        test_num_workers = torch.cuda.device_count()*4

    train_dl, test_dl = None, None
    if train:
        pip_train = HybridTrainPipe_CIFAR(batch_size=train_batch_size, seed=seed,
                                          num_threads=train_num_workers,
                                          device_id=local_rank,
                                          data_dir=datadir,
                                          dali_device=dali_device,
                                          crop=32, world_size=world_size, local_rank=local_rank, cutout=cutout)
        pip_train.build()
        dali_iter_train = DALIClassificationIterator(pip_train, size=50000 // world_size,
                                                     fill_last_batch = False, last_batch_padded = True)
        train_dl = DaliLoaderWrapper(dali_iter_train)

    if test:
        pip_val = HybridValPipe_CIFAR(batch_size=test_batch_size,
                                      num_threads=test_num_workers,
                                      device_id=local_rank,
                                      data_dir=datadir, seed=seed,
                                      dali_device=dali_device,
                                      crop=32,
                                      world_size=world_size, local_rank=local_rank)
        pip_val.build()
        dali_iter_val = DALIClassificationIterator(pip_val, size=10000 // world_size,
                                                   fill_last_batch = False, last_batch_padded = True)
        test_dl = DaliLoaderWrapper(dali_iter_val)

    return train_dl, test_dl
