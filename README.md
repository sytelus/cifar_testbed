# Cifar10 Testbed for PyTorch

This repo is born out of my frustration of not having a good standard PyTorch code to experiment on cifar10. Believe it or not but cifar10 is one of the most popular dataset to experiment new techniques and benchmark results BUT there is easy to find code that is clean, minimal, dependency free and still implementing all the best practices for speed and accuracy.

## Features

- PyTorch 1.x, Python 3.6+ compatible
- Use built-in optimizers, schedulers etc from PyTorch
- Use `torchvision` models modified for cifar10
- Dependency free, minimal code that implements all the best practices
- Reproduces original resnet paper results for sanity
- Simple experiment management, folder with all artifacts and logging
- Report timings for performance analysis
- Half precision support for newer GPUs


## Non-Features

To keep code minimal, fast and simple below are currently not implemented:

- Tensorboard support
- Distributed or multi-gpu
- Checkpointing

## Install
From the repo directory, run:

```
pip install -e .
```

## Use

Run resnet34 model with optimizer and scheduler as in resnet original paper:

```
python scripts/main.py --optim-sched resnet --experiment-name resnet_paper
```

Run resnet18 model with optimizer as in resnet paper but scheduler as in darts paper with **half precision** and `cutout` augmentation of size 8:

```
python scripts/main.py --optim-type resnet --sched-type darts --half --cutout 8
```

## Initial Results

All results were obtained on NVidia V100 single GPU. All results are preliminery which means I haven't confirmed 100% by comparing many runs yet and there might be creeping errors in me trying to get through dozens of logs during sleepless night. So run things yourself and feel free to provide any updates.

Check [results folder](https://github.com/sytelus/cifar_testbed/tree/master/results).

* Reproduced 7.37% error on test, 200 epochs, original resnet sgd settings at 10.4s/epoch
* darts optimizer settings is better than resnet paper settings (89.9% vs 88.3%), with scheduler darts even pushes 90.3% in 35 epochs
* darts pushes 90.0% with half prec @ 9s/epoch (it's not great compared to 10.4s/epoch at fp32 but I suspect PyTorch DataLoaders are the bottleneck)
* cutout=16 isn't adding epoch time but also isn't improving acc with darts settings
* resnet18 with resnet setting ups accuracy by 1.6% i.e. 90.3% compared to resnet34, in 35 epochs with epoch @ 6.6s
* resnet18 with darts setting + consine sched pushes 89.7% in 35 epochs, epoch @ 6.6s
* AdamW with no sched and defaults (or from paper) gives samilar result as resnet settings (88.9) but epoch @ 14s
* Best I have got so far is 89.8% accuracy in 35 epochs, 185 seconds, 5.2s/epoch using resnet18 with darts optimizer setting and half precision.


### On laptop:

* Entire cifar10 dataset can be iterated in 3.7sec +/- 1.4s (max 6.2sec) on laptop without cuda transfer, using PyTorch native dataloaders. Amazingly this time remains same with cuda. See dataloader_test.py
* On laptop: Resnet34 model can do forward pass of a 128 batch of in memory randomly generated tensor in 0.0129sec. This is 4.878sec/epoch (cifar has 391 batches). This can be brought down to 4.196sec/epoch if we pregenerate all tensors and move to cuda. All of cifar10 tensors combined takes just 615MB in cuda! This baloons to 21.4sec/epoch when forward+backward pass is added.
* setup_cuda, 128 batch size: 4.3s->3.3s without backward pass, 21.91s->12.38s with backward pass
* resnet34 model_test -> 21.63s/epoch with setup_cuda and backward pass
* 512 batch size gets 12.13s/epoch forward+backward, resnet18
* 512 batch size + fp16 gets 2.76s/epoch forward+backward, resnet18
* 1024 batch size + fp16 gets 2.349s/epoch forward+backward, resnet18
* 2048 batch size + fp16 gets 2.087s/epoch forward+backward, resnet18
* 2048*2 batch size + fp16 gets 1.948s/epoch forward+backward, resnet18
* 2048*4 batch size + fp16 gets 1.748s/epoch forward+backward, resnet18
* 2048*8 batch size + fp16 gets 1.687s/epoch forward+backward, resnet18

* main script dataloader + backward pass, 128 batch: resnet18->12.98, resnet34->21.16
* main script dataloader + backward pass, 512 batch + fp16: resnet18->7.695, resnet34->12.57
* main script dataloader + backward pass, 512 batch + fp16 + 0 workers: resnet18->15.46
* main script dataloader + backward pass, 512 batch + fp16 + 1 workers: resnet18->13.2
* main script dataloader + backward pass, 512 batch + fp16 + 2 workers: resnet18->7.726



## Work-In-Progress

Please consider contributing!

* Implement AdamW + Superconvergence - 94% in 30 epochs or with test time aug, 18 epochs (there is no code whatsoever to this date that does this using built-in AdamW and OneCycleLR in PyTorch, or even anywhere close to these numbers)
* Implement [Resnet9 (DavidNet) with CyclicLR](https://www.kesci.com/home/project/5dab446e1035d8002c363a66), 91.3% in 24 epochs
* Implement [cifar10_faster](https://github.com/vfdev-5/cifar10-faster) -94% under 20 epochs
* Translate [Tensorflow code](https://medium.com/fenwicks/tutorial-2-94-accuracy-on-cifar10-in-2-minutes-7b5aaecd9cdd) - 95% in 24 epochs
* Try augmentations from FastAutoAugmentation and FasterAutoAugmentation
* Implement test time augmentation
* Try [Ranger (RAdam+LookAhead)](https://medium.com/@lessw/new-deep-learning-optimizer-ranger-synergistic-combination-of-radam-lookahead-for-the-best-of-2dc83f79a48d)
* Try wideresnets
* Try cutout=8, mixup, cutmix augmentations
* Try PyTorch
* Add [prefetching and other tricks](https://sagivtech.com/2017/09/19/optimizing-pytorch-training-code/) for speed
* Try [NVidia Dali](https://github.com/tanglang96/DataLoaders_DALI)


## Credits

* [This repo](https://github.com/akamaster/pytorch_resnet_cifar10) was my starting point as well as provided baseline for resnet.
* cifar10 converted models for torch vision comes from [this repo](https://github.com/huyvnphan/PyTorch-CIFAR10) by Huy Phan.
* [This repo](https://github.com/kentaroy47/pytorch-cifar10-fp16) has benchmarks for some fp16 experiments.
