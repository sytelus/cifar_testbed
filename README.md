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
- Mostly just one script file

## Non-Features

To keep code minimal and simple, below are NOT implemented and I have no plans to add these features:

- Tensorboard support
- Distributed or multi-gpu
- Checkpointing

## Install

There is only one dependency:

```
pip install runstats
```
This project is not a module so no need to do anything else.

## Use

Run resnet34 model with optimizer and scheduler as in resnet original paper:

```
python main.py --optim-sched resnet --experiment-name resnet_paper
```

Run resnet18 model with optimizer as in resnet paper but scheduler as in darts paper with **half precision** and `cutout` augmentation of size 8:

```
python main.py --optim-type resnet --sched-type darts --half --cutout 8
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
