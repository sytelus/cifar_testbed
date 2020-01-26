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

Install NVidia Dali. Assuming you have CUDA 10.0 this can be done by,

```
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/10.0 nvidia-dali
```

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


### On laptop (Gigabyte Aero 15 w/ GeForce RTX 2080 with Max-Q Design with tubo disabled):

* Dataset lower bound: Entire cifar10 dataset can be iterated in 3.7sec +/- 1.4s (max 6.2sec) on laptop without cuda transfer, using PyTorch native dataloaders. Amazingly this time remains same with cuda. See dataloader_test.py.
* Model lower bound: Resnet34 model can do forward pass of a 128 batch randomly generated tensors in 0.0129sec. This is 4.878sec/epoch (cifar has 391 batches). This can be brought down to 4.196sec/epoch if we pre-generate all tensors and move to cuda. All of cifar10 tensors combined takes just 615MB in cuda! This baloons to 21.4sec/epoch when forward+backward pass is added when cudnn.benchmark=False.
* Effect of cudnn.benchmark=True on model lower bound: 128 batch size: 4.3s->3.3s without backward pass, 21.91s->12.38s with backward pass

| Model    	| Half  	| cudnn.benchmark 	| Batch 	| Mode         	| num_workers 	| sec/epoch 	|
|----------	|-------	|-----------------	|-------	|--------------	|-------------	|-----------	|
| resnet18 	| FALSE 	| FALSE           	| 128   	| in-memory    	|             	| 14.1      	|
| resnet18 	| FALSE 	| TRUE            	| 128   	| in-memory    	|             	| 7.03      	|
| resnet18 	| FALSE 	| TRUE            	| 128   	| torch-loader 	|             	| 12.98     	|
| resnet18 	| TRUE  	| TRUE            	| 128   	| torch-loader 	|             	| 10.29     	|
| resnet18 	| TRUE  	| TRUE            	| 128   	| in-memory    	|             	| 5.741     	|
| resnet18 	| FALSE 	| TRUE            	| 256   	| in-memory    	|             	| 5.32      	|
| resnet18 	| TRUE  	| TRUE            	| 256   	| in-memory    	|             	| 3.78      	|
| resnet18 	| FALSE 	| TRUE            	| 512   	| in-memory    	|             	| 12.13     	|
| resnet18 	| FALSE	    | TRUE            	| 512   	| torch-loader 	| 4          	| 11.46       	|
| resnet18 	| TRUE 	    | TRUE            	| 512   	| torch-loader 	| 2-4          	| 7.8       	|
| resnet18 	| TRUE 	    | TRUE            	| 512   	| torch-loader 	| 1           	| 13.2      	|
| resnet18 	| TRUE  	| TRUE            	| 512   	| torch-loader 	| 0           	| 15.46     	|
| resnet18 	| TRUE  	| TRUE            	| 512   	| in-memory    	|             	| 2.76      	|
| resnet18 	| TRUE  	| TRUE            	| 1024  	| in-memory    	|             	| 2.35      	|
| resnet18 	| FALSE 	| TRUE            	| 2048  	| in-memory    	|             	| 4.68      	|
| resnet18 	| TRUE  	| TRUE            	| 2048  	| in-memory    	|             	| 2.087     	|
| resnet18 	| TRUE  	| TRUE            	| 4096  	| in-memory    	|             	| 1.95      	|
| resnet18 	| TRUE  	| TRUE            	| 8192  	| in-memory    	|             	| 1.75      	|
| resnet18 	| TRUE  	| TRUE            	| 16384 	| in-memory    	|             	| 1.69      	|
| resnet34 	| FALSE 	| FALSE           	| 128   	| in-memory    	|             	| 20.71     	|
| resnet34 	| FALSE 	| TRUE            	| 128   	| in-memory    	|             	| 12.38     	|
| resnet34 	| FALSE 	| TRUE            	| 128   	| torch-loader 	|             	| 21.16     	|
| resnet34 	| TRUE  	| TRUE            	| 128   	| torch-loader 	|             	| 16.4       	|
| resnet34 	| FALSE 	| TRUE            	| 256   	| in-memory    	|             	| 9.41      	|
| resnet34 	| TRUE  	| TRUE            	| 256   	| in-memory    	|             	| 7.39      	|
| resnet34 	| FALSE 	| TRUE            	| 512   	| in-memory    	|             	| 7.44      	|
| resnet34 	| FALSE 	| TRUE            	| 512   	| torch-loader 	|             	| 18.34      	|
| resnet34 	| TRUE  	| TRUE            	| 512   	| torch-loader 	|             	| 12.66      	|
| resnet34 	| TRUE  	| TRUE            	| 512   	| in-memory    	|             	| 5.68      	|
| resnet34 	| FALSE 	| TRUE            	| 2048  	| in-memory    	|             	| 7.2       	|
| resnet34 	| TRUE  	| TRUE            	| 2048  	| in-memory    	|             	| 4.517     	|
|          	|       	|                 	|       	|              	|             	|           	|

**Remarks**

* V100 is 2X faster than my laptop RTX2080 GPUs in fp16 as well as fp32 across batch sizes.
* Best epoch/secis achieved for fp16 batch size for 3.2sec/epoch with resnet18
* Number of workers can make 2X difference on torch dataloader. Min value 2, ideally 4 per GPU.
* cudnn.benchmark=True makes 2X difference
* Doubling model size increases epoch time by 60-70% across batch sizes regardless of fp16 or fp32
* fp16 reduces epoch time by 23-27%
* Dali and torchdataloaders show similar performance overall even though dali loaders have much higher throughput
* In memory random tensors vs cifar tensors have more than 2X difference!

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
* Use apex instead of .half()
* Use LARS https://github.com/kakaobrain/torchlars

## Credits

* [This repo](https://github.com/akamaster/pytorch_resnet_cifar10) was my starting point as well as provided baseline for resnet.
* cifar10 converted models for torch vision comes from [this repo](https://github.com/huyvnphan/PyTorch-CIFAR10) by Huy Phan.
* [This repo](https://github.com/kentaroy47/pytorch-cifar10-fp16) has benchmarks for some fp16 experiments.
* https://towardsdatascience.com/diving-into-dali-1c30c28731c0