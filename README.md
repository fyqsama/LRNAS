# LRNAS: Differentiable Searching for Adversarially Robust Lightweight Neural Architecture

This is the code implementation for the paper "LRNAS: Differentiable Searching for Adversarially Robust Lightweight Neural Architecture". 

## Requirements

- Python 3.7.4
- torch 1.8.0 + cu111
- torchvision 0.9.0 + cu111
- torchattacks 3.5.1

## Run

- Run `train_search.py` to perform the search process to search for the CNN architectures.
- Run `adv_train.py` to train the searched architectures on CIFAR-10 or CIFAR-100.
- Run `adv_train_tinyimagenet.py` or `adv_train_imagenet.py` to train architectures on Tiny-ImageNet-200 or ImageNet-1K, respectively.
- Run `adv_test.py` to test the trained architectures on CIFAR-10, CIFAR-100, or Tiny-ImageNet-200. 
- Run `adv_test_imagenet.py` to test the trained architectures on ImageNet-1K.

## References
- [DARTS](https://github.com/quark0/darts)
- [AdvRush](https://github.com/nutellamok/advrush/tree/main)
- [torchattacks](https://github.com/Harry24k/adversarial-attacks-pytorch)
- [Fast Adversarial Training](https://github.com/ByungKwanLee/Super-Fast-Adversarial-Training)
