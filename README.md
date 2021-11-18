# Label Hallucination for Few-Shot Classification (Coming Soon)

This repo covers the implementation of the following paper:  **"Label Hallucination for Few-Shot Classification"** .
If you find this repo useful for your research, please consider citing the paper.

## Requirements

This repo was tested with Ubuntu 18.04.5 LTS, Python 3.6, PyTorch 1.4.0, and CUDA 10.1. You will need at least 32GB RAM and 22GB VRAM (i.e. two Nvidia RTX-2080Ti) for running full experiments in this repo.

## Download Data
The data we used here is preprocessed by the repo of [MetaOptNet](https://github.com/kjunelee/MetaOptNet), Please find the renamed versions of the files in below link by [RFS](https://github.com/WangYueFt/rfs).

Download and unzip the dataset, put them under ```data``` directory.

## Embedding Learning
Please follow [RFS](https://github.com/WangYueFt/rfs), [SKD](https://github.com/brjathu/SKD) and [Rizve et al.](https://github.com/nayeemrizve/invariance-equivariance) for the embedding learning. Besides, [RFS](https://github.com/WangYueFt/rfs) provides a Dropbox link for downloading their pre-trained models for miniImageNet.

Once finish the embedding training, put the pre-trained models in ```models_pretrained``` directory.

## Running Our Fine-tuning
To perform 5-way 5-shot classifications, run:
```shell
# For CIFAR-FS
CUDA_VISIBLE_DEVICES=0 python -W ignore eval_fewshot_SoftPseudoLabel.py --dataset CIFAR-FS --data_root data/CIFAR-FS/ --model_path models_pretrained/cifar-fs_skd_gen1.pth --n_shot 5 --n_aug_support 5 --epoch 1 --norm_feat

# For FC100
CUDA_VISIBLE_DEVICES=0 python -W ignore eval_fewshot_SoftPseudoLabel.py --dataset FC100 --data_root data/FC100/ --model_path models_pretrained/fc100_skd_gen1.pth --n_shot 5 --n_aug_support 5 --epoch 1 --norm_feat

# For miniImageNet (require multiple GPUs)
CUDA_VISIBLE_DEVICES=0,1 python -W ignore eval_fewshot_SoftPseudoLabel.py --dataset miniImageNet --data_root data/miniImageNet/ --model_path models_pretrained/mini_skd_gen1.pth --n_shot 5 --n_aug_support 5 --epoch 1 --norm_feat

# For tieredImageNet (require multiple GPUs)
CUDA_VISIBLE_DEVICES=0,1 python -W ignore eval_fewshot_SoftPseudoLabel_tieredImageNet.py --dataset tieredImageNet --data_root data/tieredImageNet/ --model_path models_pretrained/tiered_skd_gen0.pth --n_shot 5 --n_aug_support 5  --early 200 --print 50 --norm_feat
```
To perform 5-way 1-shot classifications, run:
```shell
# For CIFAR-FS
CUDA_VISIBLE_DEVICES=0 python -W ignore eval_fewshot_SoftPseudoLabel.py --dataset CIFAR-FS --data_root data/CIFAR-FS/ --model_path models_pretrained/cifar-fs_skd_gen1.pth --n_shot 1 --n_aug_support 25 --epoch 3 --norm_feat

# For FC100
CUDA_VISIBLE_DEVICES=0 python -W ignore eval_fewshot_SoftPseudoLabel.py --dataset FC100 --data_root data/FC100/ --model_path models_pretrained/fc100_skd_gen1.pth --n_shot 1 --n_aug_support 25 --epoch 5 --norm_feat

# For miniImageNet (require multiple GPUs)
CUDA_VISIBLE_DEVICES=0,1 python -W ignore eval_fewshot_SoftPseudoLabel.py --dataset miniImageNet --data_root data/miniImageNet/ --model_path models_pretrained/mini_skd_gen1.pth --n_shot 1 --n_aug_support 25 --early 150 --norm_feat

# For tieredImageNet (require multiple GPUs)
CUDA_VISIBLE_DEVICES=0,1 python -W ignore eval_fewshot_SoftPseudoLabel_tieredImageNet.py --dataset tieredImageNet --data_root data/tieredImageNet/ --model_path models_pretrained/tiered_skd_gen0.pth --n_shot 1 --n_aug_support 25  --early 200 --print 50 --norm_feat
```


## Acknowlegements
Thanks to [RFS](https://github.com/WangYueFt/rfs), for the preliminary implementations.
