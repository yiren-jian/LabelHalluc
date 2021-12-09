# Label Hallucination for Few-Shot Classification

This repo covers the implementation of the following paper:  **"Label Hallucination for Few-Shot Classification"** .
If you find this repo useful for your research, please consider citing the paper.
```bibtex
@article{Jian2022LabelHalluc,
    author = {Yiren Jian and Lorenzo Torresani},
    title = {Label Hallucination for Few-shot Classification},
    journal = {AAAI},
    year = {2022}
}
```

## Requirements

This repo was tested with Ubuntu 18.04.5 LTS, Python 3.6, PyTorch 1.4.0, and CUDA 10.1. You will need at least 32GB RAM and 22GB VRAM (i.e. two Nvidia RTX-2080Ti) for running full experiments in this repo.

## Download Data
The data we used here is preprocessed by the repo of [MetaOptNet](https://github.com/kjunelee/MetaOptNet), Please find the renamed versions of the files in below link by [RFS](https://github.com/WangYueFt/rfs).

Download and unzip the dataset, put them under ```data``` directory.

## Embedding Learning
Please follow [RFS](https://github.com/WangYueFt/rfs), [SKD](https://github.com/brjathu/SKD) and [Rizve et al.](https://github.com/nayeemrizve/invariance-equivariance) (or other transfer learning methods) for the embedding learning. [RFS](https://github.com/WangYueFt/rfs) provides a Dropbox link for downloading their pre-trained models for miniImageNet.

We provide our pretrained embedding models by [SKD] and [Rizve et al.] at [Dropbox](https://www.dropbox.com/sh/6af4q91qrvv4t7u/AACrC960J_sc85dlYh0-K_MSa?dl=0). Note that this is NOT an official release of models by original authors, and those models perform slightly worse than what reported in their papers. Better models could be trained with longer durations and/or by hyper-parameters tuning.

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

## Reading the outputs
```
400it RFS/SKD/baseline acc: 0.7200 for this episode
==> training...
Epoch: [1][100/288]    Time 0.121 (0.115)    Data 0.001 (0.003)    ..
Epoch: [1][200/288]    Time 0.112 (0.114)    Data 0.001 (0.002)    ...
epoch 400, total time 32.77
acc1: 0.6567, std1: 0.0076, acc2: 0.6820, std2: 0.0080,
epochs: 1, acc2: 0.6400, std2: 0.0080
...
```
The above is an example print-out for FC100 5-shot. ```acc1: 0.6567, std1: 0.0076``` is the accuracy and the deviation of LinearRegression method with fixed embeddings (used in [RFS](https://github.com/WangYueFt/rfs) and [SKD](https://github.com/brjathu/SKD)). ```acc2: 0.6820, std2: 0.0080``` is the result by our method.

## Contacts
For any questions, please contact authors.


## Acknowlegements
Thanks to [RFS](https://github.com/WangYueFt/rfs), for the preliminary implementations.
