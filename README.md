# Label Hallucination for Few-Shot Classification

This repo covers the implementation of the following paper:  **[Label Hallucination for Few-Shot Classification](https://arxiv.org/abs/2112.03340)** ([Appendix](https://cs.dartmouth.edu/~yirenjian/data/LabelHalluc-appendix.pdf)) by [Yiren Jian](https://cs.dartmouth.edu/~yirenjian/) and [Lorenzo Torresani](https://ltorresa.github.io/home.html), presented at AAAI-2022.
If you find this repo useful for your research, please consider citing the paper.

```bibtex
@inproceedings{Jian2022LabelHalluc,
  author = {Jian, Yiren and Torresani, Lorenzo},
  title = {Label Hallucination for Few-shot Classification},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year = {2022}
}
```

```bibtex
@article{jian2021label,
  title = {Label Hallucination for Few-Shot Classification},
  author = {Jian, Yiren and Torresani, Lorenzo},
  journal = {arXiv preprint arXiv:2112.03340},
  year = {2021}
}
```

## Requirements

This repo was tested with Ubuntu 18.04.5 LTS, Python 3.6, PyTorch 1.4.0, and CUDA 10.1. You will need at least 32GB RAM and 22GB VRAM (i.e. two Nvidia RTX-2080Ti) for running full experiments in this repo.

It's also tested on a machine with Python 3.6, PyTorch 1.10 and CUDA 11, with RTX A6000 (conda environment for this configuration is provided in `labelhalluc.yml`).

## Download Data
The data we used here is preprocessed by the repo of [MetaOptNet](https://github.com/kjunelee/MetaOptNet), Please find the renamed versions of the files in below link by [RFS](https://github.com/WangYueFt/rfs).

Download and unzip the dataset, put them under ```data``` directory.

## Embedding Learning
Please follow [RFS](https://github.com/WangYueFt/rfs), [SKD](https://github.com/brjathu/SKD) and [Rizve et al.](https://github.com/nayeemrizve/invariance-equivariance) (or other transfer learning methods) for the embedding learning. [RFS](https://github.com/WangYueFt/rfs) provides a Dropbox link for downloading their pre-trained models for miniImageNet.

We provide our pretrained embedding models by [SKD] and [Rizve et al.] at [Dropbox]([https://www.dropbox.com/sh/6af4q91qrvv4t7u/AACrC960J_sc85dlYh0-K_MSa?dl=0](https://www.dropbox.com/sh/ikipligbneta9qk/AABew7LSYDMG7lbSC9BQgcMsa?dl=0)). Note that those models are NOT the official release by original authors, and they perform slightly worse than what reported in their papers. Better models could be trained with longer durations and/or by hyper-parameters tuning.

Once finish the embedding training, put the pre-trained models in ```models_pretrained``` directory.

**Update**: If you consider training your own embedding models, I created a new github [page](https://github.com/yiren-jian/embedding-learning-FSL). It corrects some issues in SKD and IER (minor) to run smoothly on my own machine.

## Running Our Fine-tuning
To perform 5-way 5-shot classifications, run:
```shell
# For CIFAR-FS
CUDA_VISIBLE_DEVICES=0 python -W ignore eval_fewshot_SoftPseudoLabel.py --dataset CIFAR-FS --data_root data/CIFAR-FS/ --model_path models_pretrained/cifar-fs_skd_gen1.pth --n_shot 5 --n_aug_support 5 --epoch 1 --norm_feat

# For FC100
CUDA_VISIBLE_DEVICES=0 python -W ignore eval_fewshot_SoftPseudoLabel.py --dataset FC100 --data_root data/FC100/ --model_path models_pretrained/fc100_skd_gen1.pth --n_shot 5 --n_aug_support 5 --epoch 1 --norm_feat

# For miniImageNet (require multiple GPUs, or one GPU with 24 GB)
CUDA_VISIBLE_DEVICES=0,1 python -W ignore eval_fewshot_SoftPseudoLabel.py --dataset miniImageNet --data_root data/miniImageNet/ --model_path models_pretrained/mini_skd_gen1.pth --n_shot 5 --n_aug_support 5 --epoch 1 --norm_feat

# For tieredImageNet (require multiple GPUs, or one GPU with 24 GB)
CUDA_VISIBLE_DEVICES=0,1 python -W ignore eval_fewshot_SoftPseudoLabel_tieredImageNet.py --dataset tieredImageNet --data_root data/tieredImageNet/ --model_path models_pretrained/tiered_skd_gen0.pth --n_shot 5 --n_aug_support 5  --early 200 --print 50 --norm_feat
```
To perform 5-way 1-shot classifications, run:
```shell
# For CIFAR-FS
CUDA_VISIBLE_DEVICES=0 python -W ignore eval_fewshot_SoftPseudoLabel.py --dataset CIFAR-FS --data_root data/CIFAR-FS/ --model_path models_pretrained/cifar-fs_skd_gen1.pth --n_shot 1 --n_aug_support 25 --epoch 3 --norm_feat

# For FC100
CUDA_VISIBLE_DEVICES=0 python -W ignore eval_fewshot_SoftPseudoLabel.py --dataset FC100 --data_root data/FC100/ --model_path models_pretrained/fc100_skd_gen1.pth --n_shot 1 --n_aug_support 25 --epoch 5 --norm_feat

# For miniImageNet (require multiple GPUs, or one GPU with 24 GB)
CUDA_VISIBLE_DEVICES=0,1 python -W ignore eval_fewshot_SoftPseudoLabel.py --dataset miniImageNet --data_root data/miniImageNet/ --model_path models_pretrained/mini_skd_gen1.pth --n_shot 1 --n_aug_support 25 --early 150 --norm_feat

# For tieredImageNet (require multiple GPUs, or one GPU with 24 GB)
CUDA_VISIBLE_DEVICES=0,1 python -W ignore eval_fewshot_SoftPseudoLabel_tieredImageNet.py --dataset tieredImageNet --data_root data/tieredImageNet/ --model_path models_pretrained/tiered_skd_gen0.pth --n_shot 1 --n_aug_support 25  --early 200 --print 50 --norm_feat
```

Overall, our method has more significant improvements in 5-shot (than 1-shot).

## Additional experiments
### Strong results in 10-way and 20-way
During the review of NeurIPS, we found that our method can achieve even larger gains in 10-ways and 20-ways settings.

We apply Label-Halluc (our method pretrained w/ SKD) on 10-way 5-shot and 20-way 10-shot classification problems on FC-100, CIFAR-FS and miniImageNet. We use a 5x data augmentation for both SKD and our method in the 10-way 5-shot setting, the same data augmentation strategy used in the main paper. Due to the limitation of GPU memory, for the 20-way 10-shot setting, we do not use data augmentation for SKD or our method. The following results use the same learning policy discussed in the main paper, with the exception of finetuning for 2 epochs in the case of FC-100 and CIFAR-FS.

| FC100 |     10-way 5-shot     |     20-way 10-shot     |
| :---: | :-------------------: | :--------------------: |
|  SKD  |    46.66 +/- 0.49     |     37.55 +/- 0.22     |
| ours  |    49.63 +/- 0.49     |     42.21 +/-0.22      |

Similarly, we carry out 10-way and 20-way experiments on CIFAR-FS, our finetuning runs for 2 epochs on baseset.
| CIFAR-FS |     10-way 5-shot     |    20-way 10-shot      |
| :------: | :-------------------: | :--------------------: |
| SKD 	   |     79.01+/-0.80      |   71.33+/-0.30         |
| ours 	   |     80.99+/-0.75      |   75.40+/-0.31         |

​We also try on miniImageNet. Due to the limitation of GPU memory, we apply no data augmentation for neither SKD or ours. Our finetuning runs for 300 steps on baseset.
| miniImageNet |      10-way 5-shot     |       20-way 10-shot    |
| :----------: | :--------------------: | :---------------------: |
| SKD          |     70.46 +/- 0.40     |      57.09 +/- 0.28     |
| ours         |     72.41 +/- 0.45     |      62.14 +/-0.27      |

### Base and novel sets are far away
Reviewers also raised questions on how the method performs when the base and novel sets are far away (i.e., base set has only animal classes and novel set has only non-animal classes).

We use the 16 classes of animals from the 64 class of miniImageNet base set to construct our new base dataset. The set of novel classes is now restricted to 11 classes of non-animals out of the original 20 classes. During meta-testing, we sample 5 classes from those 11 novel classes to construct our 5-way episodes. To accelerate the evaluation time, we follow the shorthened-learning strategy already explored during the rebuttal. This reduces the finetuning steps of Label-Halluc from ~300 steps to 160 steps.

|     	     |  SKD-GEN0  |  label-hallc  |
| :--------: | :--------: | :-----------: |
|  1-shot    |    39.38	  |      43.56	  |
|  5-shot    |    56.78   |      64.99    |

The reviewer further pointed to experiments on Meta-dataset (Triantafilou et al., 2020, Meta-dataset: A dataset of datasets for learning to learn from few examples. ICLR 2020), which we do not have.

I believe those are experiments to strengthen one's submission in this area.

## Tuning the hyper-parameters
Better results can be obtained by tuning hyper-parameters in our method. `--alpha1` and `--alpha2` (default to 1) are used to control the weighting between CE loss on support set and KD loss on pseudo-labeled baseset. By default, we also keep 1:1 ratio of base and novel images in a finetuning batch. This can be tuned by setting `--n_aug_batch_sizes` (default to 1). For example, having `--n_aug_batch_sizes 2` will lead to 1:2 ratio of novel and base images in a finetuning batch. `--T` controls the temperature scaling in the distillation loss. Total learning steps can also be adjusted by `--early_stop_steps`.

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
Thanks to [RFS](https://github.com/WangYueFt/rfs) for the preliminary implementations, [SKD](https://github.com/brjathu/SKD) and [Rizve et al.](https://github.com/nayeemrizve/invariance-equivariance) for their embedding pre-training scripts.
