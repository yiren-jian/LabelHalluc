from __future__ import print_function

import os
import argparse
import time
import sys

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from models import model_dict, model_pool
from models.util import create_model

from dataset.mini_imagenet import ImageNet, MetaImageNet
from dataset.tiered_imagenet import TieredImageNet, MetaTieredImageNet
from dataset.cifar import CIFAR100, MetaCIFAR100
from dataset.transform_cfg import transforms_options, transforms_list

from eval.meta_eval_SupportFinetuneOnly import meta_test


def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    # load pretrained model
    parser.add_argument('--model', type=str, default='resnet12', choices=model_pool)
    parser.add_argument('--model_path', type=str, default='./models_pretrained/mini_distilled.pth', help='absolute path to pretrained base model')

    # dataset
    parser.add_argument('--dataset', type=str, default='miniImageNet', choices=['miniImageNet', 'tieredImageNet',
                                                                                'CIFAR-FS', 'FC100'])
    parser.add_argument('--transform', type=str, default='A', choices=transforms_list)

    # meta setting
    parser.add_argument('--n_test_runs', type=int, default=600, metavar='N',
                        help='Number of test runs')
    parser.add_argument('--n_ways', type=int, default=5, metavar='N',
                        help='Number of classes for doing each classification run')
    parser.add_argument('--n_shots', type=int, default=5, metavar='N',
                        help='Number of shots in test')
    parser.add_argument('--n_queries', type=int, default=15, metavar='N',
                        help='Number of query in test')
    parser.add_argument('--n_aug_support_samples', default=5, type=int,
                        help='The number of augmented samples for each meta test sample')
    parser.add_argument('--n_aug_batch_sizes', default=1, type=int,
                        help='The number of augmented samples for each meta test sample')
    parser.add_argument('--data_root', type=str, default='./data/miniImageNet/', metavar='N',
                        help='Root dataset')
    parser.add_argument('--num_workers', type=int, default=2, metavar='N',
                        help='Number of workers for dataloader')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='test_batch_size',
                        help='Size of test batch)')

    # retraining
    parser.add_argument('--norm_feat', action='store_true', help='normalize feature')
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--epochs', type=int, default=1, help='number of training epochs for each episode')
    parser.add_argument('--early_stop_steps', type=int, default=-1, help='number of training steps for each episode (use early stopping)')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.025, help='learning rate')
    parser.add_argument('--clf_learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # distillation
    parser.add_argument('--T', type=float, default=4.0, help='temperature in KD')
    parser.add_argument('--alpha1', type=float, default=1.0, help='alpha weight on loss1')
    parser.add_argument('--alpha2', type=float, default=1.0, help='alpha weight on loss2')

    opt = parser.parse_args()

    # opt.lr_decay_epochs = [int(item) for item in opt.lr_decay_epochs.split(',')]
    if opt.early_stop_steps > 0:
        opt.epochs = 1

    if 'trainval' in opt.model_path:
        opt.use_trainval = True
    else:
        opt.use_trainval = False

    # set the path according to the environment
    opt.data_aug = True

    return opt


def main():
    # RuntimeError: received 0 items of ancdata
    # https://discuss.pytorch.org/t/runtimeerror-received-0-items-of-ancdata/4999/4
    torch.multiprocessing.set_sharing_strategy('file_system')

    opt = parse_option()
    for arg in vars(opt):
        print(arg, getattr(opt, arg))

    # test loader
    args = opt
    opt.train_batch_size = args.n_ways * args.n_shots * args.n_aug_support_samples # for miniImageNet, 125=5x5x5

    opt.train_batch_size = opt.train_batch_size * opt.n_aug_batch_sizes

    train_partition = 'trainval' if opt.use_trainval else 'train'
    if opt.dataset == 'miniImageNet':
        train_trans, test_trans = transforms_options[opt.transform]
        train_loader1 = DataLoader(ImageNet(args=opt, partition=train_partition, transform=test_trans),
                                   batch_size=opt.train_batch_size, shuffle=False, drop_last=False,
                                   num_workers=opt.num_workers)
        train_loader2 = DataLoader(ImageNet(args=opt, partition=train_partition, transform=train_trans),
                                   batch_size=opt.train_batch_size, shuffle=True, drop_last=True,
                                   num_workers=opt.num_workers)
        meta_testloader = DataLoader(MetaImageNet(args=opt, partition='test',
                                                  train_transform=train_trans,
                                                  test_transform=test_trans,
                                                  fix_seed=False),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaImageNet(args=opt, partition='val',
                                                 train_transform=train_trans,
                                                 test_transform=test_trans,
                                                 fix_seed=False),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        if opt.use_trainval:
            n_cls = 80
        else:
            n_cls = 64
    elif opt.dataset == 'tieredImageNet':
        train_trans, test_trans = transforms_options[opt.transform]
        train_loader1 = DataLoader(TieredImageNet(args=opt, partition=train_partition, transform=test_trans),
                                   batch_size=opt.train_batch_size, shuffle=False, drop_last=False,
                                   num_workers=opt.num_workers)
        train_loader2 = DataLoader(TieredImageNet(args=opt, partition=train_partition, transform=train_trans),
                                   batch_size=opt.train_batch_size, shuffle=True, drop_last=True,
                                   num_workers=opt.num_workers)
        meta_testloader = DataLoader(MetaTieredImageNet(args=opt, partition='test',
                                                        train_transform=train_trans,
                                                        test_transform=test_trans,
                                                        fix_seed=False),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaTieredImageNet(args=opt, partition='val',
                                                       train_transform=train_trans,
                                                       test_transform=test_trans,
                                                       fix_seed=False),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        if opt.use_trainval:
            n_cls = 448
        else:
            n_cls = 351
    elif opt.dataset == 'CIFAR-FS' or opt.dataset == 'FC100':
        train_trans, test_trans = transforms_options['D']
        train_loader1 = DataLoader(CIFAR100(args=opt, partition=train_partition, transform=test_trans),
                                   batch_size=opt.train_batch_size, shuffle=False, drop_last=False,
                                   num_workers=opt.num_workers)
        train_loader2 = DataLoader(CIFAR100(args=opt, partition=train_partition, transform=train_trans),
                                   batch_size=opt.train_batch_size, shuffle=True, drop_last=True,
                                   num_workers=opt.num_workers)
        meta_testloader = DataLoader(MetaCIFAR100(args=opt, partition='test',
                                                  train_transform=train_trans,
                                                  test_transform=test_trans,
                                                  fix_seed=False),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaCIFAR100(args=opt, partition='val',
                                                 train_transform=train_trans,
                                                 test_transform=test_trans,
                                                 fix_seed=False),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        if opt.use_trainval:
            n_cls = 80
        else:
            if opt.dataset == 'CIFAR-FS':
                n_cls = 64
            elif opt.dataset == 'FC100':
                n_cls = 60
            else:
                raise NotImplementedError('dataset not supported: {}'.format(opt.dataset))
    else:
        raise NotImplementedError(opt.dataset)

    opt.n_cls = n_cls

    # load model
    base_model = create_model(opt.model, n_cls, opt.dataset)
    ckpt = torch.load(opt.model_path)['model']

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in ckpt.items():
        name = k.replace("module.","")
        new_state_dict[name]=v

    base_model.load_state_dict(new_state_dict, strict=False)

    if torch.cuda.is_available():
        base_model = base_model.cuda()
        cudnn.benchmark = True

    # evalation
    start = time.time()
    (test_acc_feat, test_std_feat), (test_acc2_feat, test_std2_feat) = meta_test(base_model, train_loader1, train_loader2, meta_testloader, use_logit=False, opt=opt)
    test_time = time.time() - start
    print('test_acc_feat: {:.4f}, test_std: {:.4f}, test_acc2_feat: {:.4f}, test_std2: {:.4f} time: {:.1f}'.format(test_acc_feat,
                                                                                                                   test_std_feat,
                                                                                                                   test_acc2_feat,
                                                                                                                   test_std2_feat,
                                                                                                                   test_time))


if __name__ == '__main__':
    main()
