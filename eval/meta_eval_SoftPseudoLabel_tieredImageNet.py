from __future__ import print_function

import sys
import os
import time
import pickle

from PIL import Image
import numpy as np
import scipy
from scipy.stats import t
from scipy.special import softmax
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from sklearn import metrics
from sklearn.linear_model import LogisticRegression

from models import model_pool
from models.util import create_model
from util import accuracy, AverageMeter


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * t._ppf((1+confidence)/2., n-1)
    return m, h


def normalize(x):
    norm = x.pow(2).sum(1, keepdim=True).pow(1. / 2)
    out = x.div(norm)
    return out


class DistillKL(nn.Module):
    """KL divergence for distillation"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='sum') * (self.T**2) / y_s.shape[0]
        return loss


def meta_test(net, trainloader1, trainloader2, testloader, use_logit=True, is_norm=True, classifier='LR', opt=None):
    net = net.eval()                                   # base model trained on base dataset
    acc = []                                           # baseline accuracy

    if not os.path.exists("tiered-acc1.pkl"):
        acc_new = [[] for i in range( opt.early_stop_steps // opt.print_freq + 1)]  # ours accuracies, at the end of each Finetune epoch
    else:
        with open('tiered-acc1.pkl', 'rb') as f:
            acc_new = pickle.load(f)

    net_state_dict = net.state_dict().copy()          # Starting point for each episode finetuning.
    del net_state_dict['classifier.weight']    # Remove parameters the final classifier
    del net_state_dict['classifier.bias']      # So at the beginning of each episode learning, model.feature is
                                               # initialized with base model, model.classifier is initialized randomly

    for idx, data in tqdm(enumerate(testloader)):

        support_xs, support_ys, query_xs, query_ys, (numpy_support_xs, _) = data
        numpy_support_xs = [x.squeeze(0).numpy() for x in numpy_support_xs]  # novel support images in each episode in numpy format
        support_xs = support_xs.cuda()
        query_xs = query_xs.cuda()
        batch_size, _, channel, height, width = support_xs.size()
        support_xs = support_xs.view(-1, channel, height, width)
        query_xs = query_xs.view(-1, channel, height, width)

        with torch.no_grad():
            if use_logit:
                support_features = net(support_xs).view(support_xs.size(0), -1)
                query_features = net(query_xs).view(query_xs.size(0), -1)
            else:
                feat_support, _ = net(support_xs, is_feat=True)
                support_features = feat_support[-1].view(support_xs.size(0), -1)
                feat_query, _ = net(query_xs, is_feat=True)
                query_features = feat_query[-1].view(query_xs.size(0), -1)

        if is_norm:
            support_features = normalize(support_features)
            query_features = normalize(query_features)

        support_features = support_features.detach().cpu().numpy()
        query_features = query_features.detach().cpu().numpy()

        support_ys = support_ys.view(-1).numpy()
        query_ys = query_ys.view(-1).numpy()

        #  Following the RFS (ECCV 2020) LinearRegression learning
        if classifier == 'LR':
            clf = LogisticRegression(penalty='l2',
                                     random_state=0,
                                     C=1.0,
                                     solver='lbfgs',
                                     max_iter=1000,
                                     multi_class='multinomial')
            clf.fit(support_features, support_ys)
            query_ys_pred = clf.predict(query_features)
        else:
            raise NotImplementedError('classifier not supported: {}'.format(classifier))

        print(' RFS/SKD/baseline acc: {:.4f} for this episode'.format(metrics.accuracy_score(query_ys, query_ys_pred)))
        acc.append(metrics.accuracy_score(query_ys, query_ys_pred))


        # A 5-ways classification model
        learner = create_model(opt.model, opt.n_ways, opt.dataset)   # model for each episode
        learner.load_state_dict(net_state_dict, strict=False)        # model.feature is initialized with base model
                                                                           # model.classifier is initialized randomly, because the state_dict is deleted already

        # optimizer for ResNet12
        optimizer = optim.SGD([
                                {'params': learner.layer1[0].bn1.weight},
                                {'params': learner.layer1[0].bn1.bias},
                                {'params': learner.layer1[0].conv2.weight},
                                {'params': learner.layer1[0].bn2.weight},
                                {'params': learner.layer1[0].bn2.bias},
                                {'params': learner.layer1[0].conv3.weight},
                                {'params': learner.layer1[0].bn3.weight},
                                {'params': learner.layer1[0].bn3.bias},
                                {'params': learner.layer1[0].downsample[0].weight},
                                {'params': learner.layer1[0].downsample[1].weight},
                                {'params': learner.layer1[0].downsample[1].bias},
                                {'params': learner.layer2[0].conv1.weight},
                                {'params': learner.layer2[0].bn1.weight},
                                {'params': learner.layer2[0].bn1.bias},
                                {'params': learner.layer2[0].conv2.weight},
                                {'params': learner.layer2[0].bn2.weight},
                                {'params': learner.layer2[0].bn2.bias},
                                {'params': learner.layer2[0].conv3.weight},
                                {'params': learner.layer2[0].bn3.weight},
                                {'params': learner.layer2[0].bn3.bias},
                                {'params': learner.layer2[0].downsample[0].weight},
                                {'params': learner.layer2[0].downsample[1].weight},
                                {'params': learner.layer2[0].downsample[1].bias},
                                {'params': learner.layer3[0].conv1.weight},
                                {'params': learner.layer3[0].bn1.weight},
                                {'params': learner.layer3[0].bn1.bias},
                                {'params': learner.layer3[0].conv2.weight},
                                {'params': learner.layer3[0].bn2.weight},
                                {'params': learner.layer3[0].bn2.bias},
                                {'params': learner.layer3[0].conv3.weight},
                                {'params': learner.layer3[0].bn3.weight},
                                {'params': learner.layer3[0].bn3.bias},
                                {'params': learner.layer3[0].downsample[0].weight},
                                {'params': learner.layer3[0].downsample[1].weight},
                                {'params': learner.layer3[0].downsample[1].bias},
                                {'params': learner.layer4[0].conv1.weight},
                                {'params': learner.layer4[0].bn1.weight},
                                {'params': learner.layer4[0].bn1.bias},
                                {'params': learner.layer4[0].conv2.weight},
                                {'params': learner.layer4[0].bn2.weight},
                                {'params': learner.layer4[0].bn2.bias},
                                {'params': learner.layer4[0].conv3.weight},
                                {'params': learner.layer4[0].bn3.weight},
                                {'params': learner.layer4[0].bn3.bias},
                                {'params': learner.layer4[0].downsample[0].weight},
                                {'params': learner.layer4[0].downsample[1].weight},
                                {'params': learner.layer4[0].downsample[1].bias},
                                {'params': learner.classifier.weight, 'lr':opt.clf_learning_rate},
                                {'params': learner.classifier.bias, 'lr':opt.clf_learning_rate}
                              ],
                              lr=opt.learning_rate,
                              momentum=opt.momentum,
                              weight_decay=opt.weight_decay)

        kd_criterion = DistillKL(T=opt.T)

        if torch.cuda.is_available():
            learner = nn.DataParallel(learner)
            learner = learner.cuda()
            kd_criterion = kd_criterion.cuda()
            cudnn.benchmark = True

        # routine: supervised pre-training

        for epoch in range(1, opt.epochs + 1):

            # adjust_learning_rate(epoch, opt, optimizer)
            print("==> training...")

            time1 = time.time()
            return_acc = train_one_epoch(epoch, trainloader2, net, clf, numpy_support_xs, support_ys, query_xs, query_ys, learner, kd_criterion, optimizer, opt)
            # scheduler.step()
            time2 = time.time()
            print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

            # record the accuracies
            for e in range(len(return_acc)-1):
                acc_new[e].append(return_acc[e+1])


            # evaluate the model at the end of each epoch
            learner.eval()

            query_pred = learner(query_xs, norm_feat=opt.norm_feat)
            _, query_pred = torch.max(query_pred, dim=1)
            query_ys_pred = query_pred.cpu().numpy()

            acc_new[-1].append(metrics.accuracy_score(query_ys, query_ys_pred))

        # print out
        print('acc1: {:.4f}, std1: {:.4f}, acc2: {:.4f}, std2: {:.4f},'.format(mean_confidence_interval(acc)[0],
                                                                               mean_confidence_interval(acc)[1],
                                                                               mean_confidence_interval(acc_new[-1])[0],
                                                                               mean_confidence_interval(acc_new[-1])[1]))

        # save acc_new into pickle file
        # with open('tiered-acc1.pkl', 'wb') as f:
        #     pickle.dump(acc_new, f)

        # print out for accuracy at the end of each epoch
        for e in range(len(acc_new)-1):
            print('steps: {:d}, acc2: {:.4f}, std2: {:.4f}'.format((e+1)*opt.print_freq,
                                                      mean_confidence_interval(acc_new[e])[0],
                                                      mean_confidence_interval(acc_new[e])[1]))

    return mean_confidence_interval(acc), mean_confidence_interval(acc_new[-1])


def train_one_epoch(epoch, train_loader, net, clf, numpy_support_xs, support_ys, query_xs, query_ys, model, kd_criterion, optimizer, opt):
    """One epoch training"""

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses1 = AverageMeter()    # KD loss on base set examples with pseudo soft labels (in 5 ways)
    losses2 = AverageMeter()    # CrossEntropyLoss on novel support examples with support hard labels
    top1 = AverageMeter()       # accuracy for 'base' images with hard pseudo label, not very useful actually
    top5 = AverageMeter()       # not used at all

    return_acc = []

    # Default data augmentation, from RFS (ECCV 2020)
    if opt.dataset == 'miniImageNet' or opt.dataset == 'tieredImageNet':
        image_mean = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
        image_std = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]
        image_transform = transforms.Compose([
                                                lambda x: Image.fromarray(x),
                                                transforms.RandomCrop(84, padding=8),
                                                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                                transforms.RandomHorizontalFlip(),
                                                lambda x: np.asarray(x),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=image_mean, std=image_std)
                                                ])
    else:
        image_mean = [0.5071, 0.4867, 0.4408]
        image_std = [0.2675, 0.2565, 0.2761]
        image_transform = transforms.Compose([
                                                lambda x: Image.fromarray(x),
                                                transforms.RandomCrop(32, padding=4),
                                                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                                transforms.RandomHorizontalFlip(),
                                                lambda x: np.asarray(x),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=image_mean, std=image_std)
                                                ])

    support_ys = torch.from_numpy(support_ys).long().cuda()
    query_ys = torch.from_numpy(query_ys).long().cuda()


    end = time.time()
    for idx, (input, target, item) in enumerate(train_loader):

        data_time.update(time.time() - end)
        model.train()

        #
        input = input.float()
        input = input.cuda()

        # this version is for tieredImageNet dataset only
        with torch.no_grad():
            input_feat, _ = net(input, is_feat=True)
            input_feat = input_feat[-1].view(input.size(0), -1)
            input_feat = normalize(input_feat)

        input_feat = input_feat.detach().cpu().numpy()
        label = target.numpy() # hard label
        pseudo_y = clf.decision_function(input_feat)

        pseudo_y = torch.from_numpy(pseudo_y).float().cuda()  # pseudo_y (for base set example) is the logit by teacher (base_model.features + newly learned LinearRegression in RFS)
        _, pseudo_target = torch.max(pseudo_y, dim=1)         # pseudo_target (for base set example) is the hard label by teacher (base_model.features + newly learned LinearRegression in RFS)

        support_xs = torch.stack(list(map(lambda x: image_transform(x.squeeze()), numpy_support_xs)))  # apply train_transform onto support_xs
        support_xs = support_xs.cuda()

        # ===================forward=====================
        input = torch.cat((input, support_xs), dim=0)   # concatenate base images with novel support_xs into a single mini-batch, i.e., in miniImageNet, (125,C,H,W) + (125,C,H,W)
        output = model(input, norm_feat=opt.norm_feat)  # end-to-end model with option to normalize the feature before model.classifier
        loss1 = kd_criterion(output[:opt.train_batch_size], pseudo_y)               # first half of mini-batch is base examples, use KD loss
        loss2 = nn.CrossEntropyLoss()(output[opt.train_batch_size:], support_ys)    # second half of mini-batch is novel support example, use CrossEntropyLoss

        loss = opt.alpha1*loss1 + opt.alpha2*loss2

        acc1, acc5 = accuracy(output[:opt.train_batch_size], pseudo_target, topk=(1, 5))  # accuray for base images wrt pseudo hard label, not really useful
        losses1.update(loss1.item(), input[:opt.train_batch_size].size(0))                # input[:opt.train_batch_size].size(0)
        losses2.update(loss2.item(), input[opt.train_batch_size:].size(0))                # input[opt.train_batch_size:].size(0)
        top1.update(acc1[0], input[:opt.train_batch_size].size(0))
        top5.update(acc5[0], input[:opt.train_batch_size].size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # tensorboard logger
        pass

        # print info
        if idx % opt.print_freq == 0:
            # validation on query set
            model.eval()

            query_pred = model(query_xs, norm_feat=opt.norm_feat)
            _, query_pred = torch.max(query_pred, dim=1)
            query_correct = (query_pred == query_ys).sum().item()
            query_accuracy = query_correct / query_xs.size(0)

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss1 {loss1.val:.4f} ({loss1.avg:.4f})\t'
                  'Loss2 {loss2.val:.4f} ({loss2.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Query@1 {query_accuracy:.3f}'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss1=losses1, loss2=losses2,
                   top1=top1, query_accuracy=query_accuracy))
            sys.stdout.flush()

            return_acc.append(query_accuracy)

        if idx == opt.early_stop_steps:
            return return_acc # early stopping for 1-shot cases

    return 0
