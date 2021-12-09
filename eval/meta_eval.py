from __future__ import print_function

import sys
import os
import time

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
    acc_new = [[] for i in range(opt.epochs)]  # ours accuracies, at the end of each Finetune epoch

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

        # print out
        print('acc1: {:.4f}, std1: {:.4f}'.format(mean_confidence_interval(acc)[0],
                                                                               mean_confidence_interval(acc)[1]))

    return mean_confidence_interval(acc), mean_confidence_interval(acc)
