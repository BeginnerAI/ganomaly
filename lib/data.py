#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : lib/data.py
# Author            : Tianming Jiang <djtimy920@gmail.com>
# Date              : 19.12.2018
# Last Modified Date: 31.12.2018
# Last Modified By  : Tianming Jiang <djtimy920@gmail.com>
"""
LOAD DATA from file.
可以分为3个主要的步骤：
1）利用torchvision中的MNIST模块下载MINST数据，并对数据进行transform；
2）按照配置的异常值将MNIST数据划分为正常样本和异常样本，训练集包含80%的正常样本，测试集包含20%的正常样本和全部的异常样本；
3）torch.utils.data.DataLoader将数据转化为PyTorchm模型的输入格式
"""

##
import os
import torch
import numpy as np
from torchvision.datasets import MNIST
import torch.utils.data as dataf
from torch.utils.data import Dataset, Subset, ConcatDataset
import torchvision.transforms as transforms

import glob
import pandas as pd
import numpy as np

##
def load_data(opt):
    """ 
    Args:
        opt ([type]): Argument Parser
    Raises:
        IOError: Cannot Load Dataset
    Returns:
        [type]: dataloader
    """

    ##
    # LOAD DATA SET
    if opt.dataroot == '':
        opt.dataroot = './data/{}'.format(opt.dataset)

    if opt.dataset in ['disk']:
        splits = ['train', 'test']
        drop_last_batch = {'train': True, 'test': False}
        shuffle = {'train': True, 'test': True}

        # transforms.Compose就是将transforms组合在一起
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(32),
                transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
            ]
        )

        good_sample = LoadDiskDataset(opt, 'good_sample', transform)
        failed_sample = LoadDiskDataset(opt, 'failed_sample', transform)

        dataset = {}
        dataset['train'], dataset['test'] = get_disk_anomaly_dataset(good_sample, failed_sample, 1)

        # 生成一个dict
        # DataLoader中的shuffle，洗牌。默认设置为False。在每次迭代训练时是否将数据洗牌，默认设置是False。
        # 将输入数据的顺序打乱，是为了使数据更有独立性，但如果数据是有序列特征的，就不要设置成True了。
        # 在disk failure prediction中是不能够打乱次序的，因为SMART是有序的
        dataloader = {x: torch.utils.data.DataLoader(dataset=dataset[x],
                                                     batch_size=opt.batchsize,
                                                     shuffle=shuffle[x],
                                                     num_workers=int(opt.workers),
                                                     drop_last=drop_last_batch[x]) for x in splits}
        return dataloader
    ##
    elif opt.dataset in ['mnist']:
        opt.anomaly_class = int(opt.anomaly_class)

        splits = ['train', 'test']
        drop_last_batch = {'train': True, 'test': False}
        shuffle = {'train': True, 'test': True}

        transform = transforms.Compose(
            [
                transforms.Scale(opt.isize),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]
        )

        dataset = {}
        dataset['train'] = MNIST(root='./data', train=True, download=True, transform=transform)
        dataset['test'] = MNIST(root='./data', train=False, download=True, transform=transform)

        dataset['train'].train_data, dataset['train'].train_labels, \
        dataset['test'].test_data, dataset['test'].test_labels = get_mnist_anomaly_dataset(
            trn_img=dataset['train'].train_data,
            trn_lbl=dataset['train'].train_labels,
            tst_img=dataset['test'].test_data,
            tst_lbl=dataset['test'].test_labels,
            abn_cls_idx=opt.anomaly_class
        )

        dataloader = {x: torch.utils.data.DataLoader(dataset=dataset[x],
                                                     batch_size=opt.batchsize,
                                                     shuffle=shuffle[x],
                                                     num_workers=int(opt.workers),
                                                     drop_last=drop_last_batch[x]) for x in splits}
        return dataloader

##
def get_disk_anomaly_dataset(good_sample, failed_sample, manualseed=-1):
    """
    Arguments:
        good_sample {Dataset} -- total good sample
        failed_sample {Dataset} -- total failed sample
    Returns:
        [Tensor] -- New training-test images and labels.
    """
    if manualseed != -1:
        # Split the normal data into the new train and tests.
        idx = np.arange(len(good_sample))
        np.random.seed(manualseed)
        # np.random.shuffle(idx)
        
        # 取80%的健康样本到训练集中
        good_train_lens = int(len(idx) * 0.80)
        good_train_idx = idx[:good_train_lens]
        good_test_idx = idx[good_train_lens:]
        
        good_train_sample = Subset(good_sample, good_train_idx)
        good_test_sample = Subset(good_sample, good_test_idx)

        train_sample = good_train_sample
        test_sample = ConcatDataset((good_test_sample, failed_sample))

    return train_sample, test_sample


class LoadDiskDataset(Dataset):
    def __init__(self, opt, label=None, transform=None):
        opt.logger.info('loading data for {}'.format(label))
        self.transform = transform

        data_dir = '/home/jtm/data/cnn_disk/intermediate_result/data_2017_with_change_rate/baidu'
        sn_dir = os.path.join(data_dir, label)
        sn_list = glob.glob('{0}/*'.format(sn_dir))
        self.sample_list = []
        for index, sn in enumerate(sn_list):
            if index == opt.n_sn:
                break
            self.sample_list.extend(glob.glob('{0}/*'.format(sn)))

        assert label == 'good_sample' or label == 'failed_sample', 'label must be good_sample or failed_sample!'
        if label == 'good_sample':
            self.label = 0
        else:
            self.label = 1

        self.count = len(self.sample_list)
        self.data_x = np.empty((self.count,1,12,12), dtype='float32')
        self.data_y = []

        i = 0
        for sample in self.sample_list:
            one_figure = pd.read_csv(sample, header=None)
            arr = one_figure.values
            self.data_x[i,:,:,:] = arr
            i+=1
            self.data_y.append(self.label)

        self.data_y = np.asarray(self.data_y)

        self.data_x = torch.from_numpy(self.data_x)
        self.data_y = torch.from_numpy(self.data_y)

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        feature = self.data_x[idx]
        label = self.data_y[idx]
        if self.transform:
            feature = self.transform(feature)

        return [feature, label]

##
def get_mnist_anomaly_dataset(trn_img, trn_lbl, tst_img, tst_lbl, abn_cls_idx=0, manualseed=-1):
    """[summary]

    Arguments:
        trn_img {np.array} -- Training images
        trn_lbl {np.array} -- Training labels
        tst_img {np.array} -- Test     images
        tst_lbl {np.array} -- Test     labels

    Keyword Arguments:
        abn_cls_idx {int} -- Anomalous class index (default: {0})

    Returns:
        [np.array] -- New training-test images and labels.
    """
    # --
    # Find normal abnormal indexes.
    nrm_trn_idx = torch.from_numpy(np.where(trn_lbl.numpy() != abn_cls_idx)[0])
    abn_trn_idx = torch.from_numpy(np.where(trn_lbl.numpy() == abn_cls_idx)[0])
    nrm_tst_idx = torch.from_numpy(np.where(tst_lbl.numpy() != abn_cls_idx)[0])
    abn_tst_idx = torch.from_numpy(np.where(tst_lbl.numpy() == abn_cls_idx)[0])

    # --
    # Find normal and abnormal images
    nrm_trn_img = trn_img[nrm_trn_idx]    # Normal training images
    abn_trn_img = trn_img[abn_trn_idx]    # Abnormal training images.
    nrm_tst_img = tst_img[nrm_tst_idx]    # Normal training images
    abn_tst_img = tst_img[abn_tst_idx]    # Abnormal training images.

    # --
    # Find normal and abnormal labels.
    nrm_trn_lbl = trn_lbl[nrm_trn_idx]    # Normal training labels
    abn_trn_lbl = trn_lbl[abn_trn_idx]    # Abnormal training labels.
    nrm_tst_lbl = tst_lbl[nrm_tst_idx]    # Normal training labels
    abn_tst_lbl = tst_lbl[abn_tst_idx]    # Abnormal training labels.

    # --
    # Assign labels to normal (0) and abnormals (1)
    nrm_trn_lbl[:] = 0
    nrm_tst_lbl[:] = 0
    abn_trn_lbl[:] = 1
    abn_tst_lbl[:] = 1

    # --
    if manualseed != -1:
        # Random seed.
        # Concatenate the original train and test sets.
        nrm_img = torch.cat((nrm_trn_img, nrm_tst_img), dim=0)
        nrm_lbl = torch.cat((nrm_trn_lbl, nrm_tst_lbl), dim=0)
        abn_img = torch.cat((abn_trn_img, abn_tst_img), dim=0)
        abn_lbl = torch.cat((abn_trn_lbl, abn_tst_lbl), dim=0)

        # Split the normal data into the new train and tests.
        idx = np.arange(len(nrm_lbl))
        np.random.seed(manualseed)
        np.random.shuffle(idx)

        nrm_trn_len = int(len(idx) * 0.80)
        nrm_trn_idx = idx[:nrm_trn_len]
        nrm_tst_idx = idx[nrm_trn_len:]

        nrm_trn_img = nrm_img[nrm_trn_idx]
        nrm_trn_lbl = nrm_lbl[nrm_trn_idx]
        nrm_tst_img = nrm_img[nrm_tst_idx]
        nrm_tst_lbl = nrm_lbl[nrm_tst_idx]

    # Create new anomaly dataset based on the following data structure:
    new_trn_img = nrm_trn_img.clone()
    new_trn_lbl = nrm_trn_lbl.clone()
    new_tst_img = torch.cat((nrm_tst_img, abn_trn_img, abn_tst_img), dim=0)
    new_tst_lbl = torch.cat((nrm_tst_lbl, abn_trn_lbl, abn_tst_lbl), dim=0)

    return new_trn_img, new_trn_lbl, new_tst_img, new_tst_lbl
