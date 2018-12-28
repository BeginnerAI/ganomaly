#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : lib/data.py
# Author            : Tianming Jiang <djtimy920@gmail.com>
# Date              : 19.12.2018
# Last Modified Date: 28.12.2018
# Last Modified By  : Tianming Jiang <djtimy920@gmail.com>
"""
LOAD DATA from file.
可以分为3个主要的步骤：
1）利用torchvision中的MNIST模块下载MINST数据，并对数据进行transform；
2）按照配置的异常值将MNIST数据划分为正常样本和异常样本，训练集包含80%的正常样本，测试集包含20%的正常样本和全部的异常样本；
3）torch.utils.data.DataLoader将数据转化为PyTorchm模型的输入格式
"""

##
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

    if opt.dataset in ['disk_1d']:
        splits = ['train', 'test']
        drop_last_batch = {'train': True, 'test': False}
        shuffle = {'train': True, 'test': True}

        # transforms.Compose就是将transforms组合在一起
        transform = transforms.Compose(
            [
                transforms.Scale(opt.isize),
                transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
            ]
        )
        from sklearn import datasets
        import numpy as np
        # how to generate imbalanced dataset
        data, labels = datasets.make_classification(n_samples=1000, n_features=3, n_redundant=0)

        good_index = np.where(labels==1)
        nrm_trn_len = int(len(good_index) * 0.80)
        train_good_index = good_index[:nrm_trn_idx]
        test_good_index = good_index[nrm_trn_idx:]

        fail_index = np.where(labels!=1)

        dataset = {}
        dataset['train'] = {
                'train_data': data[train_good_index],
                'train_lables': labels[train_good_index]
                }

        dataset['test'] = {
                'test_data': np.concatenate((data[test_good_index], data[fail_index])),
                'test_labels': np.concatenate((labels[test_good_index], labels[fail_index]))
                }

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

        good_sample_dir = '/home/jtm/data/cnn_disk/intermediate_result/data_2017_with_change_rate/baidu/good_sample/00443'
        failed_sample_dir = '/home/jtm/data/cnn_disk/intermediate_result/data_2017_with_change_rate/baidu/failed_sample/00052'
        good_sample = LoadDataset(good_sample_dir, 'good', transform)
        failed_sample = LoadDataset(failed_sample_dir, 'failed', transform)

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
        np.random.shuffle(idx)

        good_train_lens = int(len(idx) * 0.80)
        good_train_idx = idx[:good_train_lens]
        good_test_idx = idx[good_train_lens:]
        
        good_train_sample = Subset(good_sample, good_train_idx)
        good_test_sample = Subset(good_sample, good_test_idx)

        train_sample = good_train_sample
        test_sample = ConcatDataset((good_test_sample, failed_sample))

    return train_sample, test_sample


class LoadDataset(Dataset):
    def __init__(self, figure_dir, label=None, transform=None):
        self.file_name_list = glob.glob('{0}/*'.format(figure_dir))
        self.label = label
        self.transform = transform

        assert self.label == 'good' or self.label == 'failed', 'label must be good or failed!'
        if self.label == 'good':
            self.label = 1
        else:
            self.label = 0

        self.count = len(self.file_name_list)
        self.data_x = np.empty((self.count,1,12,12), dtype='float32')
        self.data_y = []

        i = 0
        for file_name in self.file_name_list:
            one_figure = pd.read_csv(file_name, header=None)
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
