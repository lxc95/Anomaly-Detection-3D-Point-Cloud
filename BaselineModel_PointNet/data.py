#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Modified from 
  https://github.com/WangYueFt/dgcnn/blob/master/pytorch/data.py
by
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: data.py
@Time: 2018/10/13 6:21 PM
"""

import os
import sys
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


# def load_data(partition):
#     BASE_DIR = os.path.dirname("/content/")
#     DATA_DIR = os.path.join(BASE_DIR, 'Datasets')
#     all_data = []
#     all_label = []
#     for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
#         f = h5py.File(h5_name)
#         data = f['data'][:].astype('float32')
#         label = f['label'][:].astype('int64')
#         f.close()
#         all_data.append(data)
#         all_label.append(label)
#     all_data = np.concatenate(all_data, axis=0)
#     all_label = np.concatenate(all_label, axis=0)
#     return all_data, all_label

# def load_data_DAIS(partition):
#     # BASE_DIR = os.path.dirname("/content/")
#     # DATA_DIR = os.path.join(BASE_DIR, 'Datasets')
#     BASE_DIR = os.path.dirname("/content/drive/MyDrive/DAISproject/")
#     DATA_DIR = os.path.join(BASE_DIR, 'Datasets')
#     train_data_path = os.path.join(DATA_DIR, 'new_train_data.npy')
#     train_label_path = os.path.join(DATA_DIR, 'train_label.npy')
#     data = np.load(train_data_path)
#     label = np.load(train_label_path)
#     if partition == 'train':
#       data = data[0:190, :, :]
#       label = label[0:190]
#     else:
#       data = data[0:190, :, :]
#       label = label[0:190]
#     all_data = np.float32(data)
#     all_label = np.zeros((len(label), 1))
#     label = label.astype('int64')
#     all_label[:, 0] = label.astype('int64')
#     all_label = all_label.astype('int64')
#     return all_data, all_label

def load_data_DAIS(partition):
    # BASE_DIR = os.path.dirname("/content/")
    # DATA_DIR = os.path.join(BASE_DIR, 'Datasets')
    BASE_DIR = os.path.dirname("C:/Users/xl037/PycharmProjects/DAISproject/")
    DATA_DIR = os.path.join(BASE_DIR, 'Datasets')

    if partition == 'train':
      train_data_path = os.path.join(DATA_DIR, 'X_train_data.npy')
      train_label_path = os.path.join(DATA_DIR, 'y_train_data.npy')
      data = np.load(train_data_path)
      label = np.load(train_label_path)
    else:
      train_data_path = os.path.join(DATA_DIR, 'X_test_data.npy')
      train_label_path = os.path.join(DATA_DIR, 'y_test_data.npy')
      data = np.load(train_data_path)
      label = np.load(train_label_path)
    all_data = np.float32(data)
    all_label = np.zeros((len(label), 1))
    label = label.astype('int64')
    all_label[:, 0] = label.astype('int64')
    all_label = all_label.astype('int64')
    return all_data, all_label


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def pointcloud2graph(pointcloud, all_edges, threshold=0.2):
    N = all_edges.shape[0]
    dist = pointcloud.repeat((N, 1, 1))
    dist = (dist - pointcloud.unsqueeze(1)).norm(dim=2, keepdim=False)
    k = threshold ** 2 / 2
    mask = dist < threshold
    dist = torch.exp(-dist ** 2 / k) * (mask).float()
    edges = all_edges[mask]
    rowsum = dist.sum(dim=1)
    factor = 1. / rowsum[edges[:, 0]].float()
    factor[factor > 1] = 0.
    # adj = torch.sparse.FloatTensor(edges.t(), torch.ones(edges.shape[0], dtype = torch.float) * factor, torch.Size([N,N]))
    # return adj
    return (edges, dist[mask] * factor)  # edges and their values in adjacency matrix


class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_data_DAIS(partition)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
        pointcloud = torch.from_numpy(pointcloud)
        return pointcloud, torch.from_numpy(label)

    def __len__(self):
        return self.data.shape[0]
