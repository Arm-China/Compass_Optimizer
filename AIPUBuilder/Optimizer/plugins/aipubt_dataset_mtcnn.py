# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from AIPUBuilder.Optimizer.framework import *

from torch.utils.data import Dataset
import numpy as np
import os


@register_plugin(PluginType.Dataset, '1.0')
class MTCNNDataset(Dataset):
    """
    This MTCNNDataset plugin is mainly used for mtcnn_caffe model in Optimizer.
    """

    def __init__(self, data_file, label_file=None):
        '''
        :param data_file: ndarray in npy file or a valid path which including several data npy files.
        :param label_file: ndarray in npy file or a valid path which including several label npy files.
        '''
        self.single_npy_file = False
        self.data = None
        self.label = None
        self.label_list = None
        if data_file.endswith(".npy"):
            self.single_npy_file = True
            self.data = np.load(data_file, mmap_mode='c').astype(np.float32)
            if label_file is not None:
                try:
                    self.label = np.load(label_file, mmap_mode='c')
                except ValueError:
                    self.label = np.load(label_file, allow_pickle=True)
            self.data_len = len(self.data)
        else:
            file_name_list = [c for c in os.listdir(data_file)]
            self.feature_list = [os.path.join(data_file, c) for c in file_name_list]
            if label_file is not None:
                self.label_list = [os.path.join(label_file, c) for c in file_name_list]
            self.data_len = len(self.feature_list)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        if self.single_npy_file:
            sample = [self.data[idx], float("-inf")]
            if self.label is not None:
                sample[1] = self.label[idx]
            return sample
        sample = [np.load(self.feature_list[idx], allow_pickle=True)[0], float("-inf")]
        if self.label_list is not None:
            sample[1] = np.load(self.label_list[idx], allow_pickle=True).item()
        return sample
