# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.


from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.logger import *
from torch.utils.data import Dataset
import numpy as np


@register_plugin(PluginType.Dataset, '2.0')
class MyNumpyNHWCRGB2NCBGRHWDataset(Dataset):
    def __init__(self, data_file, label_file=None):
        '''
        :param data_file: ndarray in npy file.
        :param label_file: ndarray in npy file.
        '''
        OPT_INFO('Customized dataset plugin is enabled.')
        self.data = None
        self.label = None
        try:
            self.data = np.load(data_file, mmap_mode='c')
            self.data = np.flip(self.data, -1).copy()
            self.data = np.transpose(self.data, [0, 3, 1, 2])
        except Exception as e:
            OPT_FATAL('the data of MyNumpyNHWCRGB2NCBGRHWDataset plugin should be Numpy.ndarray and allow_pickle=False.')
        if label_file is not None:
            try:
                self.label = np.load(label_file, mmap_mode='c')
            except ValueError:
                self.label = np.load(label_file, allow_pickle=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = [[self.data[idx]], float("-inf")]
        if self.label is not None:
            sample[1] = self.label[idx]
        return sample
