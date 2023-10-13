# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.


from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.logger import *
from torch.utils.data import Dataset
import numpy as np
import torch


@register_plugin(PluginType.Dataset, '1.0')
class SphereFaceLFWDataset(Dataset):
    """
    This SphereFaceLFWDataset plugin is mainly used for sphereface_caffe model.
    The data in npy file has the same datalayout with the input datalayout in model.

    Assume that all preprocesses of data have been done before save to npy file if the CompassIR doesnot have preprocess part.
    """

    def __init__(self, data_file, label_file=None):
        '''
        :param data_file: ndarray in npy file.
        :param label_file: ndarray in npy file.
        '''
        self.data = None
        self.label = []

        self.data = np.load(data_file, mmap_mode='c')
        if label_file is not None:
            try:
                label = np.load(label_file, mmap_mode='c')
            except ValueError:
                label = np.load(label_file, allow_pickle=True).item()
            keys = list(label.keys())
            for key in keys:
                self.label.append(label[key])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = [[self.data[idx]], float("-inf")]
        if len(self.label) != 0:
            sample[1] = self.label[idx]
        return sample
