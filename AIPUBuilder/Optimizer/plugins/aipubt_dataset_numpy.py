# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.


from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.logger import *
from torch.utils.data import Dataset
import numpy as np
import torch


@register_plugin(PluginType.Dataset, '1.0')
class NumpyDataset(Dataset):
    """
    This NumpyDataset plugin is mainly used for image classification domain models which have one input.
    The data in npy file has the same datalayout with the input datalayout in model.

    Assume that all preprocesses of data have been done before save to npy file if the CompassIR doesnot have preprocess part.
    """
    # when used as calibration dataset, label_file can be omitted.

    def __init__(self, data_file, label_file=None):
        '''
        :param data_file: ndarray in npy file.
        :param label_file: ndarray in npy file.
        '''
        self.data = None
        self.label = None

        try:
            self.data = np.load(data_file, mmap_mode='c')
        except Exception as e:
            OPT_FATAL(
                'the data of NumpyDataset plugin should be Numpy.ndarray and allow_pickle=False.')
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
