# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.logger import *
from torch.utils.data import Dataset
import numpy as np


@register_plugin(PluginType.Dataset, '1.0')
class NumpyNCHW2NHWCDataset(Dataset):
    """
    This NumpyNCHW2NHWCDataset plugin is used for the dataset has NCHW data format, but the CompassIR needs NHWC data format.
    This plugin automatically transfers the NCHW data format to NHWC data format, which meets the CompassIR requirement.

    Assume that all preprocesses of data have been done before save to npy file if the CompassIR doesnot have preprocess part.
    """

    def __init__(self, data_file, label_file=None):
        '''
        :param data_file: ndarray in npy file.
        :param label_file: ndarray in npy file.
        '''
        self.data = None
        self.label = None
        try:
            self.data = np.load(data_file, mmap_mode='c')
            self.data = np.transpose(self.data, [0, 2, 3, 1])
        except Exception as e:
            OPT_FATAL('the data of NumpyNCHW2NHWCDataset plugin should be Numpy.ndarray and allow_pickle=False.')
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
