# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.logger import OPT_INFO

from AIPUBuilder.Optimizer.framework import *

from torch.utils.data import Dataset
import numpy as np


@register_plugin(PluginType.Dataset, '1.0')
class NumpyMultiInputDataset(Dataset):
    """
    This NumpyMultiInputDataset plugin is mainly used for the models with multi-inputs.
    The order of input datas in npy file has the same with the order of input tensors in CompassIR.

    Assume that all preprocesses of data have been done before save to npy file if the CompassIR doesnot have preprocess part.
    """

    # when used as calibration dataset, label_file can be omitted.
    def __init__(self, data_file, label_file=None):
        '''
        :param data_file: a dict format in npy file: {key0: ndarray, key1: ndarray, ...}.
        :param label_file: a dict format in npy file.
        '''
        self.data = []
        self.label = None

        try:
            data = np.load(data_file, mmap_mode='c')
        except ValueError:
            data = np.load(data_file, allow_pickle=True).item()
        keys = list(data.keys())
        for key in keys:
            self.data.append(data[key])

        if label_file is not None:
            self.label = []
            try:
                label = np.load(label_file, mmap_mode='c')
            except ValueError:
                label = np.load(label_file, allow_pickle=True).item()
            keys = list(label.keys())
            for key in keys:
                self.label.append(label[key])

    def __len__(self):
        return self.data[0].shape[0]

    def __getitem__(self, idx):
        sample = [[], []]

        if 1 == len(self.data):
            sample[0] = self.data[0][idx]
        else:
            sample[0] = []
            for d in self.data:
                sample[0].append(d[idx])

        if self.label is not None:
            input_batch = 1
            output_batch = 1
            label_idx = idx // input_batch
            for l in self.label:
                l_per_batch = l[label_idx *
                                output_batch: (label_idx + 1) * output_batch]
                sample[1].append(l_per_batch)

        return sample
