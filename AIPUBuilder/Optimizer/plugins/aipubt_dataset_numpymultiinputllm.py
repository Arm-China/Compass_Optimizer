# Copyright © 2025 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from AIPUBuilder.Optimizer.logger import OPT_INFO

from AIPUBuilder.Optimizer.framework import *

from torch.utils.data import Dataset
import numpy as np
import os


@register_plugin(PluginType.Dataset, '1.0')
class NumpyMultiInputDatasetllm(Dataset):
    """
    This NumpyMultiInputDatasetllm plugin is mainly used for the models with multi-inputs,but sequence len is changed during running llm models.

    The order of input datas in npy file has the same with the order of input tensors in CompassIR.

    Assume that all preprocesses of data have been done before save to npy file if the CompassIR doesnot have preprocess part.
    """

    # when used as calibration dataset, label_file can be omitted.
    def __init__(self, data_file, label_file=None):
        '''
        :param data_file: a dict format in npy file: {key0: input0_file_name, key1: input1_file_name, ...}.
        :
        '''
        self.data = []
        self.label = None
        self.directory = os.path.dirname(data_file)
        try:
            data = np.load(data_file)
        except ValueError:
            data = np.load(data_file, allow_pickle=True).item()
        # keys = list(data.keys())
        for key in range(len(data)):
            self.data.append(data[key].decode('utf-8'))

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
        return len(self.data)

    def __getitem__(self, idx):
        sample = [[], []]
        datafile = os.path.join(self.directory, self.data[idx]+str(idx)+'.npz')
        try:
            data = np.load(datafile, mmap_mode='c')
        except ValueError:
            data = np.load(datafile, allow_pickle=True).item()
        keys = list(data.keys())
        sample[0] = []
        for key in keys:
            sample[0].append(data[key][0])

        return sample
