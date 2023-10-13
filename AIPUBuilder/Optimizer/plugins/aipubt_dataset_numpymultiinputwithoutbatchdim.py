# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.logger import OPT_INFO
from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.utils import *
from torch.utils.data import Dataset
import numpy as np


@register_plugin(PluginType.Dataset, '1.0')
class NumpyMultiInputWithoutBatchDimDataset(Dataset):
    """
    This NumpyMultiInputWithoutBatchDimDataset plugin is mainly used for the models with multi-inputs without batch dim.
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
        sorted(keys)
        for key in keys:
            if not key.startswith('shape'):
                self.data.append(data[key])

        if label_file is not None:
            self.label = []
            try:
                label = np.load(label_file, mmap_mode='c')
            except ValueError:
                label = np.load(label_file, allow_pickle=True).item()
            keys = list(label.keys())
            sorted(keys)
            for key in keys:
                if not key.startswith('shape'):
                    self.label.append(label[key])

        self.add_batch_dim()

    def __len__(self):
        return self.data[0].shape[0]

    def __getitem__(self, idx):
        sample = [[], []]

        if 1 == len(self.data):
            sample[0] = torch.tensor(self.data[0][idx].astype(
                torch_type2nptype(nptype2torch_type(self.data[0][idx].dtype.type))))
        else:
            sample[0] = []
            for d in self.data:
                sample[0].append(torch.tensor(d[idx].astype(torch_type2nptype(nptype2torch_type(d[idx].dtype.type)))))

        if self.label is not None:
            for l in self.label:
                sample[1].append(torch.tensor(l.astype(torch_type2nptype(nptype2torch_type(l.dtype.type)))))

        return sample

    def add_batch_dim(self):
        for idx, data in enumerate(self.data):
            self.data[idx] = np.expand_dims(data, axis=0)
