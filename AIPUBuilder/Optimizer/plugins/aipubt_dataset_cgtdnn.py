# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from AIPUBuilder.Optimizer.logger import OPT_INFO

from AIPUBuilder.Optimizer.framework import *

from torch.utils.data import Dataset
import numpy as np


@register_plugin(PluginType.Dataset, '1.0')
class cgtdnnDataset(Dataset):
    """
    This cgtdnnDataset plugin is mainly used for cgtdnn3_tensorflow model in Optimizer.
    """
    # when used as calibration dataset, label_file can be omitted.

    def __init__(self, data_file, label_file=None):
        """
        :param data_file:  a dict format which the number of items are same to the inputs number in .npy
        :param label_file: a dict format
        """
        self.data = []
        self.label = []
        try:
            data = np.load(data_file, mmap_mode='c')
        except ValueError:
            data = np.load(data_file, allow_pickle=True).item()
        keys = list(data.keys())
        sorted(keys)
        for i, key in enumerate(keys):
            self.data.append(data[key])

        if label_file is not None:
            try:
                label = np.load(label_file, mmap_mode='c')
            except ValueError:
                label = np.load(label_file, allow_pickle=True).item()
            keys = list(label.keys())
            self.label = []
            sorted(keys)
            for i, key in enumerate(keys):
                self.label.append(label[key])

    def __len__(self):
        return self.data[0].shape[0]

    def __getitem__(self, idx):
        sample = [[], []]
        if 1 == len(self.data):
            sample[0] = np.expand_dims(self.data[0][idx], 0)
        else:
            sample[0] = []
            for d in self.data:
                sample[0].append(np.expand_dims(d[idx], 0))

        if self.label is not None:
            input_batch = 1
            output_batch = 1
            label_idx = idx // input_batch
            for l in self.label:
                l_per_batch = l[label_idx *
                                output_batch: (label_idx + 1) * output_batch]
                sample[1].append(l_per_batch)
        return sample
