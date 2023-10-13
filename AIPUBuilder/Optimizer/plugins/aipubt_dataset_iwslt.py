# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.


from AIPUBuilder.Optimizer.framework import *

from torch.utils.data import Dataset
import numpy as np


@register_plugin(PluginType.Dataset, '1.0')
class IWSLTDataset(Dataset):
    """
    This IWSLTDataset plugin is used for the transformer_tensorflow model in Optimizer.
    """

    # when used as calibration dataset, label_file can be omitted.
    def __init__(self, data_file, label_file=None):
        self.data = None
        self.label = None
        try:
            self.data = np.load(data_file, mmap_mode='c')
        except ValueError:
            self.data = np.load(data_file, allow_pickle=True)
        if label_file is not None:
            self.label = np.load(label_file, allow_pickle=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = [self.data[idx], float("-inf")]
        if self.label is not None:
            sample[1] = self.label[idx]
        return sample
