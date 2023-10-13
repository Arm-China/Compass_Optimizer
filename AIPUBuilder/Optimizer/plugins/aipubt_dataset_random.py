# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.


from AIPUBuilder.Optimizer.framework import *

from torch.utils.data import Dataset


@register_plugin(PluginType.Dataset, '1.0')
class RandomDataset(Dataset):
    """
    This RandomDataset plugin is based on the input data shape and label shape to generate the random data/label as dataset/labelset.

    Assume that all preprocesses of data have been done before save to npy file if the CompassIR doesnot have preprocess part.
    """

    def __init__(self, sample_shape, label_shape=(), num_sample=1, seed=None):
        import numpy as np
        if seed:
            np.random.seed(seed)
        self.num_sample = num_sample
        self.sample_shape = sample_shape
        self.label_shape = label_shape
        self.data = []
        self.label = []
        for _ in range(num_sample):
            self.data.append(np.random.randn(*tuple(self.sample_shape)).astype(np.float32))
            self.label.append(np.random.randn(*tuple(self.label_shape)).astype(np.float32))

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)
