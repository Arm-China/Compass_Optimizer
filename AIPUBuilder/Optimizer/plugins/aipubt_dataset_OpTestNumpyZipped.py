# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.utils import *
from torch.utils.data import Dataset
import numpy as np

"""
For OPT OP Test, the format of the data.npz(genrate by tester):
data = {
    '0': numpy.array(.....) input of 1st tensor
    '1': numpy.array(.....) input of 2nd tensor
    ......
    'shape_0' = [2, 3, 4] shape from ir
    'shape_1' = [3, 3, 3, 3] shape from ir
}

label = {
    '0': numpy.array(.....) input of 1st tensor
    '1': numpy.array(.....) input of 2nd tensor
    ......
    'shape_0' = [2, 3, 4] shape from ir
    'shape_1' = [3, 3, 3, 3] shape from ir
}
"""


@register_plugin(PluginType.Dataset, '0.01')
class OpTestNumpyZippedDataset(Dataset):
    def __init__(self, data_file, label_file=None):
        self.data = []
        self.intput_shapes = []
        self.label = None
        self.output_shapes = []
        try:
            data = np.load(data_file, mmap_mode='c')
        except ValueError:
            data = np.load(data_file, allow_pickle=True)
        keys = list(data.keys())
        sorted(keys)
        for i, key in enumerate(keys):
            if key.startswith('shape'):
                self.intput_shapes.append(data[key])
            else:
                OPT_INFO(f"NumpyZippedDataset load data:{i} name:{key} shape:{data[key].shape}")
                self.data.append(data[key])

        if label_file is not None:
            try:
                label = np.load(label_file, mmap_mode='c')
            except ValueError:
                label = np.load(label_file, allow_pickle=True)
            keys = list(label.keys())
            self.label = []
            sorted(keys)
            for i, key in enumerate(keys):
                if key.startswith('shape'):
                    self.output_shapes.append(label[key])
                else:
                    OPT_INFO(f"NumpyZippedDataset load label:{i} name:{key} shape:{label[key].shape}")
                    self.label.append(label[key])

        OPT_INFO(f"NumpyZippedDataset expect using shape:{self.intput_shapes} -> {self.output_shapes} per batch")

    def __len__(self):
        return self.data[0].shape[0]

    def __getitem__(self, idx):
        sample = [[], []]

        for d in self.data:
            npd = d[idx]
            sample[0].append(torch.tensor(npd.astype(torch_type2nptype(nptype2torch_type(npd.dtype.type)))))

        if self.label is not None:
            for index, l in enumerate(self.label):
                if not self.output_shapes[index].size:
                    l_per_batch = l
                else:
                    l_per_batch = l[idx]
                sample[1].append(torch.tensor(l_per_batch.astype(
                    torch_type2nptype(nptype2torch_type(l_per_batch.dtype.type)))))

        return sample
