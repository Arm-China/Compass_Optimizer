# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.


import numpy as np

from AIPUBuilder.Optimizer.framework import *

from torch.utils.data import Dataset
from collections import defaultdict


@register_plugin(PluginType.Dataset, '1.0')
class tusimpleDataset(Dataset):
    def __init__(self, data_file=None, label_file=None):
        self.data = np.load(data_file, mmap_mode='c').astype(np.float32)
        self.label = None
        if label_file is not None:
            self.label = np.load(label_file, allow_pickle=True).item()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_data = self.data[idx]
        sample = [image_data, {}]
        if self.label is not None:
            raw_label = self.label[idx]
            image_name = idx

            sample[1].update({
                'image_name': np.array(image_name),
                'lanes': np.array(raw_label[0]),
                'h_samples': np.array(raw_label[1]),

            })
        return sample
