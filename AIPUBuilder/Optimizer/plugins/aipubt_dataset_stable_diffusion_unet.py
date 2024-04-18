# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

import numpy as np
from torch.utils.data import Dataset, DataLoader
from AIPUBuilder.Optimizer.framework import *


@register_plugin(PluginType.Dataset, '0.01')
class StableDiffusionUNetDataset(Dataset):
    def __init__(self, data_file, label_file=None):
        dataset = np.load(data_file, allow_pickle=True).item()
        keys = list(dataset.keys())
        self.s_dataset = dataset[keys[0]]
        self.t_dataset = dataset[keys[1]]
        self.h_dataset = dataset[keys[2]]
        '''
        self.s_dataset = dataset['input1']
        self.t_dataset = dataset['input2']
        self.h_dataset = dataset['input3']
        '''

    def __getitem__(self, idx):
        sample = [[self.s_dataset[idx], self.t_dataset[idx], self.h_dataset[idx]], float("-inf")]

        return sample

    def __len__(self):
        return len(self.s_dataset)
