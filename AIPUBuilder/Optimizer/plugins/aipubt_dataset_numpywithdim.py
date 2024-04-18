# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.logger import *
from torch.utils.data import Dataset
import numpy as np
import torch


@register_plugin(PluginType.Dataset, '1.0')
class NumpyWithDimDataset(Dataset):
    """
    This NumpyWithDimDataset plugin is used for batch dimition is not in default 0, so customer can set the data_batch_dim and
    label_batch_dim to instruct the dataloader to get the data and label in setted batch_dim.
    This plugin customizes the collate_fn function for pytorch dataloader.

    Assume that all preprocesses of data have been done before save to npy file if the CompassIR doesnot have preprocess part.
    """

    data_batch_dim = 0
    label_batch_dim = 0
    # when used as calibration dataset, label_file can be omitted.

    def __init__(self, data_file, label_file=None):
        '''
        :param data_file: ndarray in npy file.
        :param label_file: ndarray in npy file.
        '''
        self.data = None
        self.label = None

        try:
            self.data = np.load(data_file, mmap_mode='c')
        except Exception as e:
            OPT_FATAL(
                'the data of NumpyWithDimDataset plugin should be Numpy.ndarray and allow_pickle=False.')
        if label_file is not None:
            try:
                self.label = np.load(label_file, mmap_mode='c')
            except ValueError:
                self.label = np.load(label_file, allow_pickle=True)

    def __len__(self):
        # return len(self.data)
        return self.data.shape[NumpyWithDimDataset.data_batch_dim]

    def __getitem__(self, idx):
        sample = [[self.data.take(
            indices=idx, axis=NumpyWithDimDataset.data_batch_dim)], float("-inf")]
        if self.label is not None:
            sample[1] = self.label.take(
                indices=idx, axis=NumpyWithDimDataset.label_batch_dim)
        return sample

    @staticmethod
    def collate_fn(batch):
        batch_data_all = []
        batch_label = []
        input_data_num = len(batch[0][0])
        for sample_idx in range(input_data_num):
            batch_data = None
            for batch_idx in range(len(batch)):
                els_sample = batch[batch_idx][sample_idx][0]
                single_data = torch.unsqueeze(torch.tensor(
                    els_sample), NumpyWithDimDataset.data_batch_dim)
                if batch_idx == 0:
                    batch_data = single_data
                else:
                    batch_data = torch.cat(
                        (batch_data, single_data), NumpyWithDimDataset.data_batch_dim)
            batch_data_all.append(batch_data)

        for idx, sample in enumerate(batch):
            if isinstance(sample[1], (int, float)):
                batch_label.append(sample[1])
            elif isinstance(sample[1], dict):
                OPT_WARN(
                    'currently NumpyWithDimDataset does not that supprot label type is dict')
            else:
                single_label = torch.unsqueeze(torch.tensor(
                    sample[1]), NumpyWithDimDataset.label_batch_dim)
                if idx == 0:
                    batch_label = single_label
                else:
                    batch_label = torch.cat(
                        (batch_label, single_label), NumpyWithDimDataset.label_batch_dim)
        batch_label = torch.tensor(batch_label) if isinstance(
            batch[0][1], (int, float)) else batch_label
        return batch_data_all, batch_label
