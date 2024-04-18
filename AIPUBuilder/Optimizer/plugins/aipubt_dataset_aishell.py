# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


from AIPUBuilder.Optimizer.framework import *

from torch.utils.data import Dataset
import numpy as np


@register_plugin(PluginType.Dataset, '1.0')
class AiShellDataset(Dataset):
    """
    This AiShellDataset plugin is mainly used for wavenet_tensorflow model in Optimizer.
    This plugin will pad the data to the 390 length, and pad the labels to the maximum length of all labels.

    Aishell is an open-source Chinese Mandarin speech corpus published by Beijing Shell Shell Technology Co.,Ltd.
    """

    def __init__(self, data_file=None, label_file=None):
        '''
        :param data_file: list(ndarray), len(ndarray.shape) = 2
        :param label_file: list(list)
        '''
        self.data = np.load(data_file, allow_pickle=True)
        self.label = None
        if label_file is not None:
            self.label = np.load(label_file, allow_pickle=True)
            self.label, self.act_label_len = self.padding_label(self.label)

        self.data, self.len_list = self.padding_data(self.data)

    def padding_data(self, data, padded_len=390):
        def padding(pick_data, padded_len):
            origin_len = pick_data.shape[0]
            act_data = np.zeros([padded_len, pick_data.shape[-1]])
            act_len = min(origin_len, padded_len)
            act_data[:act_len, :] = pick_data[:act_len, :]
            return act_data, origin_len

        batch_size = len(data)
        batch_padded_data = np.zeros(
            [batch_size, padded_len, data[0].shape[1]])
        batch_origin_len = []
        for i, d in enumerate(data):
            padded_data, origin_len = padding(d, padded_len)
            batch_padded_data[i] = padded_data
            batch_origin_len.append(origin_len)
        return batch_padded_data, batch_origin_len

    def padding_label(self, label):
        batch_size = len(label)
        act_label_len = [len(l) for l in label]
        max_label_len = max(act_label_len)
        batch_padded_label = np.zeros([batch_size, max_label_len])
        for i, d in enumerate(label):
            batch_padded_label[i][:act_label_len[i]] = label[i]
        return batch_padded_label, act_label_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pick_label = []
        pick_label_len = []
        if self.label is not None:
            pick_label = self.label[idx]
            pick_label_len = np.array(self.act_label_len[idx], dtype=np.int32)
        sample = [[self.data[idx], np.array(self.len_list[idx], dtype=np.int32)], [
            pick_label, pick_label_len]]
        return sample
