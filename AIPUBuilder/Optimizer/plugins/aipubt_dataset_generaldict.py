# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.


from AIPUBuilder.Optimizer.framework import *

from torch.utils.data import Dataset
import numpy as np


@register_plugin(PluginType.Dataset, '1.0')
class GeneralDictDataset(Dataset):
    """
    This GeneralDictDataset plugin is used for the data and label are both dict format in npy file.
    """

    def __init__(self, data_file=None, label_file=None):
        '''
        :param data_file: a dict format in npy file
        :param label_file: a dict format in npy file
        '''
        self.data = np.load(data_file, allow_pickle=True).tolist()
        self.label = None
        if label_file is not None:
            self.label = np.load(label_file, allow_pickle=True).tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Assume that all preprocesses have been done before save to npy file.
        If the graph has single input tensor,
        the data part sample[0] will be passed to the input node as is,
        If the graph has multiple input tensors,
        the data part sample[0][i] should be consistent with input_tensors[i] in IR.
        If the graph has multiple output tensors,
        the label part sample[1][i] should be consistent with output_tensors[i] in IR.
        """

        inps = self.data[idx]
        local_inp = []
        for k, v in inps.items():
            if v.shape[0] == 1:
                v = np.squeeze(v, 0)
            local_inp.append(v)
        if len(self.data[idx]) == 1:
            local_inp = local_inp[0]
        sample = [local_inp, float("-inf")]
        if self.label is not None:
            oups = self.label[idx]
            local_oup = []
            for k, v in oups.items():
                if v.shape[0] == 1:
                    v = np.squeeze(v, 0)
                local_oup.append(v)
            if len(self.label[idx]) == 1:
                local_oup = local_oup[0]
            sample[1] = local_oup
        return sample
