# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


from AIPUBuilder.Optimizer.framework import *
from torch.utils.data import Dataset
import numpy as np
import torch


@register_plugin(PluginType.Dataset, '1.0')
class LLama2Dataset(Dataset):

    def __init__(self, data_file=None, label_file=None):
        self.seq_len = 2048

        self.data = []
        self.label = None
        self.consumer = None

        try:
            data = np.load(data_file, mmap_mode='c')
        except ValueError:
            data = np.load(data_file, allow_pickle=True).item()
        keys = list(data.keys())
        for key in keys:
            self.data.append(np.concatenate([data[key], data[key]], axis=0))

        if label_file is not None:
            self.label = []
            try:
                label = np.load(label_file, mmap_mode='c')
            except ValueError:
                label = np.load(label_file, allow_pickle=True).item()
            keys = list(label.keys())
            for key in keys:
                self.label.append(label[key])

    def __len__(self):
        return self.data[0].shape[0]

    def __getitem__(self, idx):
        # input_tensors = [embeded_0,attention_mask_0,position_id_0,history_k_0,history_v_0]
        sample = [[self.data[0][idx], self.data[1][idx], self.data[2][idx], self.data[3][idx], self.data[4][idx]], []]
        if idx == 0:
            # sample[0][3] = np.array([]).astype(np.float32)
            # sample[0][4] = np.array([]).astype(np.float32)
            sample[0][3] = np.zeros_like(sample[0][3])
            sample[0][4] = np.zeros_like(sample[0][4])
        else:
            if self.consumer is not None and isinstance(self.consumer, PyGraph):
                # OPT_ERROR(f"llama2 Dataset has consumer {self.consumer}")
                history_k = self.consumer.output_tensors[1].betensor.detach()
                history_v = self.consumer.output_tensors[2].betensor.detach()
                # real_histor_k =torch.concat([history_k[:, :, :(idx-1), :], history_k[:,:, self.seq_len:self.seq_len+1]])
                sample[0][3] = history_k[0, :, self.seq_len:, :]
                sample[0][4] = history_v[0, :, self.seq_len:, :]
            else:
                OPT_WARN(f"llama2 Dataset has no consumer {self.consumer} when idx == {idx} > 0")
                sample[0][3] = np.zeros_like(sample[0][3])
                sample[0][4] = np.zeros_like(sample[0][4])

        return sample
