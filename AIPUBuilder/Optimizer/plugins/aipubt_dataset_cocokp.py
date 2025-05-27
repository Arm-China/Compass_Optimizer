# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

import numpy as np
import torch

from AIPUBuilder.Optimizer.framework import *

from torch.utils.data import Dataset
from collections import defaultdict


@register_plugin(PluginType.Dataset, '1.0')
class CocokpDataset(Dataset):
    """
    This CocoDataset plugin is used for yolov4_onnx/yolov4_tflite/fasterrcnn_tensorflow models in Optimizer.
    """

    def __init__(self, data_file=None, label_file=None):
        """
        :param data_file: a .npy file
        :param label_file: a dict format in .npy file and format is {
                        are: list box area,
                        bbox: list of boxes,
                        keypoint: list of 17*3}
        """
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
            sample[1] = raw_label
        return sample

    @staticmethod
    def collate_fn(batch):
        batch_label = []
        batch_data = None
        for batch_idx in range(len(batch)):
            els_sample = batch[batch_idx][0]
            single_data = torch.unsqueeze(torch.tensor(els_sample), 0)
            batch_data = single_data if batch_idx == 0 else torch.cat(
                (batch_data, single_data), 0)

        for idx, sample in enumerate(batch):
            if not sample[1]:
                continue
            label = {}
            for k, v in sample[1].items():
                label[k] = torch.tensor(v)
            batch_label.append(label)
        return batch_data, batch_label
