# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.

import numpy as np
import torch

from AIPUBuilder.Optimizer.framework import *

from torch.utils.data import Dataset
from collections import defaultdict


@register_plugin(PluginType.Dataset, '1.0')
class VOCNCHWDataset(Dataset):
    """
    This VOCNCHWDataset plugin is used for yolact_onnx/vgg_ssd_caffe/mobilenet_v2_ssd_onnx/mobilenet_v2_ssd_caffe models in Optimizer.
    This plugin would transfer the NHWC data format to NCHW data format, which meets the CompassIR requirement.
    This plugin customizes the collate_fn function for pytorch dataloader.

    Assume that all preprocesses of data have been done before save to npy file if the CompassIR doesnot have preprocess part.
    """

    def __init__(self, data_file=None, label_file=None):
        '''
        :param data_file: ndarray in npy file.
        :param label_file: a dict format in .npy file and format is {image_name_idx: [label_index, ymin, xmin, ymax, xmax, confidence]}
                        image_name_idx: int,
                        label_index: list,
                        ymin: list, the len(ymin) == len(label_index)
                        xmin: list, the len(xmin) == len(label_index)
                        ymax: list, the len(ymax) == len(label_index)
                        xmax: list, the len(xmax) == len(label_index)
                        confidence: list, the len(confidence) == len(label_index)
        '''
        self.data = np.load(data_file, mmap_mode='c').astype(np.float32)
        self.data = np.transpose(self.data, [0, 3, 1, 2])
        self.label = None
        if label_file is not None:
            self.label = np.load(label_file, allow_pickle=True).item()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = [self.data[idx], {}]
        if self.label is not None:
            raw_label = self.label[idx]
            image_name = idx
            label_idx = raw_label[0]
            confidence = raw_label[-1]
            bbox = np.stack(
                (raw_label[1], raw_label[2], raw_label[3], raw_label[4]), -1)

            sample[1].update({
                'image_name': np.array(image_name),
                'label_index': np.array(label_idx),
                'bbox': bbox,
            })
        return sample

    @staticmethod
    def collate_fn(batch):
        batch_label = {}
        batch_data = None
        for batch_idx in range(len(batch)):
            els_sample = batch[batch_idx][0]
            single_data = torch.unsqueeze(torch.tensor(els_sample), 0)
            batch_data = single_data if batch_idx == 0 else torch.cat(
                (batch_data, single_data), 0)

        image_list = []
        label_list = []
        bbox_list = []
        ori_img_shape_list = []
        for idx, sample in enumerate(batch):
            if not sample[1]:
                continue
            image_name = sample[1]['image_name']
            label_index = sample[1]['label_index']
            bbox = sample[1]['bbox']
            image_list.append(torch.tensor(image_name))
            label_list.append(torch.tensor(label_index))
            bbox_list.append(torch.tensor(bbox))
        batch_label.update({'image_name': image_list})
        batch_label.update({'label_index': label_list})
        batch_label.update({'bbox': bbox_list})
        return batch_data, batch_label
