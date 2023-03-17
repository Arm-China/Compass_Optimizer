# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import argparse
from string import Template
# from AIPUBuilder.Optimizer.logger import *

op_str = """
from AIPUBuilder.Optimizer.framework import *

@op_register($op_type)
def $op_name(self, *args):
    '''please implementation forward of this op'''
    raise NotImplementedError()\n
@quant_register($op_type)
def $op_quant_name(self, *args):
    '''please implementation quantize of this op'''
    raise NotImplementedError()\n
"""

dataset_str = """

from AIPUBuilder.Optimizer.framework import *
from torch.utils.data import Dataset
import numpy as np

@register_plugin(PluginType.Dataset, $version)
class $dataset_name(Dataset):
    def __init__(self, data_file=None, label_file=None):
        pass

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, idx):
        '''return input data order should be same to the input_tensor in FP32 IR'''
        raise NotImplementedError()
"""

metric_str = """

from AIPUBuilder.Optimizer.framework import *

import torch

@register_plugin(PluginType.Metric, $version)
class $metric_name(OptMetric):
    def __init__(self, *args):
        pass

    def __call__(self, pred, target,*args):
        raise NotImplementedError()

    def compute(self):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def report(self):
        raise NotImplementedError()
"""


def generate_op_template(op_type):
    op_name = op_type.split('.')[1] if '.' in op_type else op_type
    op_quant_name = '_'.join(['quant', op_name.lower()])

    op_template = Template(op_str)
    template = op_template.substitute(op_type=op_type, op_name=op_name, op_quant_name=op_quant_name)
    return template


def generate_dataset_template(dataset_name, version='0.01'):
    op_template = Template(dataset_str)
    template = op_template.substitute(version=version, dataset_name=dataset_name)
    return template


def generate_metric_template(metric_name, version='0.01'):
    op_template = Template(metric_str)
    template = op_template.substitute(version=version, metric_name=metric_name)
    return template


def serialize(path, str):
    with open(path, 'w') as fw:
        fw.write(str)


def get_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument('--op_type', type=str, required=False, help='op_type of op plugin.\n')
    parser.add_argument('--dataset_name', type=str, required=False, help='dataset name of plugin.\n')
    parser.add_argument('--dataset_version', type=str, required=False, help='dataset version of plugin.\n')
    parser.add_argument('--metric_name', type=str, required=False, help='metric name of plugin.\n')
    parser.add_argument('--metric_version', type=str, required=False, help='metric version of plugin.\n')
    parser.add_argument('--plugin_path', type=str, required=False, help='plugin path.\n')
    argv = parser.parse_args()

    return argv


def main():
    argv = get_argv()
    if argv.plugin_path is not None:
        plugin_path = argv.plugin_path
        dpath = os.path.dirname(plugin_path)
        if not os.path.exists(dpath):
            os.makedirs(dpath)
    else:
        pass
    serialize_dict = {}
    if argv.op_type is not None:
        op_type = argv.op_type  # 'OpType.Eltwise'
        plugin_type = 'op'
        op_name = op_type.split('.')[-1].lower()
        file_name = '_'.join(['aipubt', plugin_type, op_name+'.py'])
        path = os.path.join(plugin_path, file_name)

        op_str = generate_op_template(op_type)
        serialize_dict.update({path: op_str})

    if argv.dataset_name is not None:
        dataset_name = argv.dataset_name  # 'NumpyDataset'
        version = argv.dataset_version if argv.dataset_version is not None else '0.01'
        plugin_type = 'dataset'
        file_name = '_'.join(['aipubt', plugin_type, dataset_name.lower()+'.py'])
        dataset_str = generate_dataset_template(dataset_name, version)
        path = os.path.join(plugin_path, file_name)
        serialize_dict.update({path: dataset_str})

    if argv.metric_name is not None:
        metric_name = argv.metric_name  # 'TopKMetric'
        version = argv.metric_version if argv.metric_version is not None else '0.01'
        plugin_type = 'metric'
        file_name = '_'.join(['aipubt', plugin_type, metric_name.lower()+'.py'])
        metric_str = generate_metric_template(metric_name, version)
        path = os.path.join(plugin_path, file_name)
        serialize_dict.update({path: metric_str})

    for path, temp in serialize_dict.items():
        serialize(path, temp)


if __name__ == "__main__":
    main()
