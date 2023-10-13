# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.ops.decodebox import decodebox_ssd
from AIPUBuilder.Optimizer.ops.nms import Nms
from AIPUBuilder.Optimizer.logger import OPT_ERROR
from AIPUBuilder.Optimizer.framework import *


'''
this op only implements the float forward for called by metric plugin.
'''


class SSDPostProcess(object):
    db_default_params = {
        'layer_bottom_shape': [[1, 2034, 91], [1, 2034, 4]],
        'layer_top_shape': [[1, 5000, 4], [1, 5000], [1, 1], [1, 5000], [1, 5000]],
        'class_num': 90,
        'feature_map': [[19, 19], [10, 10], [5, 5], [3, 3], [2, 2], [1, 1]],
        'max_box_num': 5000,
        'score_threshold': 0.3,
        'variance': [10., 10., 5., 5.],
        'image_height': 320,
        'image_width': 320,

    }
    nms_default_params = {
        'layer_bottom_shape': [[1, 5000, 4], [1, 5000], [1, 1], [1, 5000]],
        'layer_top_shape': [[1, 1000, 4], [1, 5000], [1, 1000], [1, 1000]],
        'center_point_box': 0,
        'image_height': 320,
        'image_width': 320,
        'iou_threshold': 0.6,
        'method': 'GAUSSIAN',
        'max_output_size': 100,
        # if method == GAUSSIAN, then optimizer nms needs the parameter of score_threshold
        'score_threshold': 0.0,
    }

    def __init__(self, decodebox_params=None, nms_params=None):
        if 'anchor' not in decodebox_params:
            OPT_ERROR(f"please provide the anchor data in decodebox_params")

        self.db_constants = {'weights': decodebox_params['anchor']}

        self.db_bottom_shape = self.get_params(decodebox_params, 'layer_bottom_shape', SSDPostProcess.db_default_params)
        self.db_top_shape = self.get_params(decodebox_params, 'layer_top_shape', SSDPostProcess.db_default_params)

        self.db_params = {
            'class_num': self.get_params(decodebox_params, 'class_num', SSDPostProcess.db_default_params),
            'feature_map': self.get_params(decodebox_params, 'feature_map', SSDPostProcess.db_default_params),
            'max_box_num': self.get_params(decodebox_params, 'max_box_num', SSDPostProcess.db_default_params),
            'score_threshold': self.get_params(decodebox_params, 'score_threshold', SSDPostProcess.db_default_params),
            'image_height': self.get_params(decodebox_params, 'image_height', SSDPostProcess.db_default_params),
            'image_width': self.get_params(decodebox_params, 'image_width', SSDPostProcess.db_default_params),
        }
        self.nms_bottom_shape = self.get_params(nms_params, 'layer_bottom_shape', SSDPostProcess.nms_default_params)
        self.nms_top_shape = self.get_params(nms_params, 'layer_top_shape', SSDPostProcess.nms_default_params)
        self.nms_params = {
            'center_point_box': self.get_params(nms_params, 'center_point_box', SSDPostProcess.nms_default_params),
            'image_height': self.get_params(nms_params, 'image_height', SSDPostProcess.nms_default_params),
            'image_width': self.get_params(nms_params, 'image_width', SSDPostProcess.nms_default_params),
            'iou_threshold': self.get_params(nms_params, 'iou_threshold', SSDPostProcess.nms_default_params),
            'method': self.get_params(nms_params, 'method', SSDPostProcess.nms_default_params),
            'max_output_size': self.get_params(nms_params, 'max_output_size', SSDPostProcess.nms_default_params),
            'score_threshold': self.get_params(nms_params, 'score_threshold', SSDPostProcess.nms_default_params),
        }
        self.aipu_decodebox = None
        self.aipu_nms = None

    def __call__(self, input_tensors):
        self.aipu_decodebox = self.create_aipu_node(self.db_params, OpType.DecodeBox,
                                                    self.db_bottom_shape, self.db_top_shape, self.db_constants)
        self.aipu_decodebox.quantized = False
        self.aipu_decodebox.attrs['layer_id'] = '0'
        self.aipu_nms = self.create_aipu_node(self.nms_params, OpType.NMS, self.nms_bottom_shape, self.nms_top_shape)
        self.aipu_nms.quantized = False
        self.aipu_nms.attrs['layer_id'] = '1'
        db_out, nms_out = self.forward(input_tensors)
        return db_out, nms_out

    def get_params(self, params, key, opt_params):
        if params is not None and key in params:
            return params[key]
        elif key in opt_params:
            # print(f"key={key}")
            return opt_params[key]
        else:
            OPT_ERROR(f"{key} is not existed.")

    def create_aipu_node(self, params, op_type, i_shapes, o_shapes, constants=None):
        aipu_node = PyNode(str(op_type), op_type)
        for i, ishape in enumerate(i_shapes):
            ti = PyTensor(f"{op_type}_in_{i}", TensorShape(ishape))
            aipu_node.add_input(ti)
        for o, oshape in enumerate(o_shapes):
            to = PyTensor(f"{op_type}_out_{o}", TensorShape(oshape))
            aipu_node.add_output(to)

        for k, v in params.items():
            aipu_node.params.update({k: v})

        if constants is not None:
            for kc, vc in constants.items():
                tc = PyTensor(kc, vc)
                aipu_node.constants[kc] = tc

        return aipu_node

    def aipu_forward(self, node, input_tensors, forward_func):
        for itd, it in zip(input_tensors, node.inputs):
            it.betensor = itd

        out_tensors = forward_func(node)
        return out_tensors

    def forward(self, input_tensors):
        outt_db = self.aipu_forward(self.aipu_decodebox, input_tensors, decodebox_ssd)
        outt_nms = self.aipu_forward(self.aipu_nms, outt_db, Nms)
        return outt_db, outt_nms
