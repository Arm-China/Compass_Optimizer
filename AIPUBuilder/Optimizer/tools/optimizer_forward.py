# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from AIPUBuilder.Optimizer.plugins import *
from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.optmaster import *
from AIPUBuilder.Optimizer.logger import *


class OptForward(object):

    def __init__(self, ir_txt, ir_bin, forward_type='float'):
        '''
        OptForward is a interface for float/quantized graph forward calling the optimizer ops's forward, and only
        provides the Compass IR.
        :param ir_txt: Compass IR .txt path
        :param ir_bin:  Compass IR .bin path
        :param forward_type: ['float', 'quantized', 'mixed']
        '''
        traverse_opt_plugins()
        self.ir_txt = ir_txt
        self.ir_bin = ir_bin
        self.forward_type = forward_type.lower()

        if not self.check_path():
            OPT_FATAL(f"please check the ir_txt and ir_bin, and OptFloatForward init failed.")

        self.graph = QuantizeGraph.parse(ir_txt, ir_bin)
        self.optimizer = OptMaster(self.graph, None)
        for node in self.optimizer.g.nodes:
            node.attrs['layer_id'] = str(node.attrs.get('layer_id', -1))
        if self.forward_type != 'float':
            if self.optimizer.g.quantgraph is None:
                self.optimizer.g.quantgraph = self.optimizer.g.clone()
            self.optimizer.deduce_quantization_infos(self.optimizer.g.quantgraph)

    def check_path(self):
        ret = True
        ret = ret and os.path.isfile(self.ir_txt) and os.path.isfile(self.ir_bin)
        return ret

    def check_input_data(self, data):
        if isinstance(data, dict):
            data_list = [d for k, d in data.items()]
        elif len(self.optimizer.g.input_tensors) == 1 and not isinstance(data, (list, tuple, set)):
            data_list = [data, ]
        else:
            data_list = data

        if len(self.optimizer.g.input_tensors) != len(data_list):
            OPT_ERROR(f"please check the input data number={len(data_list)}, "
                      f"which is not equal to input_tensors number={len(self.optimizer.g.input_tensors)}")
            return False
        return data_list

    def get_qgraph_input_output(self):
        ''' this is use to get the quantized graph input and output tensors after quantized forward.'''
        if self.optimizer.g.quantgraph is None:
            OPT_ERROR(f"please run quantized forward firstly.")
            return
        qgraph = self.optimizer.g.quantgraph
        input_tensors = [i.betensor.cpu().numpy().astype(dtype2nptype(i.dtype)) for i in qgraph.input_tensors]
        output_tensors = [o.betensor.cpu().numpy().astype(dtype2nptype(o.dtype)) for o in qgraph.output_tensors]
        return input_tensors, output_tensors

    def _float_forward(self, data, keep_tensors=False):
        input_data = self.check_input_data(data)
        if input_data:
            output = self.optimizer.g.forward(input_data, keep_tensors=keep_tensors)
            output_data = [o.betensor.cpu().numpy() for o in output]
        else:
            output_data = None

        return output_data

    def _quantized_forward(self, data, transfer_to_float, keep_tensors=False):
        input_data = self.check_input_data(data)
        if input_data:
            out = self.optimizer.g.qforward(input_data, keep_tensors=keep_tensors)
            if transfer_to_float:
                output_data = [linear_dequantize(o.betensor, o.scale, o.zerop).cpu().numpy() for o in out]
            else:
                output_data = [o.betensor.cpu().numpy() for o in out]
        else:
            output_data = None

        return output_data

    def forward(self, data, transfer_to_float=True, keep_tensors=False):
        if self.forward_type == 'float':
            out = self._float_forward(data, keep_tensors=keep_tensors)
        else:
            out = self._quantized_forward(data, transfer_to_float=transfer_to_float, keep_tensors=keep_tensors)
        return out

    def forward_with_quantized_data(self, quantized_data, transfer_to_float=True, batch_size=1, keep_tensors=False):
        '''default this function is used in quantized graph forward'''

        if self.optimizer.g.quantgraph is None:
            OPT_ERROR(f"please use forward_type=quantized mode when calling forward_with_quantized_data function.")
        input_data = self.check_input_data(quantized_data)
        input_tensors = self.optimizer.g.quantgraph.input_tensors
        self.optimizer.g.quantgraph.current_batch_size = batch_size
        self.optimizer.g.quantgraph.current_batch_idx = 0
        dequantized_data = []
        for data, inp_t in zip(input_data, input_tensors):
            d = (data.astype('int64') + inp_t.zerop) / inp_t.scale
            dequantized_data.append(d)
        out = self._quantized_forward(dequantized_data, transfer_to_float=transfer_to_float, keep_tensors=keep_tensors)
        return out
