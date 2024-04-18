# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

import os
from AIPUBuilder.Optimizer.plugins import *
from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.utils import dtype2nptype, torch_tensor
from AIPUBuilder.Optimizer.optmaster import *
from AIPUBuilder.Optimizer.logger import *


class OptForward(object):

    def __init__(self, ir_txt_or_pyg, ir_bin=None):
        """
        OptForward is a interface for float/quantized/mixed graph forward calling the optimizer ops's forward.
        :param ir_txt_or_pyg: Compass IR .txt path or a PyGraph instance
        :param ir_bin: Compass IR .bin path when ir_txt_or_pyg is Compass IR .txt, otherwise ir_bin is None
        """

        traverse_opt_plugins()

        if isinstance(ir_txt_or_pyg, PyGraph):
            self.graph = ir_txt_or_pyg
        else:

            self.ir_txt = ir_txt_or_pyg
            self.ir_bin = ir_bin

            if not self.check_path():
                OPT_ERROR(f"please check the ir_txt and ir_bin, and OptFloatForward init failed.")

            self.graph = QuantizeGraph.parse(self.ir_txt, self.ir_bin)
        self.optimizer = OptMaster(self.graph, None)
        if self.optimizer.g is None:
            OPT_ERROR("build optimizer graph failed.")

        for node in self.optimizer.g.nodes:
            node.attrs['layer_id'] = str(node.attrs.get('layer_id', -1))
            key_axes = node.get_param('activation_quantization_axis',
                                      optional=True, default_value=[None] * len(node.outputs))
            key_axes = [None if isinstance(ka, str) and ka.lower() == 'none' else ka for ka in key_axes]
            for oi, ot in enumerate(node.outputs):
                ot.key_axis = key_axes[oi]
        if not hasattr(self.optimizer.g, 'compat_quantized_model'):  # gsim output IR or opt output IR or self-define IR
            QuantizeGraph.deduce_quantization_infos(self.optimizer.g)
        elif self.optimizer.g.compat_quantized_model:
            OPT_ERROR(f"OptForward only uses Compass IR to inference output, not compat_quantized_model=true IR.")
        else:  # compat_quantized_model=False parser output float IR
            pass

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

    def get_input_output_tensors(self):
        ''' this is use to get the graph input and output tensors after graph forward.'''
        g = self.optimizer.g
        input_tensors = [i.betensor.cpu().numpy().astype(dtype2nptype(i.dtype)) for i in g.input_tensors]
        output_tensors = [o.betensor.cpu().numpy().astype(dtype2nptype(o.dtype)) for o in g.output_tensors]
        return input_tensors, output_tensors

    def forward(self, data, transfer_to_float=False, keep_tensors=False):
        """
        :param data: input float data uses to forward
        :param transfer_to_float: whether dequantize the output data, default=False
        :param keep_tensors: whether reserve the intermediate data, default=False
        :return: the forward result
        """
        output_data = []
        input_data = self.check_input_data(data)
        if input_data:
            out = self.optimizer.g.forward(input_data, keep_tensors=keep_tensors)
            for o in out:
                if transfer_to_float and (o.pnode is not None and not o.pnode.get_param('unquantifiable', optional=True, default_value=False)):
                    o_data = linear_dequantize(o.betensor, o.broadcast_scale, o.broadcast_zerop).cpu().numpy()
                else:
                    # keep the ir_dtype
                    o_data = o.betensor.cpu().numpy().astype(dtype2nptype(o.ir_dtype))
                if len(o.ir_shape) <= 1:
                    o_data = o_data.reshape(o.ir_shape)
                output_data.append(o_data)

        return output_data

    def forward_with_quantized_data(self, quantized_data, transfer_to_float=False, batch_size=1, keep_tensors=False):
        """
        default this function is used when input data which is used to forward is quantized.
        :param quantized_data: the quantized data which is used to forward
        :param transfer_to_float: whether dequantize the output data, default=False
        :param batch_size: forward batch_size, default=1
        :param keep_tensors: whether reserve the intermediate data, default=False
        :return: the forward result
        """

        input_data = self.check_input_data(quantized_data)
        input_tensors = self.optimizer.g.input_tensors
        self.optimizer.g.current_batch_size = batch_size
        self.optimizer.g.current_batch_idx = 0
        dequantized_data = []
        for data, inp_t in zip(input_data, input_tensors):
            d = linear_dequantize(torch_tensor(data).long(), inp_t.broadcast_scale, inp_t.broadcast_zerop)
            dequantized_data.append(d)
        out = self.forward(dequantized_data, transfer_to_float=transfer_to_float, keep_tensors=keep_tensors)
        return out
