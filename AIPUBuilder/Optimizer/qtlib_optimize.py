# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.logger import OPT_INFO, OPT_ERROR
from AIPUBuilder.Optimizer.framework import OpType, convert_aipu_graph_to_opt_graph, convert_opt_graph_to_aipu_graph
from AIPUBuilder.Optimizer.passes import convert_resize_to_convolution

import os


class QTLibOptimize(object):

    def __init__(self, graph, config):
        self.g = graph
        self.hparams = config

    def __call__(self):
        OPT_INFO(f"Now we do quantization transform in Optimizer.")
        try:
            from AIPUBuilder.core import quantize_transform as qtlib_quantize_transform
        except Exception as e:
            OPT_ERROR(
                f"The AIPUBuilder.core module is required when compat_quantized_model is True. now error message: {e}")

        new_quantization_method_ops_type = []
        if self.hparams.compat_quantized_model_ops != '':
            lower_optype = {}
            for k, v in OpType.__dict__.items():
                lower_optype.update({k.lower(): v})
            for op in self.hparams.compat_quantized_model_ops.strip().replace(' ', '').split(','):
                new_quantization_method_ops_type.append(lower_optype[op])
        # pre-pass
        self.hparams.__setattr__('enable_pass_convert_resize_to_convolution', True)
        convert_resize_to_convolution(self.g, self.hparams)
        for n in self.g.nodes:
            n.attrs['compat_quantized_model_simplify_dequantize_quantize'] = self.hparams.compat_quantized_model_simplify_dequantize_quantize
            n.attrs['unify_shifts_mode'] = self.hparams.compat_quantized_model_unify_shifts_mode
            # n.attrs['force_weight_asym'] = True
            # now qat model fixed to 13bits
            n.attrs['multiplier_bits'] = 13
            if 'conv_from_resize_opt' not in n.attrs:
                n.attrs['trigger_float_op'] = 'float16_preferred' if self.hparams.trigger_float_op.get(
                    n) == 'disable' else self.hparams.trigger_float_op.get(n)
            if self.hparams.compat_quantized_model_int8_to_uint8:
                n.attrs["int8_to_uint8"] = True
            if n.type == OpType.Constant:
                n.attrs['scale_zp_need_quantize'] = True
            if n.type in new_quantization_method_ops_type:
                n.attrs['tflite_quantization'] = True
            if n.type == OpType.Eltwise:
                n.attrs['left_shift_bits'] = self.hparams.compat_quantized_model_left_shift_bits
            if n.type == OpType.BasicLSTM:
                n.attrs['weight_dim'] = 1
                n.attrs['set_default_placeholder_info'] = True
            if n.type == OpType.Cast:
                n.attrs["eliminate_cast"] = self.hparams.compat_quantized_model_eliminate_cast

        cg = convert_opt_graph_to_aipu_graph(self.g)
        qtlib_quantize_transform(cg, run_mode=self.hparams.compat_quantized_model_strategy)
        name = os.path.join(self.hparams.output_dir, self.hparams.out_ir_name)
        if not os.path.exists(self.hparams.output_dir):
            os.makedirs(self.hparams.output_dir)
        if not self.hparams.eval_optimized_model or self.hparams.metric == '':
            cg.attrs['serialize_scale_zp'] = True
            cg.serialize(f"{name}.txt", f"{name}.bin")
            # no need forward, so we return self.g.quantgraph(=None)
        else:
            self.g.quantgraph = convert_aipu_graph_to_opt_graph(cg)
            self.g.quantgraph.serialize(f"{name}.txt", f"{name}.bin")
        return self.g.quantgraph
