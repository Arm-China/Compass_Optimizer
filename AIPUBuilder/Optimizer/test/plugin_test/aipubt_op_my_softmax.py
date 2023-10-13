# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.


from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.logger import *
from AIPUBuilder.Optimizer.ops.softmax import softmax, softmax_quantize

# for optype out of IR guide's definition, use register_optype(xxtype_name_string) to register optype firstly
# register_optype('Softmax')


@op_register(OpType.Softmax, version='2.0')
def my_softmax(self, *args):
    OPT_INFO('Customized OP forward function is enabled.')
    return softmax(self, *args)


@quant_register(OpType.Softmax, version='2.0')
def my_softmax_quantize(self, *args):
    OPT_INFO('Customized OP quantize function is enabled.')
    inp = self.inputs[0]
    out = self.outputs[0]
    axis = self.get_param('axis')
    shape_value_in_axis = inp.betensor.shape[axis]
    if shape_value_in_axis < 8:
        customized_softmax_quantize_func(self, *args)
    else:
        softmax_quantize(self, *args)


def customized_softmax_quantize_func(self, *args):
    softmax_quantize(self, *args)
