# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.
from AIPUBuilder import ops
from AIPUBuilder.core import *
import torch
import numpy as np

import sys
import os


################################Reset your own AIPUBuilder wheel path and simulator path################
# install AIPUBuilder wheel path, such as '/.local/lib/python3.10/site-packages'
PKG_DIR = '/.local/lib/python3.10/site-packages'
sys.path.insert(0, PKG_DIR)
owner_simulator_path = '/project/ai/scratch01/AIPU_SIMULATOR/aipu_simulator_x3'
########################################################################################################


##########################################build graph######################################################
inputs_shape = [1, 16, 256, 512]
dim = -1
with Graph() as g:
    sftm_in = Tensor(np.random.uniform(-5, 3, inputs_shape).astype(np.float32))
    # sftm_in.quantization.mins = [sftm_in.betensor.min().item()]
    # sftm_in.quantization.maxs = [sftm_in.betensor.max().item()]

    sftm_out = ops.softmax(sftm_in, axis=dim)
    sftm_out.op.attrs['unquantifiable'] = True
    sftm_out.op.attrs['approx_params'] = NodeParamValue(1)
    sftm_out.op.attrs['min_compatible_zhouyi_target'] = NodeParamValue("x3")
    sftm_in.op.attrs['unquantifiable'] = True

    g.serialize_scale_zp = True
############################################################################################################


################################################qtlib for graph#############################################
for n in g.nodes:
    n.attrs['trigger_float_op'] = NodeParamValue('float16')
q = Quantizer(g)
q.quantize()
print(g.serialize()[0])
############################################################################################################


#################################################run model##################################################
arg_map = {
    'simulator': owner_simulator_path,
    'target': 'X3_1304',
    'dump': True,
    'output': 'simout.bin',
}

input_fp32_datas = [np.random.randn(*inputs_shape).astype(np.float32)]
input_datas = []
for inp_tensor, inp_data in zip(g.input_tensors, input_fp32_datas):
    if not np.issubdtype(inp_tensor.dtype.np, np.floating):
        inp_data = np.round(
            inp_data.astype(np.float32) * inp_tensor.quantization.scale).astype(np.int32) - inp_tensor.quantization.offset
    inp_data = inp_data.astype(inp_tensor.dtype.np)
    inp_tensor.set_numpy(inp_data)
    input_datas.append(inp_data)
optimizer_g = aipurun(g, g.input_tensors, arg_map)
############################################################################################################


###################################################compare with torch##########################################
def cosine_distance(a, b):
    x = torch.tensor(a, dtype=torch.float64) if not isinstance(a, torch.Tensor) else a.double()
    y = torch.tensor(b, dtype=torch.float64) if not isinstance(b, torch.Tensor) else b.double()

    t1 = x.flatten()
    t2 = y.flatten()
    t1_m = torch.norm(t1, p=2)
    t2_m = torch.norm(t2, p=2)
    t1t2 = torch.dot(t1, t2)
    t1t2_m = t1_m * t2_m
    if t1t2_m.item() == 0.0:
        if t1_m == t2_m:
            return 1.0
        else:
            return 0.0
    else:
        if t1t2 == t1t2_m:
            return 1.0
        else:
            return (t1t2 / t1t2_m).item()


sim_out = optimizer_g.output_tensors[0].betensor
torch_out = torch.nn.functional.softmax(torch.from_numpy(input_datas[0]).to(torch.float16), dim=dim)
diff = torch_out - sim_out.to(torch_out.device)

print(f"cosine={cosine_distance(torch_out, sim_out.to(torch_out.device))}, abs_max_diff = {diff.abs().max()}")
#############################################################################################################
