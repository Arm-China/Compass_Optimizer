import argparse

import numpy as np
import torch
import os
import sys
from AIPUBuilder.Optimizer.tools.optimizer_forward import OptForward
from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.utils import *


class CmpOpForward(object):

    def __init__(self, case_path, forward_mode, forward_egine: list, dump):
        '''
        :param case_path:
        :param forward_mode: float, quantized
        :param forward_egine: [gt, opt, sim]
        :param dump:
        '''

        self.case_path = case_path
        self.forward_mode = forward_mode
        self.forward_egine = forward_egine
        self.opt_forward_egine = None
        self.g = None
        self.dump = dump

    def get_opt_forward_egine(self, ir_txt, ir_bin):
        if self.opt_forward_egine is not None:
            OPT_WARN(f"now opt forward egine is not None, we will create the new opt forward egine")
        self.opt_forward_egine = OptForward(ir_txt, ir_bin, self.forward_mode)
        return self.opt_forward_egine

    def get_graph(self, ir_txt, ir_bin):
        self.g = PyGraph.parse(ir_txt, ir_bin)
        return self.g

    def get_data(self, case_path, shapes, dtypes, key):

        def get_fix(idx, key):
            pre_fix, post_fix = '', ''
            if key in ['gt', 'output']:
                if idx > 0:
                    post_fix = str(idx)
            elif key in ['input']:
                pre_fix = str(idx)
            return pre_fix, post_fix
        key_len = len(shapes)
        datas = []
        for i in range(key_len):
            pre_fix, post_fix = get_fix(i, key)
            bin_filepath = os.path.join(case_path, f"{key}{pre_fix}.bin{post_fix}")
            data = np.fromfile(bin_filepath, dtype=dtype2nptype(dtypes[i])).reshape(shapes[i])
            datas.append(data)

        if len(datas) != len(shapes):
            OPT_ERROR(f"get data number != shape number")

        return datas

    def cmp_forward(self):
        ir_txt = os.path.join(self.case_path, 'graph.def')
        ir_bin = os.path.join(self.case_path, 'weight.bin')
        self.get_graph(ir_txt, ir_bin)
        if self.g is None:
            OPT_ERROR(f"get graph failed.")
            return
        input_tensors = self.g.input_tensors
        output_tensors = self.g.output_tensors
        input_shapes = [inp_t.ir_shape for inp_t in input_tensors]
        input_dtypes = [inp_t.dtype for inp_t in input_tensors]
        output_shapes = [outp_t.ir_shape for outp_t in output_tensors]
        output_dtypes = [outp_t.dtype for outp_t in output_tensors]

        # now we cannot consider the batch issue
        input_datas = self.get_data(self.case_path, input_shapes, input_dtypes, 'input')

        if 'opt' in self.forward_egine:
            self.opt_forward_egine = self.get_opt_forward_egine(ir_txt, ir_bin)
            opt_output_data = self.opt_forward_egine.forward_with_quantized_data(input_datas)
            if self.dump:
                for i, n in enumerate(self.opt_forward_egine.optimizer.g.quantgraph.nodes):
                    layer_id = n.attrs['layer_id']  # str
                    for oi, o in enumerate(n.outputs):
                        dat = o.betensor.cpu().numpy()
                        dat = dat.astype(dtype2nptype(o.dtype))
                        dat.tofile(f"{layer_id}_opt_{oi}_{o.name}.bin")
                        OPT_INFO(o.name, " ", dtype2nptype(o.dtype), " ", dat.shape)

        if 'sim' in self.forward_egine:
            sim_output_data = self.get_data(self.case_path, output_shapes, output_dtypes, 'output')

        if 'gt' in self.forward_egine:
            gt_output_data = self.get_data(self.case_path, output_shapes, output_dtypes, 'gt')

        if 'gt' in self.forward_egine and 'opt' in self.forward_egine:
            for oi in range(len(output_shapes)):
                cos = torch.nn.functional.cosine_similarity(torch.tensor(gt_output_data[oi].flatten()),
                                                            torch.tensor(opt_output_data[oi].flatten()),
                                                            dim=0)
                max_diff = np.max(np.abs(opt_output_data[oi] - gt_output_data[oi]))
                OPT_INFO(f"output_{oi} cos between gt and opt: {cos}, max_diff = {max_diff}")
        if 'sim' in self.forward_egine and 'opt' in self.forward_egine:
            for oi in range(len(output_shapes)):
                cos = torch.nn.functional.cosine_similarity(torch.tensor(sim_output_data[oi].flatten()),
                                                            torch.tensor(opt_output_data[oi].flatten()),
                                                            dim=0)
                max_diff = np.max(np.abs(opt_output_data[oi] - sim_output_data[oi]))
                OPT_INFO(f"output_{oi} cos between simulator and opt: {cos}, max_diff = {max_diff}")

        OPT_INFO(f"compare done in {self.forward_egine}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case_path', type=str,
                        help="the gt output case information, including graph.def, weight.bin, inputi.bin", default='')
    parser.add_argument('--forward_mode', type=str, help="float, quantized", default='quantized')
    parser.add_argument('--egine', type=str, help="egine includes gt, opt, simulator", default='gt, opt')
    parser.add_argument('--dump', action="store_true", help="dump the all output tensor data")

    args = parser.parse_args()
    return args


def main():
    argv = get_args()
    cmp_forward_hanlder = CmpOpForward(argv.case_path, argv.forward_mode, argv.egine, argv.dump)
    cmp_forward_hanlder.cmp_forward()


if __name__ == '__main__':
    main()
