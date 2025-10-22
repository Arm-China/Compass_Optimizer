# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

import argparse
import ast
import sys
import os
import numpy as np
from AIPUBuilder.Optimizer.tools.optimizer_forward import OptForward
from AIPUBuilder.Optimizer.utils import dtype2nptype, make_dir_path
from AIPUBuilder.Optimizer.logger import OPT_ERROR, OPT_INFO
from AIPUBuilder.Optimizer.framework import OpType


def arg_parser():
    args = argparse.ArgumentParser()
    args.add_argument("-v", "--version", action="version", version="1.0")
    args.add_argument(
        "graph", metavar="<graph.def>", type=str, help="graph definition file."
    )
    args.add_argument(
        "-w", "--weight", type=str, default="", help="the weight file for the graph."
    )
    args.add_argument(
        "-o",
        "--output",
        type=str,
        default="output.def",
        help="the optimized graph definition file.",
    )
    args.add_argument(
        "-b",
        "--bin",
        type=str,
        default="output.cbin",
        help="the optimized graph weight file.",
    )
    args.add_argument("--enable_ds", action="store_true", help="Enable dynamic shape")
    args.add_argument(
        "-i",
        "--input",
        type=str,
        default=None,
        help='The inputs file for the graph, if you have multiple inputs, use comma "," to split them. '
        "If no input file is provided, a valid input file will be generated randomly. "
        "Support multi-groups input, by input a .npy or other format file and specify a dataset plugin",
    )
    args.add_argument(
        "--input_shape", type=str, default=None, help="Specify input shape of graph"
    )
    args.add_argument("--fit_dtype", type=bool, default=True, help="Enable fit_dtype")
    args.add_argument(
        "--keep_tensor",
        action="store_true",
        help="reverse the intermediate tensor data",
    )
    args.add_argument(
        "-d",
        "--dump_output", action="store_true", help="whether dump the graph output data, default is false."
    )
    args.add_argument(
        "--dump_output_dir", type=str, default='./', help="dump the graph output data, default is ./"
    )

    options = args.parse_args(sys.argv[1:])
    return options


def check(args, g):
    check_ok = True
    if args.input_shape is not None:
        if not args.input_shape.startswith("[["):
            args.input_shape = "[" + args.input_shape + "]"
        input_shapes = ast.literal_eval(args.input_shape)
        check_input_shape = True
        if len(input_shapes) == len(g.input_tensors):
            for s, t in zip(input_shapes, g.input_tensors):
                if len(s) != len(t.ir_shape):
                    check_input_shape = False
                    break
        elif len(input_shapes) == args.input.count(".bin"):
            pass
        else:
            check_input_shape = False
        if not check_input_shape:
            OPT_ERROR("[opt_forward_main] check option --input_shape failed!")
            check_ok = check_input_shape

    return check_ok


def check_filename(file_path):
    # if dump file name of gt is longer than 253(255-2), cut the filename at end
    # gt start with i_gt_exe_j_(len 11), aipurun start with dumptensor_i_(len 13), 13-11=2
    if len(file_path) <= 253:
        return file_path
    path, name = os.path.split(file_path)
    name = name.split('.bin')[0]
    name = name[:249] + '.bin'
    return os.path.join(path, name)


def ds_prepass(g):
    def avgpool_to_globalpool(graph):
        for n in graph.nodes:
            if n.type == OpType.Pooling:
                o_irshape = n.outputs[0].ir_shape
                if o_irshape[1] == o_irshape[2] == 1:
                    n.params["from_pool"] = True
                    n.type = OpType.GlobalPool

    avgpool_to_globalpool(g)


def ds_middlepass(g):
    def revert_pool(graph):
        for n in graph.nodes:
            if n.type == OpType.GlobalPool and "from_pool" in n.params:
                n.type = OpType.Pooling
                n.params.pop("from_pool")

    revert_pool(g)


def ds_postpass(g):
    def change_reshape_shape(graph):
        for n in graph.nodes:
            if n.type == OpType.Reshape:
                shape_param = list(n.params['shape'])
                ir_out_shape = list(n.outputs[0].ir_shape)
                if shape_param != ir_out_shape:
                    n.params['shape'] = ir_out_shape

    change_reshape_shape(g)


def main():
    options = arg_parser()
    irt = options.graph
    irb = options.weight
    forward_handler = OptForward(irt, irb)
    g = forward_handler.graph

    ret = check(options, g)
    if not ret:
        return

    input_shapes = []
    if options.input_shape is not None:
        input_shapes = ast.literal_eval(options.input_shape)

    input_data = []
    if options.input is not None:
        input_names = options.input.split(",")
        for i, (t, name) in enumerate(zip(g.input_tensors, input_names)):
            if name.endswith(".npy"):
                input_data.append(np.load(name))
            elif name.endswith(".bin") and len(input_shapes):
                in_shape = input_shapes.pop(0)
                input_data.append(
                    np.fromfile(
                        name, dtype=dtype2nptype(g.input_tensors[i].dtype)
                    ).reshape(in_shape)
                )
            else:
                OPT_ERROR(
                    f"please check input file type={name}, now only support file name endswith .npy or .bin, and if use .bin, please set the --input_shape"
                )
    else:
        # zero input
        if len(input_shapes):  # use setted input_shape to generate the zero input
            for in_shape, inp in zip(input_shapes, g.input_tensors):
                shape = list(in_shape)
                dtype = inp.dtype
                input_data.append(np.zeros(shape).astype(dtype2nptype(dtype)))
        else:
            for inp in g.input_tensors:  # use ir input_shape to generate the zero input
                shape = list(inp.ir_shape)
                dtype = inp.dtype
                input_data.append(np.zeros(shape).astype(dtype2nptype(dtype)))

    if options.enable_ds:
        ds_prepass(g)

    quantized = False
    for t in g.input_tensors:
        if t.pnode.quantized:
            quantized = True
            break

    OPT_INFO(f"begin to inference graph in Optimizer backend...")
    if quantized:
        out = forward_handler.forward_with_quantized_data(
            input_data, keep_tensors=options.keep_tensor, fit_dtype=options.fit_dtype
        )
    else:
        out = forward_handler.forward(
            input_data, keep_tensors=options.keep_tensor, fit_dtype=options.fit_dtype
        )

    if options.enable_ds:
        OPT_INFO(f"begin to serialize the ds IR to the path {options.output}")
        ds_middlepass(g)
        for n in g.nodes:
            for ot in n.outputs:
                if "dynamic_shape" in ot.attrs:
                    dynamic_shape = ot.attrs["dynamic_shape"]
                    ot.ir_shape = dynamic_shape
        ds_postpass(g)

        g.serialize_scale_zp = True
        g.serialize(options.output, options.bin)

    # if needed, will save the graph output to file
    if options.dump_output:
        OPT_INFO(f"begin to dump the activation data to the path {options.dump_output_dir}")
        make_dir_path(options.dump_output_dir)
        if options.keep_tensor:
            for n in g.nodes:
                if n.type != OpType.Input:
                    lid = n.attrs['layer_id']
                    for ot_id, ot in enumerate(n.outputs):
                        data = ot.to_numpy().astype(dtype2nptype(ot.dtype))
                        file_name = f"{lid}_opt_{ot_id}_{n.type.name}_{ot.name.replace('/', '_')}.bin"
                        file_name = check_filename(file_name)
                        pn = os.path.join(options.dump_output_dir, file_name)
                        data.tofile(pn)
        else:
            for ot_id, ot in enumerate(g.output_tensors):
                lid = ot.pnode.attrs['layer_id']
                data = ot.to_numpy().astype(dtype2nptype(ot.dtype))
                file_name = f"{lid}_opt_{ot_id}_{ot.pnode.type.name}_{ot.name.replace('/', '_')}.bin"
                file_name = check_filename(file_name)
                pn = os.path.join(options.dump_output_dir, file_name)
                data.tofile(pn)

    OPT_INFO(f"opt inferences done.")


if __name__ == "__main__":
    main()
