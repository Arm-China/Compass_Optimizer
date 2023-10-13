# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.plugins import *
from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.config import *
from AIPUBuilder.Optimizer.optmaster import *
from AIPUBuilder.Optimizer.logger import tqdm
import onnx
import onnxruntime
import numpy as np
import psutil


def main():
    traverse_opt_plugins()
    argv = arg_parser(metric_dict=QUANTIZE_METRIC_DICT, dataset_dict=QUANTIZE_DATASET_DICT)
    g = QuantizeGraph.parse(argv.graph, argv.bin)
    opt = OptMaster(g, argv)
    opt.prepare(argv)
    # test quantized graph
    # opt.optimize()
    # opt.serialize(f'{argv.output_dir}/{argv.quant_ir_name}')
    # g = g.quantgraph
    # with_float = False
    with_float = True
    # set batch_size consistent with dataloader by forward one batch
    vdataloader = opt.validation_dataloader
    graph_inference(g, g.forward, vdataloader, [], with_float=True, max_batches=1, disable_tqdm=True)
    m = g.to_torch_module()
    print("Model's state_dict:")
    for param_tensor in m.state_dict():
        print(param_tensor, "\t", m.state_dict()[param_tensor].size())
    # torch.save(m.state_dict(), pth)
    pth = f'{argv.output_dir}/{argv.model_name}_module.onnx'
    g.export_as_onnx(pth, use_ir_shape=False)
    onnx_m = onnx.load(pth)
    ck = onnx.checker.check_model(onnx_m)
    print("Check onnx model:")
    print(ck)

    metric1 = opt.f_metrics
    metric2 = opt.q_metrics

    sess_options = onnxruntime.SessionOptions()
    sess_options.optimized_model_filepath = pth + '.optimized.onnx'
    sess_options.intra_op_num_threads = psutil.cpu_count(logical=True)
    ort_session = onnxruntime.InferenceSession(pth, sess_options)
    # ort_session = onnxruntime.InferenceSession(pth)

    g.current_batch_size = vdataloader.batch_size
    current_batch_idx = 0
    max_batches = -1
    with tqdm(vdataloader, desc='metric', file=sys.stdout) as pbar:
        for i, sample in enumerate(pbar):
            g.current_batch_idx = current_batch_idx
            current_batch_idx += 1
            if current_batch_idx * vdataloader.batch_size > len(vdataloader.dataset):
                g.current_batch_size = len(vdataloader.dataset) - (current_batch_idx - 1) * vdataloader.batch_size
            inp, target = sample
            if opt_use_cuda():
                if isinstance(target, dict):
                    target = {key: target[key].cuda() if isinstance(target[key], torch.Tensor) else target[key]
                              for key in target}
                elif isinstance(target, (list, tuple)):
                    target = [t.cuda() if isinstance(t, torch.Tensor) else t for t in target]
                else:
                    target = target.cuda() if isinstance(target, torch.Tensor) else target

                if isinstance(inp, dict):
                    inp = {key: inp[key].cuda() for key in inp}
                elif isinstance(inp, (list, tuple)):
                    inp = [ii.cuda() for ii in inp]
                else:
                    inp = inp.cuda()
            out = g.forward(inp)

            # compute ONNX Runtime output prediction
            ort_inputs = {}
            for k, t in enumerate(g.input_tensors):
                ort_inputs[ort_session.get_inputs()[k].name] = t.betensor.cpu().numpy()
            ort_out = ort_session.run(None, ort_inputs)

            # dequantize quantized forward's output tensors for consistently call metirc functions
            prediction1 = []
            prediction2 = []
            for k, t in enumerate(out):
                ot = PyTensor(f'out{k}', ort_out[k]).betensor
                if with_float:
                    prediction1.append(t.betensor)
                    prediction2.append(ot)
                else:
                    dtb1 = linear_dequantize(t.betensor, t.scale, t.zerop)
                    prediction1.append(dtb1)
                    dtb2 = linear_dequantize(ot, t.scale, t.zerop)
                    prediction2.append(dtb2)
                # np.testing.assert_allclose(prediction1[k].cpu().numpy(), prediction2[k].cpu().numpy(), rtol=1e-03, atol=1e-05)

            for metric in metric1:
                metric(prediction1, target)
            for metric in metric2:
                metric(prediction2, target)

            if max_batches > 0 and current_batch_idx >= max_batches:
                break
        pbar.refresh()
    for metric in metric1:
        print(f"opt forward: {metric.report()}")
    for metric in metric2:
        print(f"onnx forward: {metric.report()}")


if __name__ == '__main__':
    main()
