# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.


def check_nodes_similarity(float_graph, quant_graph, inputs, keep_tensors=False):
    '''
    use one input to test the network
    check each node's similarity between float graph and quant graph
    '''
    from AIPUBuilder.Optimizer.logger import OPT_INFO, OPT_DEBUG, OPT_ERROR
    from AIPUBuilder.Optimizer.utils.quant_tool_utils import cosine_distance, linear_dequantize
    from AIPUBuilder.Optimizer.framework.pycore.pytensor import PyTensor
    from torch.nn import MSELoss as mseloss

    MSE = mseloss()

    float_graph.feed_inputs_data(inputs)
    quant_graph.feed_inputs_data(inputs)
    if keep_tensors:
        # prevent deleting intermediate tensors
        float_graph.ref_count_tensors = {}
        quant_graph.ref_count_tensors = {}
    else:
        float_graph.reset_edge_tensors_ref_count()
        quant_graph.reset_edge_tensors_ref_count()
    for n, qn in zip(float_graph.nodes, quant_graph.nodes):
        n.forward()
        qn.forward()
        if n.type != qn.type or n.name != qn.name:
            OPT_ERROR(
                f"check_nodes_similarity: failed to match layer, one is '{n.type} {n.name}' and another is '{qn.type} {qn.name}'. ")
        for float_t, t in zip(n.outputs, qn.outputs):
            if float_t.name != t.name:
                OPT_ERROR(
                    f"check_nodes_similarity: failed to match tensor in '{n.type} {n.name}': '{float_t.name}' vs '{t.name}'. ")
            float_output = float_t.betensor
            de_quant_output = linear_dequantize(t.betensor, t.scale, t.zerop, t.key_axis)
            sim = cosine_distance(float_output, de_quant_output)
            mse = MSE(float_output, de_quant_output).item()
            if sim < 0.9:
                OPT_DEBUG(t.name, ' accuracy too low : %f' % sim)
                # ref = float_t.betensor+float_t.zerop
                # tar = t.betensor+t.zerop
                # import matplotlib.pyplot as plt
                # plt.subplot(131)
                # plt.title('out,%s'%t.name)
                # plt.plot(tar.reshape(-1).cpu()[:10000]/t.scale)
                # plt.subplot(132)
                # plt.title('ref')
                # plt.plot(ref.reshape(-1).cpu()[:10000])
                # plt.subplot(133)
                # plt.title('diff')
                # plt.plot(tar.reshape(-1).cpu()[:10000]/t.scale-ref.reshape(-1).cpu()[:10000])
                # plt.show()
            if t.similarity is None:
                t.similarity = [sim]
                t.mse = [mse]
            else:
                t.similarity.append(sim)
                t.mse.append(mse)
    if keep_tensors:
        pass
    else:
        tz = PyTensor('null').betensor
        for n in (float_graph.nodes + quant_graph.nodes):
            for t in n.outputs:
                del t.betensor
                t.betensor = tz
            for pld in n.placeholders:
                del pld.betensor
                pld.betensor = tz
        float_graph.reset_edge_tensors_ref_count()
        quant_graph.reset_edge_tensors_ref_count()


def show_similarity(quant_graph):
    from AIPUBuilder.Optimizer.logger import OPT_DEBUG
    type_max_len = max([len(str(n.type)) for n in quant_graph.nodes]) if len(quant_graph.nodes) > 0 else 0
    for node in quant_graph.nodes:
        msg = f"layer_type={str(node.type): <{type_max_len+4}} layer_id={node.attrs['layer_id']: <5}"
        sims = []
        for t in node.outputs:
            t_scale = t.scale[:10] if t.is_perchannel_scales() else t.scale
            t_zerop = t.zerop[:10] if t.is_perchannel_scales() else t.zerop
            t.similarity = sum(t.similarity) / len(t.similarity)
            sims.append(str(t.similarity))
            msg += '  cos_dist={: <8.6f} '.format(t.similarity)
            msg += f" mse={t.mse}    "
            msg += 'out.scale={}    out.zerop={}    '.format(t_scale, t_zerop)
            msg += 'out.qbits={: <}    out.qmin={: <}    out.qmax={: <}    '.format(
                str(t.qbits), str(t.qmin), str(t.qmax))
            msg += 'tensor_name={: <}'.format(t.name)
            sim = t.similarity

            if sim < 0.9:
                # import matplotlib.pyplot as plt
                # plt.subplot(131)
                # plt.title('out,%s'%t.name)
                # plt.plot(t.betensor.reshape(-1).cpu()/t.scale)
                # plt.subplot(132)
                # plt.title('ref')
                # plt.plot(float_t.betensor.reshape(-1).cpu())
                # plt.subplot(133)
                # plt.title('diff')
                # plt.plot(t.betensor.reshape(-1).cpu()/t.scale-float_t.betensor.reshape(-1).cpu())
                # plt.show()
                # OPT_DEBUG(t.name,'accuracy too low : %f'%sim)
                pass
        msg += '  layer_name={: <}'.format(node.name)
        # node.add_dot_section("similarity:%s" % ",".join(sims), "similarity")
        OPT_DEBUG(msg)
    out_sims = [t.similarity for t in quant_graph.output_tensors]
    OPT_DEBUG(
        f"graph output_tensors similarity (align with the order of 'output_tensors' in IR header):{str(out_sims)}")
    out_mse = [t.mse for t in quant_graph.output_tensors]
    OPT_DEBUG(
        f"graph output_tensors MSE (align with the order of 'output_tensors' in IR header):{str(out_mse)}")
