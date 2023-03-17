# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


def check_nodes_similarity(float_graph, quant_graph, inputs):
    '''
    use one input to test the network
    check each node's similarity between float graph and quant graph
    '''
    from AIPUBuilder.Optimizer.logger import OPT_INFO, OPT_DEBUG
    from AIPUBuilder.Optimizer.utils.quant_tool_utils import cosine_distance

    float_graph.forward(inputs, disable_pbar=False)
    quant_graph.forward(inputs, disable_pbar=False)

    for node in quant_graph.nodes:
        for t in node.outputs:
            float_t = float_graph.tensors(t.name)
            sim = cosine_distance(float_t.betensor+float_t.zerop, t.betensor+t.zerop)
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
            else:
                t.similarity.append(sim)


def show_similarity(quant_graph):
    from AIPUBuilder.Optimizer.logger import OPT_DEBUG
    type_max_len = max([len(str(n.type)) for n in quant_graph.nodes]) if len(quant_graph.nodes) > 0 else 0
    for node in quant_graph.nodes:
        msg = f"layer_type={str(node.type): <{type_max_len+4}} layer_id={node.attrs['layer_id']: <5}"
        sims = []
        for t in node.outputs:
            t.similarity = sum(t.similarity) / len(t.similarity)
            sims.append(str(t.similarity))
            msg += '  cos_dist={: <8.6f}    '.format(t.similarity)
            msg += 'out.scale={: <12.6f}    out.zerop={: <6.1f}    '.format(t.scale, t.zerop)
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
