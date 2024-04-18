# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

def calculate_op_running_time(f_graph, q_graph):

    from AIPUBuilder.Optimizer.logger import OPT_DEBUG
    nname_id = {}
    for idx, n in enumerate(f_graph.nodes):
        nname_id.update({n.name: idx})
    cost_times = {}
    for n in q_graph.nodes:
        key = f"{n.attrs['layer_id']} {str(n.type)[7:]}"
        q_cost_time = n.attrs['cost_time']
        f_cost_time = 0
        if n.name in nname_id.keys():
            fnodes = f_graph.nodes[nname_id[n.name]]
            f_cost_time = fnodes.attrs['cost_time']
        ct = [f_cost_time, q_cost_time]
        cost_times.update({key: ct})

    fall_times = sum([v[0] for v in cost_times.values()])
    qall_times = sum([v[1] for v in cost_times.values()])
    type_max_len = max([len(k) for k in cost_times.keys()]) if len(cost_times.keys()) > 0 else 0
    for k, v in cost_times.items():
        v.append(v[0] / fall_times * 100)
        v.append(v[1] / qall_times * 100)
        cost_times[k] = v
        ostr = (f"layer_type={k:{type_max_len}} fp32_forward_time={v[0]:<8.6f}s, quant_forward_time={v[1]:<8.6f}s, "
                f"this_fp32/all_fp32={v[2]:<3.6f}%%, this_quant/all_quant={v[3]:<3.6f}%%")
        OPT_DEBUG(ostr)

    # disable to calculate op running time
    for n in f_graph.nodes:
        n.attrs['calculate_running_time'] = False
    for n in q_graph.nodes:
        n.attrs['calculate_running_time'] = False
