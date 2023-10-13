import numpy as np
import copy
import torch
from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.utils import *


def greatest_common_divisor(x0, x1):
    m = max(x0, x1)
    n = min(x0, x1)
    t = m % n
    while t != 0:
        m, n = n, t
        t = m % n
    return n


class OptSubgraph(object):
    def __init__(self, subgraph_name='opt_subgraph'):
        self.subgraph_name = subgraph_name
        self.subgraph = QuantizeGraph(subgraph_name)
        self.quant_subgraph = None

    @staticmethod
    def get_node_param(graph, node_name, param_name):
        ret = None
        for sg_node in graph.nodes:
            if node_name in sg_node.name or sg_node.name in node_name:
                ret = sg_node.get_param(param_name)
        if ret is None:
            OPT_ERROR(f"the node_name={node_name} is not in subgraph nodes, please recheck the node name")
        return ret

    @staticmethod
    def get_node_constant(graph, node_name, constant_name):
        ret = None
        for sg_node in graph.nodes:
            if node_name in sg_node.name or sg_node.name in node_name:
                ret = sg_node.get_constant(constant_name)
        if ret is None:
            OPT_ERROR(f"the node_name={node_name} is not in subgraph nodes, please recheck the node name")
        return ret

    @staticmethod
    def set_node_param(node, params):
        # float IR params for float;
        for key, p in params.items():
            node.params[key] = p

    @staticmethod
    def create_node(node_name, op_type, inputs, outputs,  params, attrs):
        # step1. new node
        node = PyNode(node_name, op_type)  # -> str, OpType

        # step2. add input
        node.additional = True
        node.attrs["additional"] = True
        for i in inputs:
            if op_type in [OpType.Constant, ]:
                if len(inputs) != 1:
                    OPT_ERROR(f"weights tensor number must be one, please check the Constant op inputs.")
                node.constants["weights"] = i
                break
            else:
                node.add_input(i)

        # step3. add output
        for o in outputs:
            if isinstance(o, PyTensor):
                node.add_output(o)
            else:
                output_dtype = Dtype.FP32
                if len(o) == 3:  # output_name, output_shape, output_dtype:default=Dtype.FP32
                    quantization = o[2]
                    output_dtype = quantization["dtype"]
                out_tensor = PyTensor(o[0], TensorShape(o[1]), output_dtype)
                out_tensor.ir_shape = TensorShape(o[1])
                if len(o) == 3:
                    quantization = o[2]
                    for k, v in quantization.items():
                        out_tensor.__setattr__(k, v)
                node.add_output(out_tensor)

        # step4. set params(IR params)
        OptSubgraph.set_node_param(node, params)

        # step5: set attrs
        if attrs is not None:
            for k, v in attrs.items():
                node.attrs[k] = copy.deepcopy(v)
        return node

    def add_node_to_subgraph(self, node):
        if self.subgraph is None:
            self.subgraph = QuantizeGraph(self.subgraph_name)
        self.subgraph.add_node(node)


class BilinearResizeSubgraph(OptSubgraph):

    def __init__(self, g=None, subgraph_name='BilinearResize_subgraph'):
        super().__init__(subgraph_name)

        if g is not None:
            self.subgraph = g

    def add_to_subgraph(self, node_name, op_type, inputs, outputs, params, attrs):
        new_node = self.create_node(node_name, op_type, inputs, outputs, params, attrs)
        self.add_node_to_subgraph(new_node)
        return new_node

    def generate_resize_kernel(self, i_w, i_h, i_c, o_w, o_h):
        # tf format  [filter_height, filter_width, in_channels, out_channels]
        # ir format [out_channels, filter_height, filter_width, in_channels]
        if (i_w > o_w) or (i_h > o_h):
            filter_height = i_h
            filter_width = i_w
            src_loc_h = np.arange(0, i_h, 1) * o_h
            src_loc_w = np.arange(0, i_w, 1) * o_w
        else:
            filter_height = i_h + 1
            filter_width = i_w + 1
            src_loc_h = np.arange(0, i_h + 1, 1) * o_h
            src_loc_w = np.arange(0, i_w + 1, 1) * o_w
        in_channels = i_c
        out_channels = o_h * o_w * i_c
        conv_weight = np.zeros([out_channels, filter_height, filter_width, in_channels], dtype=np.float32)
        conv_bias = np.zeros(out_channels, dtype=np.float32)
        x_src_loc, y_src_loc = np.meshgrid(src_loc_w, src_loc_h)

        dst_loc_h = np.arange(0, o_h, 1) * i_h
        dst_loc_w = np.arange(0, o_w, 1) * i_w
        x_dst_loc, y_dst_loc = np.meshgrid(dst_loc_w, dst_loc_h)

        y_up = y_down = x_left = x_right = 0
        sub_out_channels = o_h * o_w

        one_conv_weight = np.zeros([sub_out_channels, filter_height, filter_width, 1])
        for c in range(sub_out_channels):
            c_y = c // o_h
            c_x = c - c_y * o_h
            y_dst = y_dst_loc[c_y][c_x]
            x_dst = x_dst_loc[c_y][c_x]

            for i in range(i_w):
                if (y_dst >= y_src_loc[i][0]) and (y_dst < y_src_loc[i + 1][0]):
                    y_up = i
                    y_down = i + 1

            for j in range(i_h):
                if (x_dst >= x_src_loc[0][j]) and (x_dst < x_src_loc[0][j + 1]):
                    x_left = j
                    x_right = j + 1

            # align_corners =fasle
            y_sc = 1 - (y_dst_loc[c_y][0] - y_src_loc[y_up][0]) / (y_src_loc[y_down][0] - y_src_loc[y_up][0])
            x_sc = 1 - (x_dst_loc[0][c_x] - x_src_loc[0][x_left]) / (x_src_loc[0][x_right] - x_src_loc[0][x_left])
            one_conv_weight[c][y_up][x_left][0] = y_sc * x_sc
            one_conv_weight[c][y_up][x_right][0] = y_sc * (1 - x_sc)
            one_conv_weight[c][y_down][x_left][0] = (1 - y_sc) * x_sc
            one_conv_weight[c][y_down][x_right][0] = (1 - y_sc) * (1 - x_sc)

        for i in range(sub_out_channels):
            for j in range(i_c):
                conv_weight[i + j * sub_out_channels, :, :, j] = one_conv_weight[i, :, :, 0]
        return conv_weight, conv_bias

    def get_quantization(self, tensor):
        quantization = {}
        for key in ["scale", "zerop", "qbits", "qmin", "qmax", "qinvariant", "dtype", "ir_dtype"]:
            quantization.update({key: tensor.__getattribute__(key)})
        return quantization

    def _update_layer_id(self, attrs, major_id, mini_id):
        l_id = f"{major_id}_{str(mini_id)}" if mini_id != 0 else f"{major_id}"
        attrs['layer_id'] = l_id
        return mini_id+1

    def build_subgraph(self, resize_node):
        resize_attrs = resize_node.attrs
        # resize_input = copy.deepcopy(resize_node.inputs[0])
        # resize_output = copy.deepcopy(resize_node.outputs[0])
        resize_input = resize_node.inputs[0].clone(resize_node.inputs[0].name)
        resize_output = resize_node.outputs[0].clone(resize_node.outputs[0].name)

        H, W, i_c = resize_node.inputs[0].ir_shape[1:]
        ResH, ResW = resize_node.outputs[0].ir_shape[1:3]
        h_gcd = greatest_common_divisor(H, ResH)
        w_gcd = greatest_common_divisor(W, ResW)
        i_w = int(W / w_gcd)
        i_h = int(H / h_gcd)
        o_w = int(ResW / w_gcd)
        o_h = int(ResH / h_gcd)
        if i_w == 1 or i_h == 1:
            i_w = i_w * 2
            i_h = i_h * 2
            o_w = o_w * 2
            o_h = o_h * 2
        i_scale = i_w
        o_scale = o_w
        layer_id = resize_node.attrs["layer_id"]

        Co = o_scale * o_scale * i_c
        lid = 0
        conv_weight, conv_bias = self.generate_resize_kernel(i_scale, i_scale, i_c, o_scale, o_scale)

        pad_output_quantization = self.get_quantization(resize_input)
        conv_output_quantization = self.get_quantization(resize_output)
        qbits = dtype2bits(conv_output_quantization['dtype'])
        if conv_output_quantization['qbits'] is None:
            conv_output_quantization['qbits'] = qbits
        if pad_output_quantization['qbits'] is None:
            pad_output_quantization['qbits'] = qbits

        base_name = resize_node.name

        conv_attrs = resize_attrs
        conv_attrs['conv_from_resize_opt'] = True
        if (ResH >= H) or (ResW >= W):
            K = i_scale + 1
            stride_x = K - 1
            stride_y = K - 1
            mirror_pad = self.add_to_subgraph(f"{base_name}_mirror_pad", OpType.Pad,
                                              [resize_input],
                                              [[f"{base_name}_mirror_pad", [1, H + 1, W + 1, i_c], pad_output_quantization]],
                                              {"mode": 'REFLECT',
                                               "pads": [[0, 0], [0, 1], [0, 1], [0, 0]],
                                               "constant_value": 0.000000},
                                              conv_attrs)
            lid = self._update_layer_id(mirror_pad.attrs, layer_id, lid)
            conv = self.add_to_subgraph(f"{base_name}_conv", OpType.Convolution,
                                        [mirror_pad.outputs[0]],
                                        [[f"{base_name}_conv", [
                                            1, int(ResH / o_scale), int(ResW / o_scale), Co], conv_output_quantization]],
                                        {"dilation_x": 1, "dilation_y": 1, "group": 1,
                                         "kernel_x": K, "kernel_y": K, "num_output": Co,
                                         "pad_bottom": 0, "pad_left": 0, "pad_right": 0, "pad_top": 0,
                                         "stride_x": stride_x, "stride_y": stride_y,
                                         "with_activation": 'NONE'},
                                        conv_attrs)
        else:
            K = i_scale
            stride_x = K
            stride_y = K
            conv = self.add_to_subgraph(f"{base_name}_conv", OpType.Convolution,
                                        [resize_input],
                                        [[f"{base_name}_conv", [
                                            1, int(ResH / o_scale), int(ResW / o_scale), Co], conv_output_quantization]],
                                        {"dilation_x": 1, "dilation_y": 1, "group": 1,
                                         "kernel_x": K, "kernel_y": K, "num_output": Co,
                                         "pad_bottom": 0, "pad_left": 0, "pad_right": 0, "pad_top": 0,
                                         "stride_x": stride_x, "stride_y": stride_y,
                                         "with_activation": 'NONE'},
                                        resize_attrs)

        lid = self._update_layer_id(conv.attrs, layer_id, lid)
        conv.constants["weights"] = PyTensor("resize_weights", conv_weight)
        conv.constants["biases"] = PyTensor("resize_biases", conv_bias)

        if i_c > 1:
            split_out_list = []
            for i in range(i_c):
                split_out_list.append([f"{base_name}_split_{i}", [1, int(ResH/o_scale),
                                      int(ResW/o_scale), o_scale * o_scale], conv_output_quantization])

            s_dim_t = []
            for i in range(i_c):
                s_dim_t.append(o_scale * o_scale)
            split = self.add_to_subgraph(f"{base_name}_split", OpType.Split,
                                         [conv.outputs[0]],
                                         split_out_list,
                                         {"axis": -1, "splits": s_dim_t},
                                         resize_attrs)
            lid = self._update_layer_id(split.attrs, layer_id, lid)

            concat = self.add_to_subgraph(f"{base_name}_concat", OpType.Concat,
                                          split.outputs,
                                          [[f"{base_name}_concat", [
                                              i_c, int(ResH/o_scale), int(ResW/o_scale), o_scale * o_scale], conv_output_quantization]],
                                          {"axis": 0},
                                          resize_attrs)
            lid = self._update_layer_id(concat.attrs, layer_id, lid)

            depth2space = self.add_to_subgraph(f"{base_name}_D2S", OpType.DepthToSpace,
                                               [concat.outputs[0]],
                                               [[f"{base_name}_D2S", [i_c, ResH, ResW, 1], conv_output_quantization]],
                                               {'block_size_x': o_scale, 'block_size_y': o_scale},
                                               resize_attrs)
            lid = self._update_layer_id(depth2space.attrs, layer_id, lid)

            trans = self.add_to_subgraph(f"{base_name}_Trans", OpType.Transpose,
                                         [depth2space.outputs[0]],
                                         [[f"{resize_node.outputs[0].name}", [1, ResH, ResW, i_c], conv_output_quantization]],
                                         {"perm": [3, 1, 2, 0]},
                                         resize_attrs)
            lid = self._update_layer_id(trans.attrs, layer_id, lid)

        else:
            depth2space = self.add_to_subgraph(f"{base_name}_D2S", OpType.DepthToSpace,
                                               [conv.outputs[0]],
                                               [[f"{resize_node.outputs[0].name}", [
                                                   i_c, ResH, ResW, 1], conv_output_quantization]],
                                               {'block_size_x': o_scale, 'block_size_y': o_scale},
                                               resize_attrs)
            lid = self._update_layer_id(depth2space.attrs, layer_id, lid)

        self.subgraph.remove_node(resize_node)

        return self.subgraph


def criteria(node):
    if node.type != OpType.Resize:
        return False
    if node.get_param("method").lower() != "bilinear":
        return False

    H, W, C = node.inputs[0].ir_shape[1:]
    OH, OW, OC = node.outputs[0].ir_shape[1:]
    ratio_x = node.get_param('ratio_x', optional=True, default_value=OW/W)
    ratio_y = node.get_param('ratio_y', optional=True, default_value=OH/H)
    if ratio_x <= 1 and ratio_y <= 1 and C > 8:
        OPT_DEBUG(f"now convert_resize_to_convolution only supports in_channel(={C}) <= 8 when shrinking the size.")
        return False

    if ratio_x > 1 and ratio_y > 1 and C > 1:
        OPT_DEBUG(f"now convert_resize_to_convolution only supports in_channel(={C}) <= 1 when enlarging the size.")
        return False

    ResH, ResW = node.outputs[0].ir_shape[1:3]
    h_gcd = greatest_common_divisor(H, ResH)
    w_gcd = greatest_common_divisor(W, ResW)

    i_w = int(W / w_gcd)
    i_h = int(H / h_gcd)
    o_w = int(ResW / w_gcd)
    o_h = int(ResH / h_gcd)

    if i_w != i_h or o_w != o_h:
        OPT_DEBUG(f"convert_resize_to_convolution only supports height(=in_h/h_gcd or out_h/h_gcd) and "
                  f"width(=in_w/w_gcd or out_w/w_gcd) should be equal in input and output tensor, and h_gcd means the "
                  f"greatest common divisor between input height and output height. now in_h/h_gcd={i_h}, "
                  f"in_w/w_gcd={i_w}, out_h/h_gcd={o_h}, out_w/w_gcd={o_w}.")
        return False

    return True


def _resize_to_convolution(g, resize_node):
    resize_to_conv_handler = BilinearResizeSubgraph(g)
    resize_to_conv_handler.build_subgraph(resize_node)


def convert_resize_to_convolution(graph):
    nodes = graph.nodes
    for n in nodes:
        if not criteria(n):
            continue
        _resize_to_convolution(graph, n)
