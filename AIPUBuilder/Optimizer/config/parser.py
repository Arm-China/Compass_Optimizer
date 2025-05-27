# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

import os
import re
import argparse
import configparser
from AIPUBuilder.Optimizer.logger import opt_workflow_register, OPT_ERROR, OPT_INFO, OPT_WARN, OPT_FATAL
from AIPUBuilder.Optimizer.framework import ALL_OPT_OP_DICT, ALL_OPT_QUANT_OP_DICT
from AIPUBuilder.Optimizer.utils import Target, string_to_base_type
from . cfg_fields import ALL_FIELDS, DEFAULT_FIELDS, PerNodeFieldDict


class CfgParser(object):
    def __init__(self, cfg, metric_dict, dataset_dict):
        self.cfg = cfg
        self.metric_dict = metric_dict
        self.dataset_dict = dataset_dict
        self.argv = {}

    def get_fields(self):
        return self.argv

    def is_cfg_existed_and_correct(self):
        if not os.path.exists(self.cfg):
            OPT_ERROR(f"{self.cfg} config file not exists.")
            return False
        with open(self.cfg, 'r') as fi:
            lines = fi.readlines()
            lines = [l.strip() for l in lines]
            lines = [l for l in lines if len(l) != 0 and not l.startswith(('#', ',', ';', "//"))]
            if lines[0] != '[Common]':
                OPT_ERROR(
                    "Optimizer cfg file must have the '[Common]' section headers and '[Common]' must be in the first line.")
                return False
        return True

    def parser(self):
        import difflib

        # update default values for different targets
        raw_config = configparser.ConfigParser()
        raw_config.read(self.cfg)
        raw_pairs = raw_config['Common'] if 'Common' in raw_config else {}
        pk = 'min_compatible_zhouyi_target'
        ptarget = ALL_FIELDS[pk].default()
        if pk in raw_pairs:
            tg = raw_pairs[pk].split('_')[0].strip().upper()
            if Target.is_valid(tg):
                ptarget = tg
        defaults = {}
        for k, v in ALL_FIELDS.items():
            defaults.update({k: v.default()})
        if Target.optimized_target_level(ptarget) >= 1:
            # x2
            defaults['bias_effective_bits'] = str(Target.aiff_bias_effective_bits(ptarget))
            defaults['trigger_float_op'] = 'disable & <[RMSNorm]:float16_preferred!>'
        if Target.optimized_target_level(ptarget) >= 2:
            # defaults['trigger_float_op'] = 'disable & <[Softmax, MVN, LayerNorm, GroupNorm, InstanceNorm, RMSNorm]:float16_preferred!>'
            # x3
            defaults['enable_pass_tune_op_complicated_activations'] = '[0][1]'
            defaults['enable_pass_tune_op_softmax'] = '[0][1]'

        config = configparser.ConfigParser(defaults=defaults)
        config.read(self.cfg)
        if 'Common' in config:
            section = 'Common'
            options = config.options(section)
            common = config[section]
            for opt in options:
                opt_v = common[opt].replace(" ", "")

                if opt not in defaults.keys():
                    candidate = difflib.get_close_matches(opt, defaults.keys(), cutoff=0.6)
                    msg = f"Optimizer does not support the '{opt}' cfg field. "
                    if len(candidate):
                        msg += f"please confirm whether it is the following parameters:{candidate}."
                    else:
                        msg += f"please remove it from the cfg file."
                    OPT_WARN(msg)
                    continue

                if len(opt_v) == 0:
                    if len(ALL_FIELDS[opt].default()) != 0:
                        OPT_WARN(f"Optimizer uses the {opt}={ALL_FIELDS[opt].default()}(default value), "
                                 f"which only has the left value in cfg file.")
                    opt_v = defaults[opt]
                opt_v = string_to_base_type(opt_v)

                if opt in ALL_FIELDS.keys():
                    try:
                        field_obj = ALL_FIELDS[opt]
                        opt_flag, parsed_v = field_obj.parse(opt_v)
                        if not opt_flag:
                            OPT_FATAL(field_obj.error(opt_v))
                    except Exception as e:
                        OPT_ERROR(f"when checking the '{opt}' field in cfg file meets error.")
                        raise e
                # special field represtented by (flag, value, original_text)
                self.argv.update({opt: parsed_v})

            for k in ['model_name', 'output_dir', 'dump_dir', 'out_ir_name']:
                if isinstance(self.argv[k], (bool, int, float)):
                    self.argv[k] = str(self.argv[k])
            if self.argv.get('quant_ir_name', '') != '':
                self.argv['out_ir_name'] = str(self.argv['quant_ir_name'])
            if self.argv.get('out_ir_name', '') == '' or 'out_ir_name' not in self.argv:
                ir_name = self.argv['model_name'] + '_o'
                self.argv.update({'out_ir_name': ir_name})

        return self.argv

    def checker(self, argv):
        ret = True

        # dataset strategy
        '''
            1. if want to use dataset to calibrate and get statistic quantization values, the [dataset] field in cfg
                file must set for optimizer finding the right dataset plugin.
            2. if want to evaluate the model, the [metric] filed in cfg file must set for optimizer finding the right
                metric plugin.
            3. [data] [calibration_data] [label] fields in cfg file can not set, and then optimizer will transfer
                default value('') to dataset plugin, so custom will handle it by themselves.
        '''
        if argv['mixed_precision_auto_search'][0] > 0 and argv['metric'] == '':
            OPT_ERROR("please set 'metric' field in cfg file if want to enable mixed_precision_auto_search.")
            ret = ret and False

        if argv['metric'] != '':
            if argv['dataset'] == '':
                OPT_ERROR("please set 'dataset' field in cfg file if want to metric the model.")
                ret = ret and False
            if argv['data'] == '':
                OPT_ERROR("please set 'data' field in cfg file if want to metric the model.")
                ret = ret and False
            if argv['label'] == '':
                OPT_ERROR("please set 'label' field in cfg file if want to metric the model.")
                ret = ret and False

        if argv['statistic_file'] == '' and str(argv['calibration_strategy_for_activation']).lower() != 'in_ir':
            compat_quantized_model = ''
            if ALL_FIELDS['graph'].default() != argv['graph']:
                with open(argv['graph'], 'r') as f:
                    for line in f.readlines():
                        line = line.strip().replace('\n', '').replace(' ', '')
                        if 'compat_quantized_model=' in line:
                            compat_quantized_model = line.split('=')[-1].lower()
                            break

                if argv['calibration_data'] == '' and compat_quantized_model != 'true':
                    OPT_WARN(f"please set 'calibration_data' field in cfg file if want to statistic quantization values."
                             f" And Optimizer will use all zeros dataset for statistic tensors information.")
                    # ret = ret and True

        if argv['dataset'] == '' and argv['calibration_data'] != '':
            OPT_ERROR("please set 'dataset' field in cfg file if want to statistic quantization values")
            ret = ret and False

        if len(argv['global_calibration']) > 0 and argv['calibration_data'] == '':
            OPT_ERROR("please set 'calibration_data' field in cfg file if want to enable 'global_calibration' field")
            ret = ret and False

        if argv['opt_config'] != '':
            (OPT_INFO("please notice that Optimizer will use the configured parameter in opt_config file, when the "
                      "configuration parameter has conflicted between the opt_config file and the .cfg file."))

        if argv['data_batch_dim'] > 0 or argv['label_batch_dim'] > 0:
            if argv['dataset'] != '':
                (OPT_INFO("'data_batch_dim' or 'label_batch_dim' is greater than zero, please use 'data_batch_dim',"
                          " 'label_batch_dim' and implement collate_fn yourself in dataset plugin."))
                dataset_cls = self.dataset_dict[argv['dataset'].lower()]
                if dataset_cls is None:
                    OPT_WARN(f"currently 'dataset' in cfg file is not implemented,"
                             f" please implement the dataset plugin firstly.")
                else:
                    if not hasattr(dataset_cls, 'collate_fn'):
                        OPT_WARN(f"collate_fn is undefined in dataset plugin,"
                                 f" please implement collate_fn yourself to override the default collate_fn of Torch.")
        if argv['without_batch_dim'] and argv['modify_batch_dim']:
            OPT_ERROR("When without_batch_dim=true, can not enable modify_batch_dim")
            ret = ret and False
        if argv['without_batch_dim'] and argv['export_parallel_batch']:
            OPT_ERROR("When without_batch_dim=true, can not enable export_parallel_batch")
            ret = ret and False
        return ret

    def update(self):
        fields = self.get_fields()
        if fields is not None:
            for fk, fv in fields.items():
                self.__setattr__(fk, fv)
            return self
        return False

    def __call__(self):
        if not self.is_cfg_existed_and_correct():
            return False

        argv = self.parser()
        ret = self.checker(argv)

        return ret

    def __getattr__(self, item):
        if item in self.argv:
            if isinstance(self.argv[item], PerNodeFieldDict):
                return self.argv[item].global_value
            else:
                return self.argv[item]
        else:
            raise AttributeError


def show_plugins(metric_dict, dataset_dict):
    OPT_INFO('Plugin Dataset: %s' % (','.join(dataset_dict.keys())))
    OPT_INFO('Plugin Metric: %s' % (','.join(metric_dict.keys())))
    PLUGIN_OP = {'quantize_op': ALL_OPT_OP_DICT,
                 'forward_op': ALL_OPT_QUANT_OP_DICT}
    for key, o_dict in PLUGIN_OP.items():
        support_plugin_op = {}
        for optype, vals in o_dict.items():
            ver_list = []
            for version, val in vals.items():
                ver_list.append(version)
            support_plugin_op.update({str(optype)[7:]: ver_list})
        if len(support_plugin_op):
            OPT_INFO(f"Plugin {key}: {str(support_plugin_op)[1:-1]}")


def fields_to_str():
    os_str = 'Now Optimizer supports configurated parameters in .cfg file:\n'
    show_dict = {}
    for fscope, fields in DEFAULT_FIELDS.items():
        for key, val in fields.items():
            if key in ['graph', 'bin', 'model_name', ]:
                continue
            show_dict.update({key: val.message()})

    max_len = max([len(k) for k in show_dict.keys()])
    for k, v in show_dict.items():
        os_str += ('\t--{: <%d}  {: <}\n' % max_len).format(k, v)
    return os_str


def show_cfg_fields():
    os_str = fields_to_str()
    OPT_INFO(os_str)


def cfg_parser(cfg, metric, dataset):
    ret = False
    if len(cfg) != 0:
        _cfg_parser = CfgParser(cfg, metric, dataset)
        ret = _cfg_parser()
        if ret:
            ret = _cfg_parser.update()
    return ret


@opt_workflow_register
def arg_parser(metric_dict, dataset_dict):
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, required=False,
                        help='the config file.\n')
    parser.add_argument("-p", "--plugin", action='store_true',
                        help='show the dataset, metric and op plugins in Optimizer.\n')
    parser.add_argument("-f", "--field", action='store_true',
                        help='show the fields which can configure in config file.\n')

    argv = parser.parse_args()
    ret = True
    if argv.plugin:
        show_plugins(metric_dict, dataset_dict)

    if argv.field:
        show_cfg_fields()

    if argv.cfg:
        ret = cfg_parser(argv.cfg, metric_dict, dataset_dict)
        return ret

    return ret


def get_info_from_graph(graph, batch_dim=0):
    def get_batch_size(g):
        batch_size = None
        batch_size_list = []
        for inp_t in g.input_tensors:
            if len(inp_t.ir_shape) == 0:  # scalar input shape
                continue
            if len(inp_t.ir_shape) in [3, 4, 5]:
                batch_size = inp_t.ir_shape[batch_dim]
                break
            else:
                batch_size_list.append(inp_t.ir_shape[batch_dim])

        if batch_size is None:
            if batch_size_list:
                batch_size = max(batch_size_list, key=batch_size_list.count)
            else:
                batch_size = 1
                OPT_ERROR('please check the graph input tensors list, which is [].')

        return batch_size
    info_dict = dict()
    info_dict["batch_size"] = get_batch_size(graph)
    for node in graph.nodes:
        info_dict[node.name] = {
            "layer_id": str(node.attrs.get("layer_id", "-1"),)
        }
        info_dict[node.name]["layer_top_type_original"] = [str(t.ir_dtype) for t in node.outputs]
    return info_dict

# TODO


def filter_valid_properties(info, initial_info):
    return info
