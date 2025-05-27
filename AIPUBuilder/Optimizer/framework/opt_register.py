# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.


from enum import Enum, unique

__all__ = [
    'OP_DICT',
    'QUANT_OP_DICT',
    'APPROX_OP_DICT',
    'ALL_OPT_OP_DICT',
    'ALL_OPT_QUANT_OP_DICT',
    'ALL_OPT_APPROX_OP_DICT',
    'QUANTIZE_DATASET_DICT',
    'QUANTIZE_METRIC_DICT',
    'QUANTIZE_CONFIG_DICT',
    'op_register',
    'quant_register',
    'approx_register',
    'PluginType',
    'register_plugin',
    'traverse_opt_plugins',
    'OptBaseMetric',
]

OP_DICT = dict()
QUANT_OP_DICT = dict()
APPROX_OP_DICT = dict()

ALL_OPT_OP_DICT = dict()  # {optype: {version: [mfun, is_plugin]}}, which version is float, is_plugin is bool
ALL_OPT_QUANT_OP_DICT = dict()  # {optype: {version: [mfun, is_plugin]}}, which version is float, is_plugin is bool
ALL_OPT_APPROX_OP_DICT = dict()

QUANTIZE_DATASET_DICT = dict()
QUANTIZE_METRIC_DICT = dict()
QUANTIZE_CONFIG_DICT = dict()
TRAIN_PLUGIN_DICT = dict()


class OptBaseMetric(object):
    def __init__(self, *args):
        pass

    def __call__(self, pred, target, *args):
        raise NotImplementedError()

    def compute(self):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def report(self):
        raise NotImplementedError()


def is_plugin_op(filename):
    return True if 'aipubt_' in filename else False


def get_file_name():
    try:
        raise Exception
    except:
        import traceback
        fs = traceback.extract_stack()[-2]
        return fs.filename.split('/')[-1]


def _register(optypes, ver, mfun, all_op_dict, op_dict, is_plugin, register_type):
    from AIPUBuilder.Optimizer.logger import OPT_ERROR, OPT_WARN, OPT_INFO

    def tofloat(ver):
        split_ver = ver.strip(' ').split('.')
        float_val = float('.'.join([split_ver[0], ''.join(split_ver[1:])]))
        return float_val

    if isinstance(ver, str):
        cur_ver = tofloat(ver)
    elif isinstance(ver, (float, int)):
        cur_ver = ver
    else:
        cur_ver = 1.0
        OPT_WARN(
            register_type+': version of op when registering only support [str, float, int] type. and now we use verison=1.0 for running')

    is_register = True
    is_update_OP_DICT = True
    for optype in optypes:
        if optype in all_op_dict.keys():
            comp_ver = all_op_dict[optype].keys()
            max_ver = max(comp_ver)
            comp_plugin_stat = all_op_dict[optype][max_ver][-1]
            if cur_ver == max_ver:
                (OPT_WARN(register_type+': register op %s has the same version(%s) to the registered op, Optimizer suggestes to change the register version.'
                          % (optype, str(cur_ver))))
                if int(is_plugin) > int(comp_plugin_stat):
                    is_register = False
                    is_update_OP_DICT = False
            elif cur_ver < max_ver:
                is_update_OP_DICT = False

            if is_update_OP_DICT and is_plugin:
                OPT_INFO(register_type+': Optimizer will use the higher version(%s) of the plugin OP: %s.' % (str(cur_ver), optype))
        if is_register:
            cur_ver_func = {}
            cur_ver_func = {cur_ver: [mfun, is_plugin]}
            if optype in all_op_dict.keys():
                all_op_dict[optype].update(cur_ver_func)
            else:
                all_op_dict[optype] = cur_ver_func
        if is_update_OP_DICT:
            op_dict[optype] = mfun


def find_nested_func(parent, child_name):
    from types import CodeType, FunctionType
    consts = parent.__code__.co_consts
    for item in consts:
        if isinstance(item, CodeType) and item.co_name == child_name:
            return FunctionType(item, globals())
    return None


def op_register(optypes, version=1., *args):
    import time
    import torch
    global OP_DICT
    from AIPUBuilder.Optimizer.logger import OPT_ERROR
    if not isinstance(optypes, (tuple, list)):
        optypes = [optypes]

    def dec(func):
        def mfun(self, *args):
            try:
                if self.get_param('unquantifiable', optional=True, default_value=False):
                    self.quantized = False
                if len(args) > 0:
                    # back=[inp.betensor for inp in self.inputs]
                    assert(len(args) == len(self.inputs))
                    for inp, t in zip(self.inputs, args):
                        inp.betensor = t
                # set readonly keys
                # self.readonly_keys_set()
                # self.disable_keys_set()
                if 'calculate_running_time' not in self.attrs or not self.attrs['calculate_running_time']:
                    ret = func(self)
                else:
                    start_t = time.time()
                    ret = func(self)
                    self.attrs['cost_time'] = time.time() - start_t

                # free readonly keys
                # self.disable_keys_free()
                # self.readonly_keys_free()
                if ret is None:
                    ret = [t.betensor for t in self.outputs]
                if not isinstance(ret, torch.Tensor) and len(ret) == 1:
                    return ret[0]
            except Exception as e:
                OPT_ERROR(f"{self}: error message: {e.__repr__()}")
                raise e
            return ret
        is_plugin = is_plugin_op(get_file_name())
        _register(optypes, version, mfun, ALL_OPT_OP_DICT, OP_DICT, is_plugin, 'forward register')
        return mfun
    return dec


def quant_register(optypes, version=1.0, *args):
    from AIPUBuilder.Optimizer.logger import OPT_ERROR
    global QUANT_OP_DICT
    if not isinstance(optypes, (tuple, list)):
        optypes = [optypes]

    def dec(func):
        def mfunc(self, *args, **kwargs):
            try:
                unquantifiable = self.get_param('unquantifiable', optional=True, default_value=False)
                if self.quantized or unquantifiable:
                    return
                # set readonly keys
                # self.readonly_keys_set()
                ret = func(self, *args, **kwargs)
                # free readonly keys
                # self.readonly_keys_free()
                self.quantized = True
            except Exception as e:
                OPT_ERROR(f"{self}: error message: {e.__repr__()}")
                raise e
            return ret
        is_plugin = is_plugin_op(get_file_name())
        _register(optypes, version, mfunc, ALL_OPT_QUANT_OP_DICT, QUANT_OP_DICT, is_plugin, 'quantize register')
        return mfunc
    return dec


def approx_register(optypes, version=1.0, *args):
    from AIPUBuilder.Optimizer.logger import OPT_ERROR
    global APPROX_OP_DICT
    if not isinstance(optypes, (tuple, list)):
        optypes = [optypes]

    def dec(func):
        def mfunc(self, *args, **kwargs):
            try:
                unquantifiable = self.get_param('unquantifiable', optional=True, default_value=False)
                if self.approximated or (not unquantifiable):
                    return
                ret = func(self, *args, **kwargs)
                self.approximated = True
            except Exception as e:
                OPT_ERROR(f"{self}: error message: {e.__repr__()}")
                raise e
            return ret
        is_plugin = is_plugin_op(get_file_name())
        _register(optypes, version, mfunc, ALL_OPT_APPROX_OP_DICT, APPROX_OP_DICT, is_plugin, 'approximate register')
        return mfunc
    return dec


@unique
class PluginType(Enum):
    Dataset = 0x110
    Metric = 0x120
    QConfig = 0x130
    Train = 0x140


DM_PLUGINS = {t: dict() for t in PluginType}
DM_VERSIONS = {t: dict() for t in PluginType}


def register_plugin(type, version=0):
    from AIPUBuilder.Optimizer.logger import OPT_WARN, OPT_ERROR
    global DM_PLUGINS, DM_VERSIONS

    def tofloat(str_version):
        try:
            fval = float(str_version)
        except:
            split_version = str_version.strip(' ').split('.')
            fval = float('.'.join([split_version[0], ''.join(split_version[1:])]))
        return fval

    def wrapper(cls):
        if type == PluginType.Dataset:
            dataset_name = cls.__name__.lower()
            if dataset_name not in QUANTIZE_DATASET_DICT:
                QUANTIZE_DATASET_DICT[dataset_name] = cls
            else:
                prev_version = tofloat(DM_VERSIONS[type][cls.__name__])
                cur_version = tofloat(version)
                if prev_version < cur_version:
                    DM_VERSIONS[type][cls.__name__] = str(version)
                    QUANTIZE_DATASET_DICT[dataset_name] = cls
                    (OPT_WARN('this dataset plugin(version=%s) %s has existed, and Optimizer will use the higher version(%s) to replace.'
                              % (str(prev_version), dataset_name, str(version))))
        elif type == PluginType.Metric:
            metric_name = cls.__name__.lower()
            if metric_name not in QUANTIZE_METRIC_DICT:
                QUANTIZE_METRIC_DICT[metric_name] = cls
            else:
                prev_version = tofloat(DM_VERSIONS[type][cls.__name__])
                cur_version = tofloat(version)
                if prev_version < cur_version:
                    DM_VERSIONS[type][cls.__name__] = str(version)
                    QUANTIZE_METRIC_DICT[metric_name] = cls
                    (OPT_WARN('this metric plugin(version=%s) %s has existed, and Optimizer will use the higher version(%s) to replace.'
                              % (str(prev_version), metric_name, version)))
        elif type == PluginType.QConfig:
            qconfig_name = cls.__name__.lower()
            if qconfig_name not in QUANTIZE_CONFIG_DICT:
                QUANTIZE_CONFIG_DICT[qconfig_name] = cls
            else:
                OPT_WARN(f"{qconfig_name} has registered.")
        elif type == PluginType.Train:
            train_loop_name = cls.__name__.lower()
            if train_loop_name not in TRAIN_PLUGIN_DICT:
                TRAIN_PLUGIN_DICT[train_loop_name] = cls
            else:
                OPT_WARN(f"{train_loop_name} has registered.")
        else:
            OPT_ERROR(f"Unsupported plugin type = {type}")
        if cls.__name__ not in DM_PLUGINS[type]:
            DM_PLUGINS[type][cls.__name__] = cls
            DM_VERSIONS[type][cls.__name__] = str(version)
        return cls
    return wrapper


def traverse_opt_plugins():
    import importlib
    import pkgutil
    import sys
    import os
    PLUGIN_PREFIX = "aipubt_"
    AIPUPLUGIN_PATH = os.environ.get("AIPUPLUGIN_PATH", "").split(":")
    AIPUPLUGIN_PATH = [i for i in set(AIPUPLUGIN_PATH+["./plugin", "."]) if len(i) != 0]
    sys.path = AIPUPLUGIN_PATH+sys.path

    _MODULES = {
        name: importlib.import_module(name)
        for finder, name, ispkg
        in pkgutil.iter_modules(AIPUPLUGIN_PATH)
        if name.startswith(PLUGIN_PREFIX)
    }
