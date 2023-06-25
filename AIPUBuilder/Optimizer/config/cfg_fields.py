# Copyright © 2023 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import re
from AIPUBuilder.Optimizer.logger import OPT_ERROR, OPT_INFO, OPT_WARN
from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder.Optimizer.framework import QUANTIZE_METRIC_DICT, QUANTIZE_DATASET_DICT, OpType

DEFAULT_FIELDS = {}
HIDDEN_FIELDS = {}


def field_register(field, ftype='default'):
    def wrapper(cls):
        if ftype.lower() == 'default':
            DEFAULT_FIELDS.update({field: cls})

        if ftype.lower() == 'hidden':
            HIDDEN_FIELDS.update({field: cls})
        return cls
    return wrapper


class BaseField(object):
    @staticmethod
    def default(*args):
        raise NotImplementedError()

    @staticmethod
    def check(*args):
        raise NotImplementedError()

    @staticmethod
    def error(*args):
        raise NotImplementedError()

    @staticmethod
    def message(*args):
        raise NotImplementedError()


@field_register('graph', 'default')
class GraphField(BaseField):
    @staticmethod
    def default():
        return ''

    @staticmethod
    def check(g):
        return True if not os.path.isfile(g) else False

    @staticmethod
    def error(g):
        return f"Require existed graph path, now graph path is {g}."

    @staticmethod
    def message():
        return "Graph description file path of Compass IR."


@field_register('bin', 'default')
class BinField(BaseField):
    @staticmethod
    def default():
        return ''

    @staticmethod
    def check(b):
        return True if not os.path.isfile(b) else False

    @staticmethod
    def error(b):
        return f"Require existed bin path, now graph path is {b}."

    @staticmethod
    def message():
        return "Binary file path of Compass IR."


@field_register('model_name', 'default')
class ModelNameField(BaseField):
    @staticmethod
    def default():
        return ''

    @staticmethod
    def check(m):
        return True if m == '' else False

    @staticmethod
    def error(m):
        return f"Require the non-empty 'model_name' field, now mode_name is '' char."

    @staticmethod
    def message():
        return f"A name for distinguishing different models."


@field_register('quant_ir_name', 'default')
class QuantIRNameField(BaseField):
    @staticmethod
    def default():
        return ''

    @staticmethod
    def check(q):
        return True if q == '' else False

    @staticmethod
    def error(q):
        return (f"Suggest the non-empty 'quant_ir_name' field, if quant_ir_name='', "
                f"Optimizer will use the <model_name> for quant IR.")

    @staticmethod
    def message():
        return f"A name for Compass quantized IR name."


@field_register('output_dir', 'default')
class OutputDirField(BaseField):
    # path to save the optimized IR and other log files
    @staticmethod
    def default():
        return './opt_log'

    @staticmethod
    def check(od):
        return False

    @staticmethod
    def error(od):
        return f"Now 'output_dir={od}' does not existed. Optimizer will create the path."

    @staticmethod
    def message():
        return f"A path to save the quantized IR, calibration statistic file and quantization configuration json file."


@field_register('dataset', 'default')
class DatasetField(BaseField):
    @staticmethod
    def _dataset_plugins():
        datasets = [k for k in QUANTIZE_DATASET_DICT.keys()]
        return datasets

    @staticmethod
    def default():
        return ''

    @staticmethod
    def check(d):
        if isinstance(d, list):
            return True
        _plugins = str([k for k in DatasetField._dataset_plugins()] + [''])
        if d.lower() not in _plugins:
            return True
        return False

    @staticmethod
    def error(d):
        if isinstance(d, (tuple, list)):
            return f"Optimizer now does not support multi-dataset, and now dataset = {d}"
        _plugins = str([k for k in DatasetField._dataset_plugins()] + [''])
        if d not in _plugins:
            return f"Require the 'dataset' field must be in {_plugins}, and now dataset = {d}."

    @staticmethod
    def message():
        return f"A dataset plugin name uses to create a Dataset. Now Optimizer supports dataset plugin names: {DatasetField._dataset_plugins()}"


@field_register('metric', 'default')
class MetricField(BaseField):
    @staticmethod
    def _metric_plugins():
        metrics = [k for k in QUANTIZE_METRIC_DICT.keys()]
        return metrics

    @staticmethod
    def _split_metrics(m):
        check = r'^(\w+\(?(\/?\w+\.?\w*,?)*\)?,?)+$'
        cpattern = re.compile(check)
        if len(re.findall(r'[^\w|\\|/|\.|,|\)|\()]', m)) > 0 or cpattern.match(m) is None:
            OPT_ERROR((f"metric format:'metricname(args,...)/metricname()/metricname'; "
                       f"'metricname'and 'args' should be [a-zA-Z0-9_]"))
            return False
        rule = r'\w+\(?(\/?\w+\.?\w*,?)*\)?,?'
        pattern = re.compile(rule)
        rmatch = pattern.match(m)
        metrics = []
        newm = m
        while rmatch is not None:
            metrics.append(rmatch.group())
            newm = newm[rmatch.span()[1]:]
            rmatch = pattern.match(newm)
        return metrics

    @staticmethod
    def _get_func_args(ms):
        func_args = []
        regEx = re.compile('[,;]')
        for m in ms:
            args = []
            if set({'(', ')'}).issubset(set(m)):
                func = m[:m.index('(')]
                argsl = m[m.index('(')+1:m.index(')')]
                args = regEx.split(argsl)
                args = [b for b in args if b != '']
            else:
                func = regEx.split(m)[0]
            func_args.append([func, args])

        return func_args

    @staticmethod
    def default():
        return ''

    @staticmethod
    def check(m):
        if not isinstance(m, str):
            return True
        m = m.replace(' ', '')
        if m != '':
            if m.count('(') != m.count(')'):
                return True
            metrics = MetricField._split_metrics(m)
            if isinstance(metrics, bool) and metrics == False:
                return True
            func_args = MetricField._get_func_args(metrics)
            mnames = [fa[0].lower() for fa in func_args]
        else:
            mnames = m
        if set(mnames).issubset(set(MetricField._metric_plugins() + [''])):
            return False
        return True

    @staticmethod
    def error(m):
        def _checker(m):
            check = '^(\w+\((\w+,?)*\),?|\w+\(\),?|\w+,?)+$'
            cpattern = re.compile(check)
            cmatch = cpattern.match(m)
            if cmatch is None:
                return True
            return False

        msg = ''
        if m.count('(') != m.count(')'):
            msg += f"The num of '(' is not same to the num of ')', please check the 'metric' field. "
        elif _checker(m):
            msg += f"Please check the 'metric' field format. "
        else:
            msg += f"Require the 'metric' field must be in {MetricField._metric_plugins()}, "
        msg += f"now 'metric={m}'"
        return msg

    @staticmethod
    def message():
        return (f"A metric plugin name for metric one model. It supports multi-metric and uses ',' to seperate. "
                f"Now Optimizer supports metric plugin name is {MetricField._metric_plugins()}.")


@field_register('eval_original_model', 'default')
class EvalOriginalModelField(BaseField):
    # whether run evaluation on original (float) model
    @staticmethod
    def default():
        return 'True'

    @staticmethod
    def check(e):
        return True if not isinstance(e, bool) else False

    @staticmethod
    def error(e):
        return f"Require the 'eval_original_model' field must be in bool type, now is {type(e)} type. default value=True"

    @staticmethod
    def message():
        return f"Whether run evaluation on original (float) model."


@field_register('data', 'default')
class DataField(BaseField):
    # the npy label file for the evaluation dataset
    @staticmethod
    def default():
        return ''

    @staticmethod
    def check(d):
        return True if not (os.path.isfile(d) or os.path.exists(d) or d == '') else False

    @staticmethod
    def error(d):
        return f"Require the existed 'data' path, now {d} does not exist."

    @staticmethod
    def message():
        return f"A dataset path for metric a model."


@field_register('label', 'default')
class LabelField(BaseField):
    # the npy label file for the evaluation dataset
    @staticmethod
    def default():
        return ''

    @staticmethod
    def check(l):
        return True if not (os.path.isfile(l) or os.path.exists(l) or l == '') else False

    @staticmethod
    def error(l):
        return f"Require the existed 'label' path, now {l} does not exist."

    @staticmethod
    def message():
        return f"A label path for metric a model."


@field_register('data_batch_dim', 'default')
class DataBatchDimField(BaseField):
    @staticmethod
    def default():
        return '0'

    @staticmethod
    def check(bd):
        return True if not (isinstance(bd, int) and bd >= 0) else False

    @staticmethod
    def error(bd):
        if bd > 0:
            return (f"'data_batch_dim'(={bd}) is greater than zero, "
                    f"please use 'data_batch_dim' and implement collate_fn yourself in dataset plugin.")
        else:
            return f"Required the positive integer(>=0) 'data_batch_dim' field, default value=0."

    @staticmethod
    def message():
        return (f"The default value of 'data_batch_dim' is zero, if 'data_batch_dim' is greater than zero, "
                f"please use 'data_batch_dim' and implement collate_fn yourself in dataset plugin.")


@field_register('label_batch_dim', 'default')
class LabelBatchDimField(BaseField):
    @staticmethod
    def default():
        return '0'

    @staticmethod
    def check(bd):
        return True if not (isinstance(bd, int) and bd >= 0) else False

    @staticmethod
    def error(bd):
        if bd > 0:
            return (f"'label_batch_dim'(={bd}) is greater than zero, "
                    f"please use 'label_batch_dim' and implement collate_fn yourself in dataset plugin.")
        else:
            return f"Required the positive integer(>=0) 'data_batch_dim' field, default value=0."

    @staticmethod
    def message():
        return (f"The default value of 'label_batch_dim' is zero, if 'label_batch_dim' is greater than zero, "
                f"please use 'label_batch_dim' and implement collate_fn yourself in dataset plugin.")


@field_register('without_batch_dim', 'default')
class WithoutBatchDimField(BaseField):
    @staticmethod
    def default():
        return 'False'

    @staticmethod
    def check(wobd):
        return True if not isinstance(wobd, bool) else False

    @staticmethod
    def error(wobd):
        return f"Require the 'without_batch_dim' field must be in bool type, now is {type(wobd)} type. default value=False."

    @staticmethod
    def message():
        return f"Whether the model has batch dim or not."


@field_register('dataloader_workers', 'default')
class DataloaderWorkersField(BaseField):
    # number of workers for dataloader
    @staticmethod
    def default():
        return '0'

    @staticmethod
    def check(dw):
        return True if not (isinstance(dw, int) and dw >= 0) else False

    @staticmethod
    def error(dw):
        msg = dw if isinstance(dw, int) else type(dw)
        return f"Required the nonnegative integer(>= 0) 'dataloader_workers' field, now is {msg}. default value=0."

    @staticmethod
    def message():
        return f"Thread workers for DataLoader."


@field_register('calibration_batch_size', 'default')
class CalibrationBatchSizeField(BaseField):
    # the batch size used for computing quantization parameters over calibration dataset
    @staticmethod
    def default():
        return '1'

    @staticmethod
    def check(cbz):
        return True if not (isinstance(cbz, int) and cbz > 0) else False

    @staticmethod
    def error(cbz):
        msg = cbz if isinstance(cbz, int) else type(cbz)
        return f"Required the positive integer(>0) 'calibration_batch_size' field, now is {msg}. default value=1."

    @staticmethod
    def message():
        return f"Batch size when calibrating a model."


@field_register('metric_batch_size', 'default')
class MetricBatchSizeField(BaseField):
    # the batch size used for evaluation dataset
    @staticmethod
    def default():
        return '1'

    @staticmethod
    def check(mbz):
        return True if not (isinstance(mbz, int) and mbz > 0) else False

    @staticmethod
    def error(mbz):
        msg = mbz if isinstance(mbz, int) else type(mbz)
        return f"Required the positive integer(>0) 'metric_batch_size' field, now is {msg}. default value=1."

    @staticmethod
    def message():
        return f"Batch size when metricing a model."


@field_register('dump', 'default')
class DumpField(BaseField):
    # whether enable dump
    @staticmethod
    def default():
        return 'False'

    @staticmethod
    def check(dp):
        return True if not isinstance(dp, bool) else False

    @staticmethod
    def error(dp):
        return f"Require the 'dump' field must be in bool type, now is {type(dp)} type. default value=False."

    @staticmethod
    def message():
        return f"Whether enable to dump all tensors and other data."


@field_register('dump_dir', 'default')
class DumpDirField(BaseField):
    # the directory to dump
    @staticmethod
    def default():
        return './opt_dump'

    @staticmethod
    def check(dd):
        return False

    @staticmethod
    def error(dd):
        return f"Now 'dump_dir={dd}' does not existed. Optimizer will create the path."

    @staticmethod
    def message():
        return f"Dump all data files will save to this path when 'dump=true'."


@field_register('opt_config', 'default')
class OptConfigField(BaseField):
    # use to store per-layer configurations, more useful co-work with GUI.
    @staticmethod
    def default():
        return ''

    @staticmethod
    def check(oc):
        return True if not (os.path.isfile(oc) or oc == '') else False

    @staticmethod
    def error(oc):
        return f"Require to give the existed 'opt_config' file, now 'opt_config={oc}' does not exist."

    @staticmethod
    def message():
        return f"A file stores quantization configurations of each node."


@field_register('statistic_file', 'default')
class StatisticFileField(BaseField):
    # statistic information file computed with this tool before, load this file can save some time juring quantization.
    @staticmethod
    def default():
        return ''

    @staticmethod
    def check(sf):
        return True if not (os.path.isfile(sf) or sf == '') else False

    @staticmethod
    def error(sf):
        return f"Require the existed 'statistic_file' file, now 'statistic_file={sf}' does not exist."

    @staticmethod
    def message():
        return f"A file stores calibration statistic information of each tensor in each node."


class CalibrationStrategyField(BaseField):
    # activation calibration method like 'mean', 'extrema', 'Nstd', 'kld', etc.
    @staticmethod
    def _check_calibration_method(cm):
        if re.match(r'^\d+std$', cm.lower()):
            return False
        elif re.match(r'^\d*kld$', cm.lower()):
            return False
        elif re.match(r'^(\d\.?\d*)*aciq_laplace$', cm.lower()):
            return False
        elif re.match(r'^(\d\.?\d*)*aciq_gauss$', cm.lower()):
            return False
        elif re.match(r'^(\d\.?\d*)*percentile$', cm.lower()):
            return False
        elif re.match(r'^weighted_scale_param\[[0-9]+\.?[0-9]*\s*,\s*[0-9]+\.?[0-9]*\s*,\s*[0-9]+\.?[0-9]*\s*,\s*[0-9]+\.?[0-9]*\s*\,?\s*]$', cm.lower()):
            return False
        elif cm.lower() in ['extrema', 'mean', 'in_ir']:
            return False
        else:
            return True

    @staticmethod
    def _need_statistic_info(cm):
        r = {'histc': True, 'std_mean': True}
        if re.match(r'^\d+std$', cm.lower()):
            r['histc'] = False
        elif re.match(r'^\d*kld$', cm.lower()):
            r['std_mean'] = False
        elif re.match(r'^(\d\.?\d*)*aciq_laplace$', cm.lower()):
            r['histc'] = False
        elif re.match(r'^(\d\.?\d*)*aciq_gauss$', cm.lower()):
            r['histc'] = False
        elif re.match(r'^weighted_scale_param\[[0-9]+\.?[0-9]*\s*,\s*[0-9]+\.?[0-9]*\s*,\s*[0-9]+\.?[0-9]*\s*,\s*[0-9]+\.?[0-9]*\s*\,?\s*]$', cm.lower()):
            r['histc'] = False
        elif cm.lower() in ['extrema', 'mean', 'in_ir'] or re.match(r'^(\d\.?\d*)*percentile$', cm.lower()):
            r['histc'] = False
            r['std_mean'] = False
        return r

    @staticmethod
    def _calibration_method():
        _calib_methods = [
            'extrema',
            'Nstd',
            'in_ir',
            '[N]kld',
            '[R]aciq_laplace',
            '[R]aciq_gauss',
            '[R]percentile',
            'weighted_scale_param[x1.y1,x2.y2,x3.y3,x4.y4]',
        ]
        return _calib_methods

    @staticmethod
    def default():
        return "extrema"

    @staticmethod
    def check(cw):
        return CalibrationStrategyField._check_calibration_method(cw)

    @staticmethod
    def error(cs):
        pass

    @staticmethod
    def message():
        pass


@field_register('calibration_strategy_for_weight', 'default')
class CalibrationStrategyForWeightField(CalibrationStrategyField):
    @staticmethod
    def default():
        return 'extrema'

    @staticmethod
    def check(cw):
        return CalibrationStrategyField.check(cw)

    @staticmethod
    def error(cw):
        msg = (f"Optimizer supports 'calibration_method={CalibrationStrategyForWeightField._calibration_method()}', Now calibration method = '{cw}' in cfg file. "
               f"The 'N' in 'Nstd' means one positive integer, like '2std'. "
               f"The '[N]' in '[N]kld' means none or one positive integer, like 'kld' or '2kld'. "
               f"The '[R]' in '[R]aciq_laplace', '[R]aciq_gauss' and '[R]percentile' means none or one positive real number, like 'aciq_laplace' or '1.5aciq_laplace'. "
               f"The pattern of 'weighted_scale_param' is like weighted_scale_param[0.1, 0.2, 0.3, 0.4]."
               )
        return msg

    @staticmethod
    def message():
        return f"'Weight calibration strategy. Now Optimizer supports weight calibration strategy: {CalibrationStrategyForWeightField._calibration_method()}."


@field_register('calibration_strategy_for_activation', 'default')
class CalibrationStrategyForActivationField(CalibrationStrategyField):
    # activation calibration method like 'mean', 'extrema', 'Nstd', 'kld', etc.
    @staticmethod
    def _calibration_method():
        _act_calib_methods = CalibrationStrategyField._calibration_method() + ['mean']
        return _act_calib_methods

    @staticmethod
    def default():
        return 'mean'

    @staticmethod
    def check(ca):
        return CalibrationStrategyField.check(ca)

    @staticmethod
    def error(ca):
        msg = (f"Optimizer supports 'calibration_method={CalibrationStrategyForActivationField._calibration_method()}', Now calibration method = '{ca}' in cfg file. "
               f"The 'N' in 'Nstd' means one positive integer, like '2std'. "
               f"The '[N]' in '[N]kld' means none or one positive integer, like 'kld' or '2kld'. "
               f"The '[R]' in '[R]aciq_laplace', '[R]aciq_gauss' and '[R]percentile' means none or one positive real number, like 'aciq_laplace' or '1.5aciq_laplace'. "
               )
        return msg

    @staticmethod
    def message():
        return f"Activation calibration strategy. Now Optimizer supports activation calibration strategy: {CalibrationStrategyForActivationField._calibration_method()}"


@field_register('global_calibration', 'default')
class GlobalCalibrationParamField(BaseField):
    # global calibration method like 'easy_quant', 'ada_round', ... , default is None, means no global calibration.
    @staticmethod
    def _parse(gc):
        rfloat = r'\s*(\-|\+)?\d+((\.\d+)|\.)?\s*'
        roptype = r'\s*[a-zA-Z_0-9]+\s*'
        rparams = r'\s*\[{}(,{})*\]\s*'.format(rfloat, rfloat)
        roptypes = r'\s*\[{}(,{})*\]\s*'.format(roptype, roptype)
        rscope = r'\s*\(\s*\d+\s*,\s*\d+\s*\)\s*'
        rlayers = r'\s*\[{}(,{})*\]\s*'.format(rscope, rscope)
        rmethod = r'\s*((svd_quant)|(easy_quant)|(mvn_correction)|(adaround)|(adaquant_zy))\s*'
        rmethod_param = r'{}{}'.format(rmethod, rparams)
        rmethod_param_optypes = r'{}{}{}'.format(rmethod, rparams, roptypes)
        rmethod_param_layers = r'{}{}{}'.format(rmethod, rparams, rlayers)
        one_method = r'\s*(({})|({})|({})|({}))\s*'.format(rmethod, rmethod_param,
                                                           rmethod_param_optypes, rmethod_param_layers)
        multi_methods = r'^{}(&{})*$'.format(one_method, one_method)
        gcstr = str(gc).lower().strip()
        if 'none' == gcstr:
            return (True, [])
        elif re.match(multi_methods, gcstr):
            valid_methods = []
            for mstr in [x.lower().strip() for x in re.split(r'&', gcstr) if x.lower().strip()]:
                name = mstr + ' '
                name_idx = name.find('[')
                name = name[:name_idx].strip()
                vec = []
                oplist = []
                if name_idx > 0:
                    param_end = mstr.find(']')
                    param_list = re.split(r',|\[|\]|\(|\)|\s+', mstr[len(name):param_end])
                    vec = [float(param) for param in param_list if param.lower().strip()]
                    ol_str = mstr[param_end+1:]
                    if re.match(rmethod_param_optypes, mstr):
                        oplist = [o.lower().strip() for o in re.split(
                            r',|\[|\]|\(|\)|\s+', ol_str) if o.lower().strip()]
                    elif re.match(rmethod_param_layers, mstr):
                        idx_list = re.split(r',|\[|\]|\(|\)|\s+', ol_str)
                        layer_ranges = [int(idx) for idx in idx_list if idx.lower().strip()]
                        for k in range(0, len(layer_ranges), 2):
                            for l in range(layer_ranges[k], layer_ranges[k+1]):
                                oplist.append(l)
                    else:
                        pass
                valid_methods.append((name, vec, set(oplist)))
            return (True, valid_methods)
        else:
            return (False, [])

    @staticmethod
    def default():
        return 'none'

    @staticmethod
    def message():
        return "Global calibration (scale/zero_point/rounding optimization) method, now supports: 'none', \
            'easy_quant' (refer to https://arxiv.org/pdf/2006.16669.pdf), \
            'easy_quant[batches, epochs, alpha, beta, nsteps, ngroups]', \
            'easy_quant[batches, epochs, alpha, beta, nsteps, ngroups][operator_type1, operator_type2, ...]', \
            'easy_quant[batches, epochs, alpha, beta, nsteps, ngroups][(layer_i, layer_j), (layer_k, layer_l), ...]', \
            'adaround' (refer to https://arxiv.org/pdf/2004.10568.pdf), \
            'adaround[batches, epochs, batch_size, lr, reg_param, beta_start, beta_end, warm_start]', \
            'adaround[batches, epochs, batch_size, lr, reg_param, beta_start, beta_end, warm_start][operator_type1, operator_type2, ...]', \
            'adaround[batches, epochs, batch_size, lr, reg_param, beta_start, beta_end, warm_start][(layer_i, layer_j), (layer_k, layer_l), ...]', \
            'adaquant_zy' (refer to https://arxiv.org/pdf/2006.10518.pdf) \
            'adaquant_zy[batches, epochs, batch_size, lr_weight, lr_bias, lr_quant_param_weight, lr_quant_param_activation]' \
            'adaquant_zy[batches, epochs, batch_size, lr_weight, lr_bias, lr_quant_param_weight, lr_quant_param_activation][operator_type1, operator_type2, ...]' \
            'adaquant_zy[batches, epochs, batch_size, lr_weight, lr_bias, lr_quant_param_weight, lr_quant_param_activation][(layer_i, layer_j), (layer_k, layer_l), ...]' \
            'mvn_correction' (arm china),\
            'mvn_correction[mode, alpha]' ,\
            'mvn_correction[mode, alpha][operator_type1, operator_type2, ...]' ,\
            'mvn_correction[mode, alpha][(layer_i, layer_j), (layer_k, layer_l), ...]' ,\
            'svd_quant' (arm china), \
            'svd_quant[mode, alpha, beta, nsteps, thresh][operator_type1, operator_type2, ...]', \
            'svd_quant[mode, alpha, beta, nsteps, thresh][(layer_i, layer_j), (layer_k, layer_l), ...]'. \
            Where 'operator_type1' and 'operator_type2' are valid operator type names that specify the operators which will be applied, \
            'layer_i', 'layer_j', 'layer_k' and ''layer_l' stand for layer_id in input IR and '(layer_i, layer_j), (layer_k, layer_l)' specify the layers which will be applied,  \
            'batches' means how many (calibartion) data batches will be used, 'epochs' means the maximum epochs if not convergence, \
            'lr' means the learning rate, 'ngroups' means groups which will be divided into when meeting per-channel quantization parameters to speed up (0 means no speed up), and the 'alpha','beta', 'nsteps', etc are the float type configurable inner hyper-parameters for corresponding methods, \
            'none' means do nothing, default to 'none'. \
            You can also apply multiple methods sequentially with `&`, e.g. `adaround[10, 3, 32] & easy_quant`. "

    @staticmethod
    def check(gc):
        m = gc.lower()
        if 'none' == m:
            return False
        elif GlobalCalibrationParamField._parse(m)[0]:
            return False
        else:
            return True

    @staticmethod
    def error(gc):
        return GlobalCalibrationParamField.message() + f" now is {gc}"


@field_register('calibration_data', 'default')
class CalibrationDataField(BaseField):
    # the npy data file for the calibration dataset
    @staticmethod
    def default():
        return ''

    @staticmethod
    def check(cd):
        return True if not (os.path.isfile(cd) or os.path.exists(cd) or cd == '') else False

    @staticmethod
    def error(cd):
        return f"Require the existed 'calibration_data' path, now {cd} does not exist."

    @staticmethod
    def message():
        return f"A dataset path for calibrating a model."


@field_register('calibration_shuffe', 'default')
class CalibrationShuffleField(BaseField):
    # whether shuffe juring computing quantization parameters over calibration dataset
    @staticmethod
    def default():
        return 'False'

    @staticmethod
    def check(cs):
        return True if not isinstance(cs, bool) else False

    @staticmethod
    def error(cs):
        return f"Require the 'calibration_shuffle' field must be in bool type, now is {type(cs)} type. default value=False."

    @staticmethod
    def message():
        return f"Whether to apply shuffle over calibration dataset."


@field_register('quantize_method_for_weight', 'default')
class QuantizeMethodForWeightField(BaseField):
    # quantization method for weights and biases, like 'per_tensor_symmetric_restricted_range, per_channel_symmetric_restricted_range'
    @staticmethod
    def _weight_quantize_method():
        return list(filter(QuantMode.is_symmetric, QuantMode.mode_names()))

    @staticmethod
    def default():
        return 'per_tensor_symmetric_restricted_range'

    @staticmethod
    def check(qmw):
        _weight_qmethod = QuantizeMethodForWeightField._weight_quantize_method()
        return True if qmw.lower() not in _weight_qmethod else False

    @staticmethod
    def error(qmw):
        _weight_qmethod = QuantizeMethodForWeightField._weight_quantize_method()
        return f"Require the 'quantize_method_for_weight' field must be in {_weight_qmethod}, now 'quantize_method_for_weight={qmw}'."

    @staticmethod
    def message():
        _weight_qmethod = QuantizeMethodForWeightField._weight_quantize_method()
        return f"Weight quantization method. Now Optimizer supports quantzation method: {_weight_qmethod}"


@field_register('quantize_method_for_activation', 'default')
class QuantizeMethodForActivationField(BaseField):
    # quantization method for activations like 'per_tensor_symmetric_full_range, per_tensor_asymmetric'
    @staticmethod
    def _activation_quantize_method():
        return list(filter(QuantMode.is_per_tensor, QuantMode.mode_names()))

    @staticmethod
    def default():
        return 'per_tensor_symmetric_full_range'

    @staticmethod
    def check(qma):
        _activation_qmethod = QuantizeMethodForActivationField._activation_quantize_method()
        return True if qma.lower() not in _activation_qmethod else False

    @staticmethod
    def error(qma):
        _activation_qmethod = QuantizeMethodForActivationField._activation_quantize_method()
        return f"Require the 'quantize_method_for_activation' field must be in {_activation_qmethod}, now 'quantize_method_for_activation={qma}'."

    @staticmethod
    def message():
        _activation_qmethod = QuantizeMethodForActivationField._activation_quantize_method()
        return f"Activation quantization method. Now Optimizer supports quantzation method: {_activation_qmethod}"


@field_register('weight_bits', 'default')
class WeightBitsField(BaseField):
    # quantization precision for weights, like '8, 16', default to '8'
    @staticmethod
    def _weight_bits():
        return list(range(4, 17))

    @staticmethod
    def default():
        return '8'

    @staticmethod
    def check(wb):
        _wbits = WeightBitsField._weight_bits()
        return True if not (isinstance(wb, int) and wb in _wbits) else False

    @staticmethod
    def error(wb):
        _wbits = WeightBitsField._weight_bits()
        return f"Required the integer 'weight_bits' field and must be in {_wbits}, now is {type(wb)} type, {wb}, default value=8."

    @staticmethod
    def message():
        _wbits = WeightBitsField._weight_bits()
        return f"Weight bits for quantizating weight data. Now Optimizer supports weight bits:{_wbits}."


@field_register('activation_bits', 'default')
class ActivationBitsField(BaseField):
    # quantization precision for activations, like '8, 16', default to '8'
    @staticmethod
    def _activation_bits():
        return list(range(8, 17))

    @staticmethod
    def default():
        return '8'

    @staticmethod
    def check(ab):
        _abits = ActivationBitsField._activation_bits()
        return True if not (isinstance(ab, int) and ab in _abits) else False

    @staticmethod
    def error(ab):
        _abits = ActivationBitsField._activation_bits()
        return f"Required the integer 'activation_bits' field and must be in {_abits}, now is {type(ab)} type, {ab}, default value=8."

    @staticmethod
    def message():
        _abits = ActivationBitsField._activation_bits()
        return f"Activation bits for quantizating activation data. Now Optimizer supports activation bits:{_abits}."


@field_register('lut_items_in_bits', 'default')
class LutItemsInBitsField(BaseField):
    # maximal lut items (in bits, as only support lut with 2**N items) amount
    # when representing nonlinear functions in quantization, default to '8'
    @staticmethod
    def _activation_bits():
        return list(range(8, 17))

    @staticmethod
    def default():
        return '8'

    @staticmethod
    def check(lb):
        _abits = LutItemsInBitsField._activation_bits()
        return True if not (isinstance(lb, int) and lb in _abits) else False

    @staticmethod
    def error(lb):
        _abits = LutItemsInBitsField._activation_bits()
        return f"Required the integer 'lut_items_in_bits' field and must be in {_abits}, now is {type(lb)} type, {lb},default value=8."

    @staticmethod
    def message():
        _abits = LutItemsInBitsField._activation_bits()
        return f"Maximal LUT items (in bits, as only support LUT with 2**N items) amount when representing nonlinear functions in quantization, default to '8', suggest to set to 10+ when quantizing activations to 16bit. Now Optimizer supports: {_abits} ."


@field_register('bias_bits', 'default')
class BiasBitsField(BaseField):
    # quantization precision for bias, like '16, 32', default to '32'
    @staticmethod
    def _bias_bits():
        return list(range(16, 49))

    @staticmethod
    def default():
        return '32'

    @staticmethod
    def check(bb):
        _bbits = BiasBitsField._bias_bits()
        return True if not (isinstance(bb, int) and bb in _bbits) else False

    @staticmethod
    def error(bb):
        _bbits = BiasBitsField._bias_bits()
        return f"Required the integer 'bias_bits' field and must be in {_bbits}, now is {type(bb)} type, {bb}, default value=32."

    @staticmethod
    def message():
        _bbits = BiasBitsField._bias_bits()
        return f"Bias bits for quantizating bias data. Now Optimizer supports bias bits:{_bbits} ."


@field_register('bias_effective_bits', 'default')
class BiasEffectiveBitsField(BaseField):
    @staticmethod
    def _bias_bits():
        return list(range(16, 49))

    @staticmethod
    def default():
        return ''

    @staticmethod
    def check(bb):
        _bbits = BiasEffectiveBitsField._bias_bits()
        return True if not (bb == '' or (isinstance(bb, int) and bb in _bbits)) else False

    @staticmethod
    def error(bb):
        _bbits = BiasEffectiveBitsField._bias_bits()
        return (f"The 'bias_effective_bits' field must be in {_bbits} or equal to '', "
                f"now is {type(bb)} type, {bb}, default value=''.")

    @staticmethod
    def message():
        return (f"The effective high bits for bias data which realy taking part "
                f"in computation (lower bits will be set to 0), due to hardware restriction. ")


@field_register('unify_shifts_for_aiff', 'default')
class UnifyShiftsForAIFFField(BaseField):
    @staticmethod
    def default():
        return 'True'

    @staticmethod
    def check(us):
        return True if not isinstance(us, bool) else False

    @staticmethod
    def error(us):
        return f"Require the 'unify_shifts_for_aiff' field must be in bool type, now is {type(us)} type, default value=True"

    @staticmethod
    def message():
        return f"Whether to unify shifts for AIFF OPs due to hardware limitations."


@field_register('cast_dtypes_for_lib', 'default')
class CastDtypesForLibField(BaseField):
    @staticmethod
    def default():
        return 'False'

    @staticmethod
    def check(cd):
        return False

    @staticmethod
    def error(cd):
        return f"Require the 'cast_dtypes_for_lib' field be 'False' or 'True' or list of valid operator type names (case insensitive, corresponding to layer_type in float IR), e.g., Abs,reshape,tile"

    @staticmethod
    def message():
        return (f"Whether to cast dtypes of OPs to adapt to lib's dtypes' spec (may cause model accuracy loss "
                f"due to corresponding spec's restriction). 'False' means no. 'True' means yes to all OPs. "
                f"A list of valid operator type names (case insensitive, corresponding to layer_type in float IR),"
                f" e.g., Abs,reshape,tile means yes to all the specified OPs.")


@field_register('force_dtype_int', 'default')
class ForceDtypeIntField(BaseField):
    @staticmethod
    def default():
        return 'False'

    @staticmethod
    def check(fdi):
        return True if not isinstance(fdi, bool) else False

    @staticmethod
    def error(fdi):
        return f"Require the 'force_dtype_int' field must be in bool type, now is {type(fdi)} type, default value=False."

    @staticmethod
    def message():
        return f"Whether force layer top tensors to be quantized to int types (may cause accuracy drop or be rejected by lib's restriction) instead of being decided automatically."


@field_register('force_shift_positive', 'default')
class ForceShiftPositiveField(BaseField):
    @staticmethod
    def default():
        return 'False'

    @staticmethod
    def check(fsp):
        return True if not isinstance(fsp, bool) else False

    @staticmethod
    def error(fsp):
        return f"Require the 'force_shift_positive' field must be in bool type, now is {type(fsp)} type, default value=False."

    @staticmethod
    def message():
        return f"Whether to force each layer's requantization parameter 'shift' to be positive (due to hardware's limitation, accuracy drop may occurs)."


@field_register('running_statistic_momentum', 'default')
class RunningStatisticMomentumField(BaseField):
    # momentum when statistic running statistics
    @staticmethod
    def default():
        return '0.9'

    @staticmethod
    def check(rsm):
        return True if not isinstance(rsm, (int, float)) else False

    @staticmethod
    def error(rsm):
        return f"Required the float 'running_statistic_momentum' field, and suggest the range of 'running_statistic_momentum' is [0., 1.]. now is {type(rsm)} type. default value=0.9."

    @staticmethod
    def message():
        return f"Momentum(range[0.0, 1.0]) used for calculating weighted average of some statistics when calibration dataset has multiple batches."


@field_register('histc_bins', 'default')
class HistcBinsField(BaseField):
    # bins when statistic histograms
    @staticmethod
    def default():
        return '2048'

    @staticmethod
    def check(hb):
        return True if not (isinstance(hb, int) and hb > 0) else False

    @staticmethod
    def error(hb):
        return f"Required the positive integer(>0) 'histc_bins' field, now is {type(hb)} type and 'histc_bins={hb}'. default value=2048."

    @staticmethod
    def message():
        return f"Bins when statistic histograms of each tensor."


@field_register('set_qinvariant_for_start_nodes', 'default')
class SetQinvariantForStartNodesField(BaseField):
    # bins when statistic histograms
    @staticmethod
    def default():
        return ''

    @staticmethod
    def check(sqs):
        if '' == sqs:
            return False
        elif re.match(r'^\s*\d+(\s*,\s*\d+)*\s*,?\s*$', str(sqs)):
            return False
        else:
            return True

    @staticmethod
    def error(sqs):
        return (f"Require the 'set_qinvariant_for_start_nodes' field must be a list of unsigned integer(corresponding"
                f" to layer_id in float IR) like '0' or '0,2' or '0,1,2', now is: {sqs}")

    @staticmethod
    def message():
        return (f"Point out those start nodes (input layers or constant layers) "
                f"which should always have 'scale=1.0, zero_point=0' when being quantized (e.g. tensors represent indexes). "
                f"Must be a list of unsigned integer(corresponding to layer_id in float IR) like '0' or '0,2' or '0,1,2'.")


@field_register('fake_quant_scopes_for_debug', 'default')
class FakeQuantScopesForDebugField(BaseField):
    # fake_quant_scopes_for_debug which will run float forward in related layers
    # in assigned scopes when calling quanted forward for debug usage.
    @staticmethod
    def default():
        return ''

    @staticmethod
    def check(fqs):
        if '' == fqs:
            return False
        elif re.match(r'^\s*\[(\s*\(\s*\d+\s*,\s*\d+\s*\)\s*,?\s*)+\]\s*$', fqs):
            return False
        else:
            return True

    @staticmethod
    def error(fqs):
        return (f"Require the 'fake_quant_scopes_for_debug' field must be a list of unsigned integer(corresponding"
                f" to layer_id in float IR) tuples like '[(i,j)]' or '[(i,j),(n,m),...,(k,l)]', now is: {fqs}")

    @staticmethod
    def message():
        return (f"The 'fake_quant_scopes_for_debug' means to run float forward in related layers in assigned scopes"
                f" when calling quanted forward for debug usage. Must be a list of unsigned integer "
                f"(corresponding to layer_id in float IR) tuples like '[(i,j)]' or '[(i,j),(n,m),...,(k,l)].")


@field_register('fake_quant_operators_for_debug', 'default')
class FakeQuantOperatorsForDebugField(BaseField):
    @staticmethod
    def default():
        return ''

    @staticmethod
    def check(fqod):
        return False

    @staticmethod
    def error(fqod):
        return (f"Require the 'fake_quant_operators_for_debug' field must be a list of valid operator type names "
                f"(case insensitive, corresponding to layer_type in float IR), e.g., Abs,reshape,tile.")

    @staticmethod
    def message():
        return (f"The 'fake_quant_operators_for_debug' meanses to run float forward in related layers "
                f"in assigned operators when calling quanted forward for debug usage. Must be a list of valid "
                f"operator type names (case insensitive, corresponding to layer_type in float IR), "
                f"e.g., Abs,reshape,tile")


@field_register('resize_degrade_to_nearest', 'default')
class ResizeDegradeToNearestField(BaseField):
    # whether degrade resize method to nearest neighbor to speed up resize
    @staticmethod
    def default():
        return 'False'

    @staticmethod
    def check(rdn):
        return True if not isinstance(rdn, bool) else False

    @staticmethod
    def error(rdn):
        return f"Require the 'resize_degrade_to_nearest' field must be in bool type, now is {type(rdn)} type, default value=False."

    @staticmethod
    def message():
        return f"Whether to degrade resize method to nearest neighbor to speed up resize."


@field_register('unify_scales_for_concat', 'default')
class UnifyScalesForConcatField(BaseField):
    # whether unify scales of each concat's branch, it may loss accuracy but good for speed.
    @staticmethod
    def default():
        return 'False'

    @staticmethod
    def check(usc):
        return True if not isinstance(usc, bool) else False

    @staticmethod
    def error(usc):
        return f"Require the 'unify_scales_for_concat' field must be in bool type, now is {type(usc)} type, default value=False."

    @staticmethod
    def message():
        return f"Whether to unify scales of each concat's branch when possible."


@field_register('unify_scales_for_concat_threshold', 'default')
class UnifyScalesForConcatThresholdField(BaseField):
    # when enable unify_scales_for_concat, you can set the threshold for branch scales.
    @staticmethod
    def default():
        return '1.05'

    @staticmethod
    def check(usct):
        return True if not (isinstance(usct, float) and usct >= 1.0) else False

    @staticmethod
    def error(usct):
        return f"Require the 'unify_scales_for_concat_threshold' >= 1.0, now is {usct}, default value=1.05."

    @staticmethod
    def message():
        return (f"For the concat operator, if max(branch_scales) / min(branch_scales) <= threshold,"
                f" then will ignore the branch scales' difference. defalut value=1.05")


@field_register('unify_scales_for_concat_max_depth', 'default')
class UnifyScalesForConcatMaxDepthField(BaseField):
    # when enable unify_scales_for_concat, you can set the search detph in the graph.
    @staticmethod
    def default():
        return '20'

    @staticmethod
    def check(uscmd):
        return True if not (isinstance(uscmd, int) and uscmd > 0) else False

    @staticmethod
    def error(uscmd):
        return f"Require the positive integer 'unify_scales_for_concat_max_depth' field, now is {uscmd}, default value=20."

    @staticmethod
    def message():
        return f"When enable unify_scales_for_concat(=True), max search depth can be setted for speeding up, defalut value=20."


@field_register('unify_scales_for_multi_inputs_operators', 'default')
class UnifyScales4MultiInputsOP(BaseField):
    # bins when statistic histograms
    @staticmethod
    def default():
        return ''

    @staticmethod
    def check(sqs):
        if '' == sqs:
            return False
        elif re.match(r'^\s*\[(\s*\(\s*[a-zA-Z_0-9]+\s*,\s*\d+\s*,\s*\d+((\.\d+)|\.)?\s*,\s*((max)|(min)|(avg)|(out))\s*,?\s*\)\s*,?\s*)+\]\s*$', str(sqs).lower().strip()):
            return False
        else:
            return True

    @staticmethod
    def parse(sqs):
        if UnifyScales4MultiInputsOP.check(sqs):
            return {}
        else:
            r = {}
            s = [x.lower().strip() for x in re.split(r',|\[|\]|\(|\)|\s+', str(sqs).lower().strip()) if x.lower().strip()]
            for i in range(0, len(s), 4):
                op_name = s[i]
                depth = int(s[i+1])
                thres = float(s[i+2])
                method = s[i+3]
                for k, v in OpType.__dict__.items():
                    if str(k).lower().strip() == op_name:
                        r[v] = (depth, thres, method)
            return r

    @staticmethod
    def error(sqs):
        return (f"Require the 'unify_scales_for_multi_inputs_operators' field must be like [(operator_type, search_depth, threshold, method), ...], "
                f"where 'operator_type' is valid operator type name, 'search_depth' is max search depth for speeding up (0 means no limits), 'threshold' means "
                f"ignoring the branch scales' difference when max(branch_scales) / min(branch_scales) <= threshold, 'method' defines the strategy "
                f"for choosing the unified scale and currently support 'max' (maximum scale of input branches), 'min' (minimum scale of input branches), 'avg' (mean scale of input branches), 'out' (scale of this layer's 1st output)."
                f"e.g. '[(Concat, 20, 1.05, min)]', '[(ScatterND, 20, 1.05, min)]', '[(Concat, 20, 1.05, min), (ScatterND, 20, 1.05, min)]'. "
                f"Now is: {sqs}")

    @staticmethod
    def message():
        return (f"The 'unify_scales_for_multi_inputs_operators' field means whether to unify scales of input branches for assigned operators "
                f"when possible, it's a generalized extension of the 'unify_scales_for_concat' field."
                f"Must be like [(operator_type, search_depth, threshold, method), ...], "
                f"where 'operator_type' is valid operator type name, 'search_depth' is max search depth for speeding up (0 means no limits), 'threshold' means "
                f"ignoring the branch scales' difference when max(branch_scales) / min(branch_scales) <= threshold, 'method' defines the strategy "
                f"for choosing the unified scale and currently support 'max' (maximum scale of input branches), 'min' (minimum scale of input branches), 'avg' (mean scale of input branches), 'out' (scale of this layer's 1st output)."
                f"e.g. '[(Concat, 20, 1.05, min)]', '[(ScatterND, 20, 1.05, min)]', '[(Concat, 20, 1.05, min), (ScatterND, 20, 1.05, min)]'. ")


@field_register('with_winograd', 'default')
class WithWinogradField(BaseField):
    #  Winograd Part: whether to enable winograd algorithm.
    @staticmethod
    def default():
        return 'False'

    @staticmethod
    def check(ww):
        return True if not isinstance(ww, bool) else False

    @staticmethod
    def error(ww):
        return f"Require the 'with_winograd' field must be in bool type, now is {type(ww)} type, default value=False."

    @staticmethod
    def message():
        return f"Whether to enable winograd algorithm when possible."


@field_register('mixed_precision_auto_search', 'default')
class MixedPrecisionAutoSearchField(BaseField):
    @staticmethod
    def default():
        return '0,0.,L'

    @staticmethod
    def check(mpas):
        return True if not re.match(r'^\s*[0-9]+\s*,\s*-?[0-9]+\.?[0-9]*\s*,\s*[G|g|L|l]\s*$', mpas) else False

    @staticmethod
    def error(mpas):
        return (f"Required float number 'mixed_precision_auto_search' field be set as "
                f"'batches,threshold,greater_equal_or_less_equal_than_baseline' (batches < 1 means disable), "
                f"like '1,0.02,L'(use 1 batch data to metric, and score(baseline model) - score(target model) <= 0.02)"
                f" or '1,-0.02,G'(use 1 batch data to metric, and "
                f"score(baseline model) - score(target model) >= -0.02). default value='0,0.,L'.")

    @staticmethod
    def message():
        return (f"Give the maximum allowed accuracy loss threshold, automatically recommend"
                f" which layers should be quantized to 8bit, which layers should be quantized to 16bit, "
                f"which layers should not be quantized. Must be set as 'batches,threshold,"
                f"greater_equal_or_less_equal_than_baseline' (batches < 1 means disable), like '1,0.02,L'"
                f"(use 1 batch data to metric, and score(baseline model) - score(target model) <= 0.02) or '1,-0.02,G'"
                f"(use 1 batch data to metric, and score(baseline model) - score(target model) >= -0.02)."
                f"default value='0,0.,L'.")


@field_register('featuremap_tiling_param', 'hidden')
class FeaturemapTilingParamField(BaseField):
    # check featuremap split parameter by end user
    # [(start_node end_node slice_h,slice_w)]
    @staticmethod
    def default():
        return ''

    @staticmethod
    def check(ftp):
        if '' == ftp:
            return False
        # layout is [(start_node end_node slice_h,slice_w)]
        elif re.match(r'^\s*\[(\s*\(\s*\d+\s*,\s*\d+\s*\,\s*\d+\s*\,\s*\d+\s*\)\s*,?\s*)+\]\s*$', ftp):
            return False
        else:
            return True

    @staticmethod
    def error(ftp):
        return (f"Require the 'featuremap_tiling_param' field must be a list of unsigned integer (corresponding "
                f"to layer_id in float IR) tuples like '[(i,j,h,w)]' or '[(i,j,h,w),...,(k,l,x,y)]', now is: {ftp}")

    @staticmethod
    def message():
        return f"the featuremap_tiling_param means to assign specific (corresponding to layer_id in float IR) layers'  tiling configurations: '[(i,j,h,w)]' or '[(i,j,h,w),...,(k,l,x,y)]', where 'i,j' and 'k,l' means the corresponding layers that will be tiled and 'h,w' and 'x,y' means the parts amount that the featuremap will be split in corresponding dimension."


@field_register('featuremap_splits_item_x', 'hidden')
class FeaturemapSplitsItemXField(BaseField):
    # data partition, like '2, 2', default to '1'
    @staticmethod
    def _item_num():
        return list(range(1, 128))

    @staticmethod
    def default():
        return '1'

    @staticmethod
    def check(ix):
        _item_n = FeaturemapSplitsItemXField._item_num()
        return True if not (isinstance(ix, int) and ix in _item_n) else False

    @staticmethod
    def error(ix):
        _item_n = FeaturemapSplitsItemXField._item_num()
        return f"Required the integer 'featuremap_splits_item_x' field and must be in {_item_n}, now is {type(ix)} type, {ix}, default value=1."

    @staticmethod
    def message():
        _item_n = FeaturemapSplitsItemXField._item_num()
        return f"Item number for featuremap data parallel in x dimension. Now Optimizer supports item num: {_item_n}."


@field_register('featuremap_splits_item_y', 'hidden')
class FeaturemapSplitsItemYField(BaseField):
    # data partition, like '2, 2', default to '1'
    @staticmethod
    def _item_num():
        return list(range(1, 128))

    @staticmethod
    def default():
        return '1'

    @staticmethod
    def check(iy):
        _item_n = FeaturemapSplitsItemYField._item_num()
        return True if not (isinstance(iy, int) and iy in _item_n) else False

    @staticmethod
    def error(iy):
        _item_n = FeaturemapSplitsItemYField._item_num()
        return f"Required the integer 'featuremap_splits_item_y' field and must be in {_item_n}, now is {type(iy)} type, {iy}, default value=1."

    @staticmethod
    def message():
        _item_n = FeaturemapSplitsItemYField._item_num()
        return f"Item number for featuremap data parallel in y dimension. Now Optimizer supports item num: {_item_n}."


@field_register('featuremap_splits_concat_blk', 'hidden')
class FeaturemapSplitsConcatBlkField(BaseField):
    @staticmethod
    def _item_num():
        return list(range(1, 128))

    @staticmethod
    def default():
        return '3'

    @staticmethod
    def check(num):
        _item_n = FeaturemapSplitsConcatBlkField._item_num()
        return True if not (isinstance(num, int) and num in _item_n) else False

    @staticmethod
    def error(num):
        _item_n = FeaturemapSplitsConcatBlkField._item_num()
        return f"Required the integer 'featuremap_splits_concat_blk' field and must be in {_item_n}, now is {type(num)} type, {num}, default value=3."

    @staticmethod
    def message():
        _item_n = FeaturemapSplitsConcatBlkField._item_num()
        return f"Split big concat node (which has too many input branches) into series of sub-concat (has amount of 'featuremap_splits_concat_blk' inputs) nodes when possible."


@field_register('featuremap_splits_overlap_rate', 'hidden')
class FeaturemapSplitsOverlapRateField(BaseField):
    @staticmethod
    def _overlap_rate():
        return list(range(0, 100))

    @staticmethod
    def default():
        return '50'

    @staticmethod
    def check(rate):
        _or = FeaturemapSplitsOverlapRateField._overlap_rate()
        return True if rate < _or[0] or rate > _or[-1] else False

    @staticmethod
    def error(rate):
        _or = FeaturemapSplitsOverlapRateField._overlap_rate()
        return f"Required the integer 'featuremap_splits_overlap_rate' field and must be in {_or}, now is {type(rate)} type, {rate}, default value=50."

    @staticmethod
    def message():
        _or = FeaturemapSplitsOverlapRateField._overlap_rate()
        return f"Maximum allowed overlap rate for featuremap data parallel. Now Optimizer supports overlap rate in [0, 100), default value=50."


@field_register('featuremap_splits_sram_size', 'hidden')
class FeaturemapSplitsSramSizeField(BaseField):
    @staticmethod
    def _sram_size():
        return list(range(0, 8192))

    @staticmethod
    def default():
        return '0'

    @staticmethod
    def check(ss):
        _ssize = FeaturemapSplitsSramSizeField._sram_size()
        return True if not(isinstance(ss, int) and ss in _ssize) else False

    @staticmethod
    def error(ss):
        _ssize = FeaturemapSplitsSramSizeField._sram_size()
        return f"Required the integer 'featuremap_splits_sram_size' field and must be in [{_ssize[0]}, {_ssize[-1]}], now is {ss} type, {ss}. default value=0 k."

    @staticmethod
    def message():
        _ssize = FeaturemapSplitsSramSizeField._sram_size()
        return f"Maximum allowed sram size for reducing memory footprint by tiling featuremaps. Now Optimizer supports sram size: {_ssize[0]}~{_ssize[-1]} k."


@field_register('scaling_bits', 'hidden')
class ScalingBitsField(BaseField):
    '''
    decodebox: [box_bits, box_num_perclass_bits, total_class_num_bits, label_perclass_bits],
    Interp: [interp_shift],
    LRN: [lut_out_bits],
    NMS: [iou_thresh_shift, box_shift, areas_shift, inter_shift],
    Postnms1: [nor_box_shift],
    PRELU: [negative_slope_shift],
    Region: [conf_sigmoid_shift, score_softmax_dtype_bits, anchor_dtype_bits, conf_sigmoid_dtype_bits,
            bbox_xy_sigmoid_dtype_bits, bbox_wh_exp_dtype_bits, grid_shift],
    Resize:[interp_shift],
    RioAlign: [quant_bits, spatial_shift],
    ROIPooling: [index_precision],
    Softmax: [max_value_scaling_bits],
    '''
    @staticmethod
    def _scaling_bits_default():
        return "{ \
                  DecodeBox: [16, 16, 16, 16], \
                  Interp: [13], \
                  LRN: [16], \
                  NMS: [8, 15, 13, 13], \
                  PostNms1: [10], \
                  PRELU: [12], \
                  Region: [15, 32, 16, 16, 16, 16, 15], \
                  Resize: [13], \
                  RoiAlign: [12, 10], \
                  ROIPooling: [16], \
                  Softmax: [20], \
                }".replace(' ', '')

    @staticmethod
    def default():
        return ScalingBitsField._scaling_bits_default()

    @staticmethod
    def message():
        _sbd = ScalingBitsField._scaling_bits_default()
        return (f"The OPs, like {_sbd} has implicit quantization parameters, "
                f"and you can set these parameters by 'scaling_bits' fields.")

    @staticmethod
    def _match(scaling_bits):
        scaling_bits = scaling_bits.replace(' ', '')
        check = "^\{((\w+:\[((\-|\+)?\d+(\.\d+)?,?)*\])*,?)*\}$"
        cpattern = re.compile(check)
        cmatch = cpattern.match(scaling_bits)
        return cmatch, cpattern

    @staticmethod
    def check(scaling_bits):
        cmatch, _ = ScalingBitsField._match(scaling_bits)
        return True if cmatch is None else False

    @staticmethod
    def _get_scaling_bits(scaling_bits):
        scaling_bits_dict = {}
        scaling_bits = scaling_bits.replace(' ', '')
        cmatch, cpattern = ScalingBitsField._match(scaling_bits)
        # example: gm_str = '{resize:[10], nms:[1,2,2]}'
        gm_str = cmatch.group()
        # delete {}
        gm_str = gm_str[1:-1]
        # pad ',' to end if end is ']', so we can use '],' to split.
        gm_str = gm_str+',' if gm_str[-1] == ']' else gm_str
        split_gms = gm_str.split('],')
        split_gms = [s for s in split_gms if s != '']
        for sgms in split_gms:
            key, values = sgms.split(':')
            if len(values) > 1:
                bits = [eval(b) for b in re.findall('\d+', values)]
                scaling_bits_dict.update({key.lower(): bits})
        return scaling_bits_dict

    @staticmethod
    def error(scaling_bits):
        msg = ''
        if scaling_bits.count('{') != scaling_bits.count('}') or scaling_bits.count('[') != scaling_bits.count(']'):
            msg += "The num of '{' or '[' is not same to the num of '}' or ']'."
        elif ScalingBitsField.check(scaling_bits):
            msg += f"Please check the 'scaling_bits' field format, which should be like a dict format: " \
                   f"scaling_bits={{key:[value], key1:[value1]}}, " \
                   f"and 'key' should be op_type or a method in Activation Op; " \
                   f"'value' should be one integer/float or a set of integer/float using ',' to seperate."
        else:
            pass
        msg += " Now scaling_bits=%s" % scaling_bits
        return msg

    @staticmethod
    def _update_to_node_attr(node, scaling_bits):
        default_scaling_bits = ScalingBitsField._get_scaling_bits(
            ScalingBitsField._scaling_bits_default())
        parsed_scaling_bits = ScalingBitsField._get_scaling_bits(scaling_bits)
        all_scaling_bits = {**default_scaling_bits, **parsed_scaling_bits}
        if str(node.type)[7:].lower() in all_scaling_bits:
            scaling_bits_key = str(node.type)[7:].lower()
            if scaling_bits_key in default_scaling_bits and scaling_bits_key in parsed_scaling_bits:
                if len(default_scaling_bits[scaling_bits_key]) != len(parsed_scaling_bits[scaling_bits_key]):
                    OPT_WARN('%s is set risky length of scaling_bits.' % str(node.type))
            node.attrs['scaling_bits'] = all_scaling_bits[scaling_bits_key]
        elif 'method' in node.params and node.params['method'].lower() in all_scaling_bits:
            scaling_bits_key = node.params['method'].lower()
            node.attrs['scaling_bits'] = all_scaling_bits[scaling_bits_key]
        else:
            pass


@field_register('trigger_float_op', 'default')
class TriggerFloatOpField(BaseField):
    @staticmethod
    def _trigger_float_op():
        return ['disable', 'automatic', 'float16_preferred', 'bfloat16_preferred', 'float32_preferred']

    @staticmethod
    def default():
        return 'disable'

    @staticmethod
    def check(tfo):
        _trg_fp_op = TriggerFloatOpField._trigger_float_op()
        return True if not (isinstance(tfo, str) and tfo.lower() in _trg_fp_op) else False

    @staticmethod
    def error(tfo):
        _trg_fp_op = TriggerFloatOpField._trigger_float_op()
        return f"The `trigger_float_op` filed requires the string type, and must be in {_trg_fp_op}."

    @staticmethod
    def message():
        _trg_fp_op = TriggerFloatOpField._trigger_float_op()
        return f"The `trigger_float_op` is used for activating layer lib's float implementation when being enabled, and the configurable options is: {_trg_fp_op}. Where `automatic` means the output IR will automatically employ float type layers according to libs' dtype spec (the calibration dataset is still needed, as probably not all of the model's operators do have float type layer lib implementation and quantization will be applied under such circumstances), 'float16_preferred', 'bfloat16_preferred' and 'float32_preferred' means corresponding float type will be selected preferentially if existed."


@field_register('save_statistic_info', 'default')
class SaveStatisticInfoField(BaseField):
    @staticmethod
    def default():
        return 'False'

    @staticmethod
    def check(ww):
        return True if not isinstance(ww, bool) else False

    @staticmethod
    def error(ww):
        return f"Require the 'save_statistic_info' field must be in bool type, now is {type(ww)} type, default value=False."

    @staticmethod
    def message():
        return f"Whether to save and dump the statisticed information file, if set false, will further just statistic information which is necessary for corresponding calibration strategy for time saving."


@field_register('trim_infinity_before_statistic', 'default')
class TrimInfinityField(BaseField):
    @staticmethod
    def default():
        return ''

    @staticmethod
    def check(ti):
        conf = str(ti).lower().strip()
        range_method = r'\s*\(\s*(((\-|\+)?\d+((\.\d+)|\.)?)|(\-inf))\s*,\s*(((\-|\+)?\d+((\.\d+)|\.)?)|(inf))\s*\)\s*:\s*((clip)|(second))\s*'
        op_range_method = r'\s*[a-zA-Z_0-9]+\s*:{}'.format(range_method)
        id_range_method = r'\s*\(\s*\d+\s*,\s*\d+\s*\)\s*:{}'.format(range_method)
        if '' == ti:
            return False
        elif re.match(r'^{}$'.format(range_method), conf):
            return False
        elif re.match(r'^\s*\[\s*{}\s*(\s*,\s*({}))*\]\s*$'.format(op_range_method, op_range_method), conf):
            return False
        elif re.match(r'^\s*\[\s*{}\s*(\s*,\s*({}))*\]\s*$'.format(id_range_method, id_range_method), conf):
            return False
        else:
            return True

    @staticmethod
    def parse(ti):
        conf = str(ti).lower().strip()
        if len(conf) > 0 and (not TrimInfinityField.check(conf)):
            y = [x.lower().strip() for x in re.split(r',|\[|\]|\(|\)|:|\s+', conf) if x.lower().strip()]
            if conf[0] == '(':
                return 1, ((float(y[0]), float(y[1])), y[-1])
            elif conf[0] == '[':
                r = {}
                if conf[1:].strip()[0] == '(':
                    for i in range(0, len(y), 5):
                        key = (int(y[i]), int(y[i+1]))
                        val = ((float(y[i+2]), float(y[i+3])), y[i+4])
                        for k in range(key[0], key[1]+1):
                            r[k] = val
                    return 3, r
                else:
                    for i in range(0, len(y), 4):
                        key = y[i]
                        val = ((float(y[i+1]), float(y[i+2])), y[i+3])
                        r[key] = val
                    return 2, r
            else:
                pass
        return 0, {}

    @staticmethod
    def error(ti):
        return (f"For uniform configuration, must be `(min_value, max_value):method`. "
                f"For specific configuration, must be `[operator_type:(min_value, max_value):method, ...]` or `[(i, j):(min_value, max_value):method, ...]`. "
                f"Where values <= `min_value` or >= `max_value` will be treated as infinite values (need: min_value <= 0 <= max_value), `method` decides how to deal with infinite values and currently "
                f"supports `clip` (infinite values will be clamped into [min_value, max_value]) and `second` (infinite values will be replaced by min/max values "
                f"after excluding infinite values), `operator_type` is valid operator type name, `i` and `j` stand for layer_id in input IR. "
                f"Valid examples: `(-inf, inf):second`, `(-65536, 65535):clip`, `[Abs:(0, inf):second, BatchNorm:(-32767, 65535):clip]`, `[(1,10):(-inf,inf):second, (20,30):(-32767,32767):clip]`. "
                f"to layer_id in float IR) tuples like '[(i,j,h,w)]' or '[(i,j,h,w),...,(k,l,x,y)]'. "
                f"Now is: {ti}")

    @staticmethod
    def message():
        return (f"Exclude the infinite or equivalent very large/small values from statistic. For uniform configuration, must be `(min_value, max_value):method`. "
                f"For specific configuration, must be `[operator_type:(min_value, max_value):method, ...]` or `[(i, j):(min_value, max_value):method, ...]`. "
                f"Where values <= `min_value` or >= `max_value` will be treated as infinite values (need: min_value <= 0 <= max_value), `method` decides how to deal with infinite values and currently "
                f"supports `clip` (infinite values will be clamped into [min_value, max_value]) and `second` (infinite values will be replaced by min/max values "
                f"after excluding infinite values), `operator_type` is valid operator type name, `i` and `j` stand for layer_id in input IR. "
                f"Valid examples: `(-inf, inf):second`, `(-65536, 65535):clip`, `[Abs:(0, inf):second, BatchNorm:(-32767, 65535):clip]`, `[(1,10):(-inf,inf):second, (20,30):(-32767,32767):clip]`. "
                f"Disabled by default.")


@field_register('similarity_data_num', 'hidden')
class SimilarityDataNumField(BaseField):
    @staticmethod
    def default():
        return '1'

    @staticmethod
    def check(sdn):
        return True if not (isinstance(sdn, int) and sdn > 0) else False

    @staticmethod
    def error(sdn):
        return f"Required the positive integer(>0) 'similarity_data_num' field, now is {type(sdn)} type and {sdn}, default value=1."

    @staticmethod
    def message():
        return f"The batches amount for checking similarity."


@field_register('run_mode', 'hidden')
class RunModeField(BaseField):
    @staticmethod
    def _run_mode():
        return ['float_ir_forward', 'quant_ir_forward', 'mixed_ir_forward', 'default']

    @staticmethod
    def default():
        return 'default'

    @staticmethod
    def check(rm):
        _rmode = RunModeField._run_mode()
        return True if not (isinstance(rm, str) and rm.lower() in _rmode) else False

    @staticmethod
    def error(rm):
        _rmode = RunModeField._run_mode()
        return f"Required string type, and only support {_rmode}, default value=default."

    @staticmethod
    def message():
        return f"Run mode for Optimizer."


@field_register('record_debug_acc_info', 'hidden')
class RecordDebugAccInfoField(BaseField):
    @staticmethod
    def default():
        return 'False'

    @staticmethod
    def check(rdai):
        return True if not isinstance(rdai, bool) else False

    @staticmethod
    def error(rdai):
        return f"Require the 'record_debug_acc_info' field must be in bool type, now is {type(rdai)} type. default value=False."

    @staticmethod
    def message():
        return f"Whether to record debug acc info to record_model_acc.txt file."


@field_register('compat_quantized_model_eliminate_cast', 'hidden')
class CompatQuantizedModelEliminateCastField(BaseField):
    @staticmethod
    def default():
        return 'True'

    @staticmethod
    def check(ec):
        return True if not isinstance(ec, bool) else False

    @staticmethod
    def error(ec):
        return f"Require the 'compat_quantized_model_eliminate_cast' field must be in bool type, now is {type(ec)} type. default value=False."

    @staticmethod
    def message():
        return f"Whether to eliminate cast op when compat_quantized_model=true."


@field_register('compat_quantized_model', 'hidden')
class CompatQuantizedModelFeild(BaseField):
    @staticmethod
    def default():
        return 'False'

    @staticmethod
    def check(cqm):
        return True if not isinstance(cqm, bool) else False

    @staticmethod
    def error(cqm):
        return f"Require the 'compat_quantized_model' field must be bool type, now is {type(cqm)}, default=False."

    @staticmethod
    def message():
        return (f"Will transform from thirdparty quantization mode (currently only tflite's schema is supported) to Zhouyi AIPU's mode when set true.")


@field_register('compat_quantized_model_strategy', 'hidden')
class CompatQuantizedModelStrategyFeild(BaseField):
    _strategies = ['8bSymWeightUnchange', '8bAsymWeightTo16bSymWeight']
    _extra_strategies = ['sym', 'asym', 'fasymp']
    _all_strategies = _strategies + _extra_strategies
    _strategies_lower = [_s.lower() for _s in _all_strategies]

    @staticmethod
    def default():
        return '8bSymWeightUnchange'

    @staticmethod
    def check(csfqm):
        return False if csfqm.lower() in CompatQuantizedModelStrategyFeild._strategies_lower else True

    @staticmethod
    def error(csfqm):
        return (f"Require the 'compat_quantized_model_strategy' field must in {CompatQuantizedModelStrategyFeild._strategies}, "
                f"now is {csfqm}, default={CompatQuantizedModelStrategyFeild.default()}.")

    @staticmethod
    def message():
        return f"When compat_quantized_model = true, the compat_quantized_model_strategy can be choose from {CompatQuantizedModelStrategyFeild._strategies}, default to {CompatQuantizedModelStrategyFeild.default()}."


@field_register('compat_quantized_model_multiplier_bits', 'hidden')
class CompatQuantizedModelMultiplierBitsField(BaseField):
    @staticmethod
    def _multiplier_bits():
        return list(range(2, 16))

    @staticmethod
    def default():
        return '13'

    @staticmethod
    def check(mb):
        _abits = CompatQuantizedModelMultiplierBitsField._multiplier_bits()
        return True if not (isinstance(mb, int) and mb in _abits) else False

    @staticmethod
    def error(mb):
        _abits = CompatQuantizedModelMultiplierBitsField._multiplier_bits()
        return f"Required the integer 'compat_quantized_model_multiplier_bits' field and must be in {_abits}, now is {type(mb)} type, {mb}."

    @staticmethod
    def message():
        _abits = CompatQuantizedModelMultiplierBitsField._multiplier_bits()
        return f"The bits used to represent 'M' when applying 'scale = M / (2**N)'. Now Optimizer supports: {_abits}."


@field_register('compat_quantized_model_int8_to_uint8', 'hidden')
class CompatQuantizedModelInt8ToUint8Field(BaseField):
    @staticmethod
    def default():
        return 'False'

    @staticmethod
    def check(itou):
        return True if not isinstance(itou, bool) else False

    @staticmethod
    def error(itou):
        return f"Require the 'compat_quantized_model_int8_to_uint8' field must be bool type, now is {type(itou)}, default=False."

    @staticmethod
    def message():
        return (f"When compat_quantized_model_int8_to_uint8 = true, we will convert tflite quantization's "
                f"activation_dtype=int8, zp=-128 to activation_dtype=uint8, zp=0.")


@field_register('compat_quantized_model_ops', 'hidden')
class CompatQuantizedModelOpsField(BaseField):
    @staticmethod
    def default():
        return ''

    @staticmethod
    def ops(cqmo):
        return cqmo.lower().strip().replace(' ', '').split(',')

    @staticmethod
    def check(cqmo):
        lower_optype = [k.lower() for k in OpType.__dict__.keys()] + ['']
        ops = cqmo.strip().split(',')
        for o in ops:
            if cqmo.lower() not in lower_optype:
                return True
        return False

    @staticmethod
    def error(cqmo):
        return f"Require the 'compat_quantized_model_ops' field must be optype, now is {cqmo}, default=''."

    @staticmethod
    def message():
        return (f"When compat_quantized_model_ops = compass_optype, like compat_quantized_model_ops = squareddifferecne,"
                f" we will use other quantization method to substitude the compass quantization.")


@field_register('compat_quantized_model_unify_shifts_mode', 'hidden')
class CompatQuantizedModelUnifyShiftsModeField(BaseField):

    @staticmethod
    def default():
        return 'auto'

    @staticmethod
    def check(usm):
        return False if usm in ['max', 'mean', 'auto', 'none', 'entropy'] else True

    @staticmethod
    def error(usm):
        return (f"Require the 'compat_quantized_model_unify_shifts_mode' field must be str type and in "
                f"['max', 'mean', 'auto', 'none', 'entropy'], now is {usm}, default='auto'.")

    @staticmethod
    def message():
        return f"compat_quantized_model_unify_shifts_mode can set the unify shift strategy."


@field_register('compat_quantized_model_left_shift_bits', 'hidden')
class CompatQuantizedModelLeftShiftField(BaseField):
    @staticmethod
    def _left_shift_bits():
        return list(range(2, 16))

    @staticmethod
    def default():
        return '8'

    @staticmethod
    def check(mb):
        _abits = CompatQuantizedModelLeftShiftField._left_shift_bits()
        return True if not (isinstance(mb, int) and mb in _abits) else False

    @staticmethod
    def error(mb):
        _abits = CompatQuantizedModelLeftShiftField._left_shift_bits()
        return f"Required the integer 'compat_quantized_model_left_shift_bits' field and must be in {_abits}, now is {type(mb)} type, {mb}."

    @staticmethod
    def message():
        _abits = CompatQuantizedModelLeftShiftField._left_shift_bits()
        return f"The bits used to set the left shift bits in eltwise op."


ALL_FIELDS = {**DEFAULT_FIELDS, **HIDDEN_FIELDS}
