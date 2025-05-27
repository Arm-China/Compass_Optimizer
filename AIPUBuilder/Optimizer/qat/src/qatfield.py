# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

import os
import re
import functools

from AIPUBuilder.Optimizer.config.cfg_fields import BaseField, field_register, get_all_fields
from AIPUBuilder.Optimizer.framework.opt_register import TRAIN_PLUGIN_DICT
from .qatlogger import QAT_ERROR


qat_field_register = functools.partial(field_register, fscope='qat')


@qat_field_register('input_model', 'default')
class InputModelField(BaseField):

    @staticmethod
    def default():
        return ''

    @staticmethod
    def parse(im):
        return im == '' or os.path.isfile(im), im

    @staticmethod
    def error(im):
        return f"Require existed input model path, now model path is {im}."

    @staticmethod
    def message():
        return "input model path for AIPU QAT."


@qat_field_register('train_data', 'default')
class TrainDataField(BaseField):
    # the npy label file for the qat dataset
    @staticmethod
    def default():
        return ''

    @staticmethod
    def parse(d):
        return os.path.isfile(d) or os.path.exists(d) or d == '', d

    @staticmethod
    def error(d):
        return f"Require the existed 'train_data' path, now {d} does not exist."

    @staticmethod
    def message():
        return f"A qat dataset path for training."


@qat_field_register('train_label', 'default')
class TrainLabelField(BaseField):
    # the npy label file for the qat dataset
    @staticmethod
    def default():
        return ''

    @staticmethod
    def parse(l):
        return os.path.isfile(l) or os.path.exists(l) or l == '', l

    @staticmethod
    def error(l):
        return f"Require the existed 'train_label' path, now {l} does not exist."

    @staticmethod
    def message():
        return f"A label path for training."


@qat_field_register('train_batch_size', 'default')
class TrainBatchSizeField(BaseField):
    # the batch size used for train dataset
    @staticmethod
    def default():
        return '1'

    @staticmethod
    def parse(mbz):
        return isinstance(mbz, int) and mbz > 0, mbz

    @staticmethod
    def error(mbz):
        msg = mbz if isinstance(mbz, int) else type(mbz)
        return f"Required the positive integer(>0) 'train_batch_size' field, now is {msg}. default value=1."

    @staticmethod
    def message():
        return f"Batch size when finetune a model's accuracy."


@qat_field_register('validate_data', 'default')
class ValidateDataField(BaseField):
    # the npy label file for the qat dataset
    @staticmethod
    def default():
        return ''

    @staticmethod
    def parse(d):
        return os.path.isfile(d) or os.path.exists(d) or d == '', d

    @staticmethod
    def error(d):
        return f"Require the existed 'validate_data' path, now {d} does not exist."

    @staticmethod
    def message():
        return f"A qat validation dataset path for validate the trained model acc."


@qat_field_register('validate_label', 'default')
class ValidateLabelField(BaseField):
    # the npy label file for the qat dataset
    @staticmethod
    def default():
        return ''

    @staticmethod
    def parse(l):
        return os.path.isfile(l) or os.path.exists(l) or l == '', l

    @staticmethod
    def error(l):
        return f"Require the existed 'validate_label' path, now {l} does not exist."

    @staticmethod
    def message():
        return f"A qat validation label path for validate the trained model acc."


@qat_field_register('train_shuffle', 'default')
class TrainShuffleField(BaseField):
    # whether shuffe juring computing quantization parameters over calibration dataset
    @staticmethod
    def default():
        return 'False'

    @staticmethod
    def parse(cs):
        return isinstance(cs, bool), cs

    @staticmethod
    def error(cs):
        return f"Require the train_shuffle field must be in bool type, now is {type(cs)} type. default value=False."

    @staticmethod
    def message():
        return f"Whether to apply shuffle over training dataset."


@qat_field_register('train', 'default')
class TrainField(BaseField):
    @staticmethod
    def _train_plugins():
        trains = [k for k in TRAIN_PLUGIN_DICT.keys()]
        return trains

    @staticmethod
    def _split_trains(m):
        check = r'^(\w+\(?(\/?\w+\.?\w*,?)*\)?,?)+$'
        cpattern = re.compile(check)
        if len(re.findall(r'[^\w|\\|/|\.|,|\)|\()]', m)) > 0 or cpattern.match(m) is None:
            QAT_ERROR((f"train format:'trainname(args,...)/trainname()/trainname'; "
                       f"'trainname'and 'args' should be [a-zA-Z0-9_]"))
            return False
        rule = r'\w+\(?(\/?\w+\.?\w*,?)*\)?,?'
        pattern = re.compile(rule)
        rmatch = pattern.match(m)
        train_plugins = []
        newm = m
        while rmatch is not None:
            train_plugins.append(rmatch.group())
            newm = newm[rmatch.span()[1]:]
            rmatch = pattern.match(newm)
        return train_plugins

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
    def parse(m):
        if not isinstance(m, str):
            return False, m
        m = m.replace(' ', '')
        if m != '':
            if m.count('(') != m.count(')'):
                return False, m
            trains = TrainField._split_trains(m)
            if isinstance(trains, bool) and trains == False:
                return False, m
            func_args = TrainField._get_func_args(trains)
            mnames = [fa[0].lower() for fa in func_args]
        else:
            mnames = m
        if set(mnames).issubset(set(TrainField._train_plugins() + [''])):
            return True, m
        return False, m

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
            msg += f"The num of '(' is not same to the num of ')', please check the 'train' field. "
        elif _checker(m):
            msg += f"Please check the 'train' field format. "
        else:
            msg += f"Require the 'train' field must be in {TrainField._train_plugins()}, "
        msg += f"now 'train={m}'."
        return msg

    @staticmethod
    def get_train(m):
        m = m.replace(' ', '')
        trains = TrainField._split_trains(m)
        func_args = TrainField._get_func_args(trains)

        # delete the repeat train
        fn_arg_dict = {}
        fas = [[s[0].lower(), s[1]] for s in func_args]
        repeat = False
        for fa in fas:
            fname = fa[0]
            args = fa[1]
            if fname in fn_arg_dict.keys():
                for argl in fn_arg_dict[fname]:
                    if args == argl:
                        repeat = True
                        break
                if not repeat:
                    fn_arg_dict[fname].append(args)
            else:
                fn_arg_dict.update({fname: [args]})
        return fn_arg_dict

    @staticmethod
    def message():
        return (f"A train plugin name for training model. It supports multi-train and uses ',' to seperate. "
                f"Now Optimizer supports train plugins: {TrainField._train_plugins()}.")


@qat_field_register('parallel', 'default')
class ParallelField(BaseField):
    # whether shuffe juring computing quantization parameters over calibration dataset
    @staticmethod
    def default():
        return 'False'

    @staticmethod
    def parse(cs):
        return isinstance(cs, bool), cs

    @staticmethod
    def error(cs):
        return f"Require the parallel field must be in bool type, now is {type(cs)} type. default value=False."

    @staticmethod
    def message():
        return f"Whether to apply DataParallel for training loop."


@qat_field_register('input_shape', 'default')
class InputShapeField(BaseField):
    @staticmethod
    def default():
        return ''

    @staticmethod
    def _parse(cs):
        def _trans_str_to_list(v):
            import re

            def is_valid_list_string(sv):
                if (len(sv) > 1) and (sv[0] == '[') and (sv[-1] == ']'):
                    stk = []
                    for i in range(len(sv)):
                        c = sv[i]
                        if '[' == c:
                            stk.append(c)
                        elif ']' == c:
                            if len(stk) < 1:
                                return False
                            else:
                                stk.pop()
                    if len(stk) < 1:
                        return True
                    else:
                        return False
                else:
                    return False

            sv = v.strip()
            if is_valid_list_string(sv):
                # list
                lt = []
                ts = sv[1:-1] + ','
                cntl = 0
                cntr = 0
                pos = 0
                for i in range(len(ts)):
                    c = ts[i]
                    if '[' == c:
                        cntl += 1
                    elif ']' == c:
                        cntr += 1
                    elif ',' == c:
                        if cntl == cntr:
                            sub_str = ts[pos:i].strip(' \'')
                            if len(sub_str) > 0:
                                lt.append(_trans_str_to_list(sub_str))
                            pos = i+1
                return lt
            elif re.match(r'^(\-|\+)?\d+$', sv.lower()):
                return int(sv)
            else:
                return str(sv)
        ret = _trans_str_to_list(cs)
        return ret

    @staticmethod
    def parse(cs):
        if cs == '':
            return True, ''
        ret = InputShapeField._parse(cs)
        return (False, cs) if isinstance(ret, str) else (True, ret)

    @staticmethod
    def error(cs):
        return f"Require the input_shape field must be in str type of tensor shape, now is {type(cs)} type. default value=''."

    @staticmethod
    def message():
        return f"input_shape is used to generate this shape model or IR."


@qat_field_register('input_dtype', 'default')
class InputDtypeField(InputShapeField):

    @staticmethod
    def error(idf):
        return f"Require the input_dtype field must be in str type of tensor dtype, now is {type(idf)} type. default value=''."

    @staticmethod
    def message():
        return f"input_dtype is used to set the dtype of input node in original model."


@qat_field_register('set_input_name', 'default')
class SetInputNameField(BaseField):
    @staticmethod
    def default():
        return ''

    @staticmethod
    def parse(iname):
        if len(iname) == 0:
            return True, None
        ret = iname.strip().replace(' ', '').split(',')
        return True, ret

    @staticmethod
    def error(cs):
        return f"'set_input_name' field is names to assign to the input nodes of the graph, in order. and default value = None."

    @staticmethod
    def message():
        return f"assign the input node's name."


@qat_field_register('set_output_name', 'default')
class SetOutputNameField(BaseField):
    @staticmethod
    def default():
        return ''

    @staticmethod
    def parse(oname):
        if len(oname) == 0:
            return True, None
        ret = oname.strip().replace(' ', '').split(',')
        return True, ret

    @staticmethod
    def error(cs):
        return f"'set_output_name' field is names to assign to the output nodes of the graph, in order. and default value = None."

    @staticmethod
    def message():
        return f"assign the output node's name."


@qat_field_register('export_format', 'default')
class ExportFormatField(BaseField):
    _support_format = ['compass', 'torch', 'onnx']

    @staticmethod
    def default():
        return 'compass'

    @staticmethod
    def parse(oname):
        if oname.lower() not in ExportFormatField._support_format:
            return False, oname
        return True, oname.lower()

    @staticmethod
    def error(cs):
        return f"'output_model_format' field should be in [{ExportFormatField._support_format}], but now is {cs}."

    @staticmethod
    def message():
        return f"set the export format"


get_all_fields()
