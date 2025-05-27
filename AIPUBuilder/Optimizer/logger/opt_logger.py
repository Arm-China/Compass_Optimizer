# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

import sys
import time
import traceback

from AIPUBuilder.Optimizer.logger.aipu_logger import INFO, DEBUG, WARN, ERROR, FATAL, LOGLEVEL, _DEBUG_LEVEL
import AIPUBuilder

__all__ = ['OPT_DEBUG', 'OPT_INFO', 'OPT_WARN', 'OPT_ERROR', 'OPT_FATAL', 'tqdm']


from tqdm import tqdm as _tqdm


class tqdm(_tqdm):
    def __init__(self, *args, **kwargs):
        from torch.utils.data import DataLoader
        file = sys.stdout
        if "file" in kwargs:
            file = kwargs["file"]

        disable = not file.isatty()
        kwargs["disable"] = disable or kwargs.get("disable", False)

        consumer = kwargs.get('consumer', False)
        self.dataset = None
        if consumer and len(args) >= 1 and isinstance(args[0], DataLoader) and hasattr(args[0], 'dataset'):
            args[0].dataset.__setattr__('consumer', kwargs['consumer'])
            self.dataset = args[0].dataset

        if 'consumer' in kwargs:
            kwargs.pop('consumer')

        super(tqdm, self).__init__(*args, **kwargs)
        self.args = args
        self.kwargs = kwargs

    def __exit__(self, *exc):
        if self.dataset is not None and hasattr(self.dataset, 'consumer'):
            self.dataset.__dict__.pop('consumer')
        super(tqdm, self).__exit__(*exc)


def get_time():
    """Return the asctime."""
    local_time = time.localtime()
    return "%02d:%02d:%02d" % (local_time.tm_hour, local_time.tm_min, local_time.tm_sec)


__time__ = get_time
BT_NAME = '[OPT]'
g_msg_mem_dt = {}
if hasattr(AIPUBuilder, '__release__') and AIPUBuilder.__dict__.get('__release__'):

    def base_logger(msg, *args, workflow_name='', op_name='', **kwargs):
        ltime = __time__()
        prefix_header = BT_NAME
        if ('prefix_header' in kwargs and kwargs['prefix_header'] is not None and
                isinstance(kwargs['prefix_header'], str) and len(kwargs['prefix_header']) > 1):
            prefix_header = kwargs['prefix_header']
        log_head = prefix_header + ' [' + ltime + ']'
        if prefix_header != BT_NAME:
            kwargs.pop('prefix_header')
        args_str = ' '.join(args) + ' '.join(kwargs.values())
        workflow_name = workflow_name + ' ' if len(workflow_name) else workflow_name
        op_name = op_name + ' ' if len(op_name) else op_name
        log_body = workflow_name + op_name + msg + ('' if len(args_str) == 0 else args_str)
        return '%s: %s' % (log_head, log_body)

    def OPT_INFO(msg, *args, workflow_name='', log_once=False, **kwargs):
        if log_once:
            global g_msg_mem_dt
            if msg in g_msg_mem_dt.keys():
                return
            else:
                g_msg_mem_dt[msg] = True
        log = base_logger(msg, *args, workflow_name=workflow_name, **kwargs)
        INFO(log)

    def OPT_DEBUG(msg, *args, workflow_name='', op_name='', log_once=False, **kwargs):

        if LOGLEVEL > _DEBUG_LEVEL:
            return
        if log_once:
            global g_msg_mem_dt
            if msg in g_msg_mem_dt.keys():
                return
            else:
                g_msg_mem_dt[msg] = True
        log = base_logger(msg, *args, workflow_name=workflow_name, op_name=op_name, **kwargs)
        DEBUG(log)

    def OPT_ERROR(msg, *args, workflow_name='', op_name='', log_once=False, **kwargs):
        if log_once:
            global g_msg_mem_dt
            if msg in g_msg_mem_dt.keys():
                return
            else:
                g_msg_mem_dt[msg] = True
        log = base_logger(msg, *args, workflow_name=workflow_name, op_name=op_name, **kwargs)
        ERROR(log)

    def OPT_WARN(msg, *args, workflow_name='', op_name='', log_once=False, **kwargs):
        if log_once:
            global g_msg_mem_dt
            if msg in g_msg_mem_dt.keys():
                return
            else:
                g_msg_mem_dt[msg] = True
        log = base_logger(msg, *args, workflow_name=workflow_name, op_name=op_name, **kwargs)
        WARN(log)

    def OPT_FATAL(msg, *args, workflow_name='', op_name='', **kwargs):
        log = base_logger(msg, *args, workflow_name=workflow_name, op_name=op_name, **kwargs)
        FATAL(log)
else:
    bt_depth = -3

    # def get_cur_info():
    #     """Return the frame object for the caller's stack frame."""
    #     f = None
    #     try:
    #         raise Exception
    #     except:
    #         f = sys.exc_info()[2].tb_frame.f_back
    #     return (f.f_code.co_name, f.f_lineno)

    def get_line_number():
        try:
            raise Exception
        except:
            try:
                fs = traceback.extract_stack()[bt_depth]
                return fs.lineno
            except:
                return ''

    def get_function_name():
        try:
            raise Exception
        except:
            try:
                fs = traceback.extract_stack()[bt_depth]
                return fs.name
            except:
                return ''

    def get_file_name():
        try:
            raise Exception
        except:
            try:
                fs = traceback.extract_stack()[bt_depth]
                return fs.filename.split('/')[-1]
            except:
                return ''

    # def get_time():
    #     """Return the asctime."""
    #     local_time = time.localtime()
    #     return "%02d:%02d:%02d" % (local_time.tm_hour, local_time.tm_min, local_time.tm_sec)

    __line__ = get_line_number
    __func__ = get_function_name
    __filen__ = get_file_name

    def get_prefix_header(**kwargs):
        prefix_header = BT_NAME
        if ('prefix_header' in kwargs and kwargs['prefix_header'] is not None and
                isinstance(kwargs['prefix_header'], str) and len(kwargs['prefix_header']) > 1):
            prefix_header = kwargs['prefix_header']
        return prefix_header

    def OPT_INFO(msg, *args, workflow_name='', log_once=False, **kwargs):
        ltime = __time__()
        prefix_header = get_prefix_header(**kwargs)
        if prefix_header != BT_NAME:
            kwargs.pop('prefix_header')
        log_head = prefix_header + ' [' + ltime + ']'
        args_str = ' '.join(args) + ' '.join(kwargs.values())
        workflow_name = workflow_name + ' ' if len(workflow_name) else workflow_name
        log_body = workflow_name + msg + ('' if len(args_str) == 0 else args_str)
        if log_once:
            global g_msg_mem_dt
            if msg in g_msg_mem_dt.keys():
                return
            else:
                g_msg_mem_dt[msg] = True
        INFO('%s: %s' % (log_head, log_body))

    def OPT_DEBUG(msg, *args, workflow_name='', op_name='', log_once=False, **kwargs):

        if LOGLEVEL > _DEBUG_LEVEL:
            return

        # debug without time
        # ltime = __time__()

        ltime = ''
        fname = __filen__()
        lineno = __line__()
        prefix_header = get_prefix_header(**kwargs)
        if prefix_header != BT_NAME:
            kwargs.pop('prefix_header')
        log_head = prefix_header + ' [' + ltime + ' ' + fname + ':' + str(lineno) + ']'

        args_str = ' '.join(args) + ' '.join(kwargs.values())
        workflow_name = workflow_name + ' ' if len(workflow_name) else workflow_name
        op_name = op_name + ' ' if len(op_name) else op_name
        log_body = workflow_name + op_name + msg + ('' if len(args_str) == 0 else args_str)
        if log_once:
            global g_msg_mem_dt
            if msg in g_msg_mem_dt.keys():
                return
            else:
                g_msg_mem_dt[msg] = True
        DEBUG('%s: %s' % (log_head, log_body))

    def OPT_ERROR(msg, *args, workflow_name='', op_name='', log_once=False, **kwargs):
        ltime = __time__()
        fname = __filen__()
        lineno = __line__()
        prefix_header = get_prefix_header(**kwargs)
        if prefix_header != BT_NAME:
            kwargs.pop('prefix_header')
        log_head = prefix_header + ' [' + ltime + ' ' + fname + ':' + str(lineno) + ']'

        args_str = ' '.join(args) + ' '.join(kwargs.values())
        workflow_name = workflow_name + ' ' if len(workflow_name) else workflow_name
        op_name = op_name + ' ' if len(op_name) else op_name
        log_body = workflow_name + op_name + msg + ('' if len(args_str) == 0 else args_str)
        if log_once:
            global g_msg_mem_dt
            if msg in g_msg_mem_dt.keys():
                return
            else:
                g_msg_mem_dt[msg] = True
        ERROR('%s: %s' % (log_head, log_body))

    def OPT_WARN(msg, *args, workflow_name='', op_name='', log_once=False, **kwargs):
        ltime = __time__()
        fname = __filen__()
        lineno = __line__()
        prefix_header = get_prefix_header(**kwargs)
        if prefix_header != BT_NAME:
            kwargs.pop('prefix_header')
        log_head = prefix_header + ' [' + ltime + ' ' + fname + ':' + str(lineno) + ']'

        args_str = ' '.join(args) + ' '.join(kwargs.values())
        workflow_name = workflow_name + ' ' if len(workflow_name) else workflow_name
        op_name = op_name + ' ' if len(op_name) else op_name
        log_body = workflow_name + op_name + msg + ('' if len(args_str) == 0 else args_str)
        if log_once:
            global g_msg_mem_dt
            if msg in g_msg_mem_dt.keys():
                return
            else:
                g_msg_mem_dt[msg] = True
        WARN('%s: %s' % (log_head, log_body))

    def OPT_FATAL(msg, *args, workflow_name='', op_name='', **kwargs):
        ltime = __time__()
        fname = __filen__()
        lineno = __line__()
        prefix_header = get_prefix_header(**kwargs)
        if prefix_header != BT_NAME:
            kwargs.pop('prefix_header')
        log_head = prefix_header + ' [' + ltime + ' ' + fname + ':' + str(lineno) + ']'

        args_str = ' '.join(args) + ' '.join(kwargs.values())
        workflow_name = workflow_name + ' ' if len(workflow_name) else workflow_name
        op_name = op_name + ' ' if len(op_name) else op_name
        log_body = workflow_name + op_name + msg + ('' if len(args_str) == 0 else args_str)

        FATAL('%s: %s' % (log_head, log_body))
