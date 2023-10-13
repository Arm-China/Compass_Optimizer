# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023 Arm Technology (China) Co. Ltd.

from AIPUBuilder.Optimizer.framework import graph_inference
from AIPUBuilder.Optimizer.passes import InsertCastOp, insert_op_pass
from AIPUBuilder.Optimizer.logger import OPT_INFO
import copy


class NaiveAutoSearchMixedPrecision(object):

    def __init__(self, g, val_dataloader, fmetrics, qmetrics, hparams):

        self.g = g
        self.validation_dataloader = val_dataloader
        self.fmetrics = fmetrics
        self.qmetrics = qmetrics
        self.hparams = hparams

        self.search_times = 0
        self.abatches = 0
        self.athres = 0
        self.aless = 0
        self.fscore = 0.0

    #################################################################
    # to speedup the search progress, we assume that the deeper layer are more sensitive
    # to quantization, so we just search a partion layer that subsequent layers are all
    # quantized by higher bits (16) or not quantized.

    def qinference_simulation(self, pid, bits):
        vdataloader = copy.deepcopy(self.validation_dataloader)
        qmetrics = copy.deepcopy(self.qmetrics)
        for qm in qmetrics:
            qm.reset()
        self.g.quantgraph = self.g.clone()
        for n in self.g.quantgraph.nodes:
            n.attrs['debug_fake_quantize'] = False
            for t in n.outputs:
                t.debug_flag = 0
            if int(n.attrs['layer_id']) >= pid:
                if bits > 0:
                    n.attrs['q_bits_activation'] = 16 if bits > 8 else 8
                    n.attrs['q_bits_weight'] = 16 if bits > 8 else 8
                    n.attrs['q_bits_bias'] = 48 if bits > 8 else 32
                    n.attrs['lut_items_in_bits'] = 10 if bits > 8 else 8
                else:
                    n.attrs['debug_fake_quantize'] = True
        if bits > 0:
            insert_obj = [InsertCastOp]
            insert_op_pass(self.g.quantgraph, self.hparams, insert_obj)
        self.g.quantize()
        graph_inference(self.g.quantgraph,
                        self.g.qforward,
                        vdataloader,
                        qmetrics,
                        with_float=False,
                        max_batches=self.abatches,
                        disable_tqdm=True)
        qscore = qmetrics[0].compute()
        for qm in qmetrics:
            qm.reset()
        self.search_times += 1
        OPT_INFO('mixed_precision_auto_search (on %d batches of validation dataset): searched %d times, with score [%s], original score [%s].' % (
            self.abatches, self.search_times, str(qscore), str(self.fscore)))
        return qscore

    def satisfy_acc_drop(self, acc_drop):
        return (self.aless and (acc_drop <= self.athres)) or ((not self.aless) and (acc_drop >= self.athres))

    def search_pid(self, pbits, init_acc_drop):
        acc_drop = init_acc_drop
        pid = max(0, len(self.g.nodes))
        pid_pre = pid + 1
        while not self.satisfy_acc_drop(acc_drop):
            pid_pre = pid
            pid = pid // 2
            if pid >= pid_pre:
                break
            flag = False
            for n in self.g.nodes:
                if int(n.attrs['layer_id']) >= pid and (int(n.attrs['q_bits_activation']) < pbits or int(n.attrs['q_bits_weight']) < pbits or pbits < 1):
                    flag = True
            if flag:
                qscore = self.qinference_simulation(pid, pbits)
                acc_drop = self.fscore - qscore
        t0 = pid
        t = 0
        s_drop = acc_drop
        while True:
            qid = t0 + 2**t
            t += 1
            if qid >= pid_pre:
                break
            qscore = self.qinference_simulation(qid, pbits)
            acc_drop = self.fscore - qscore
            if not self.satisfy_acc_drop(acc_drop):
                break
            else:
                pid = qid
                s_drop = acc_drop
        sid = pid
        # for i in range(pid+1, min(pid_pre, qid)):
        #     qscore = self.qinference_simulation(pid, pbits)
        #     acc_drop = fscore - qscore
        #     if satisfy_acc_drop(acc_drop) :
        #         sid = i
        #         s_drop = acc_drop
        #     else :
        #         break
        return sid, s_drop
    #################################################################

    def auto_search(self):
        self.abatches = self.hparams.mixed_precision_auto_search_batches
        self.athres = self.hparams.mixed_precision_auto_search_thres
        self.aless = self.hparams.mixed_precision_auto_search_less
        # get the original metric score first
        vdataloader = copy.deepcopy(self.validation_dataloader)
        fmetrics = copy.deepcopy(self.fmetrics)
        for fm in fmetrics:
            fm.reset()
        graph_inference(self.g,
                        self.g.forward,
                        vdataloader,
                        fmetrics,
                        with_float=True,
                        max_batches=self.abatches,
                        disable_tqdm=True)
        self.fscore = fmetrics[0].compute()
        for fm in fmetrics:
            fm.reset()

        gnodes = len(self.g.nodes)
        pbits = 16
        qscore = self.qinference_simulation(gnodes, pbits)
        pid, acc_drop1 = self.search_pid(pbits, self.fscore - qscore)
        smsg = 'mixed_precision_auto_search (on %d batches of validation dataset): layers(layer_id aligned to input float IR) will be optimized as follow:\n' % (
            self.abatches,)
        if self.satisfy_acc_drop(acc_drop1):
            if pid > 0:
                smsg += 'quantized as is: layer %d to %d\n' % (0, pid - 1)
            if pid < gnodes:
                for n in self.g.nodes:
                    if int(n.attrs['layer_id']) >= pid:
                        n.attrs['q_bits_activation'] = 16
                        n.attrs['q_bits_weight'] = 16
                        n.attrs['q_bits_bias'] = 48
                        n.attrs['lut_items_in_bits'] = 10
                smsg += 'quantized to 16bits: layer %d to %d\n' % (pid, gnodes - 1)
        else:
            fid, acc_drop2 = self.search_pid(0, self.fscore - qscore)
            if fid > 0:
                smsg += 'quantized as is: layer %d to %d\n' % (0, fid - 1)
            if fid < gnodes:
                for n in self.g.nodes:
                    if int(n.attrs['layer_id']) >= fid:
                        n.params['unquantifiable'] = True
                        n.attrs['trigger_float_op'] = 'float32_preferred'
                    else:
                        n.params['unquantifiable'] = False
                # dummy_op_list = self.g.insert_dummy_node_ahead(OpType.DeQuantize, condition_func=lambda node, parent_node, edge_tensor: (not parent_node.get_param(
                #     'unquantifiable', optional=True, default_value=False)) and node.get_param('unquantifiable', optional=True, default_value=False))
                # dummy_op_list += self.g.insert_dummy_node_ahead(OpType.Quantize, condition_func=lambda node, parent_node, edge_tensor: parent_node.get_param(
                #     'unquantifiable', optional=True, default_value=False) and (not node.get_param('unquantifiable', optional=True, default_value=False)))
                # for n in dummy_op_list:
                #     n.params['unquantifiable'] = True
                smsg += (f"unquantifiable layers is [{fid}, {gnodes-1}], and these layers would run on CPU device "
                         f"using float32 type. if you want to run these unquantifiable layers in AIPU device, "
                         f"please make sure AIPU hardware configuration can support float type. "
                         f"and then configure 'trigger_float_op' in Optimizer cfg or "
                         f"modify in per-layer json format template file. \n!Attention: because AIPU lib operators "
                         f"may not completely support float type on some hardware configureations, "
                         f"the model maybe has different accuracy comparing to running on CPU device "
                         f"when running on AIPU device with float type.")

        OPT_INFO(smsg)
        self.g.quantgraph = None
