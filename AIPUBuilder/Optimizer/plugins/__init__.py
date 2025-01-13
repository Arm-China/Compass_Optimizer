# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.

# for built in dataset
from . import aipubt_dataset_numpy
from . import aipubt_dataset_random
from . import aipubt_dataset_vocnchw
from . import aipubt_dataset_vocnhwc
from . import aipubt_dataset_aishell
from . import aipubt_dataset_mtcnn
from . import aipubt_dataset_librispeech
from . import aipubt_dataset_numpymultiinput
from . import aipubt_dataset_generaldict
from . import aipubt_dataset_iwslt
from . import aipubt_dataset_fasterrcnnvoc
from . import aipubt_dataset_coco
from . import aipubt_dataset_nhwcrgb2nhwcbgr
from . import aipubt_dataset_numpynhwc2nchw
from . import aipubt_dataset_numpynchw2nhwc
from . import aipubt_dataset_numpynhwcrgb2ncbgrhw
from . import aipubt_dataset_numpywithdim
from . import aipubt_dataset_widerface
from . import aipubt_dataset_cgtdnn
from . import aipubt_dataset_mpii
from . import aipubt_dataset_sphereface_lfw
from . import aipubt_dataset_numpymultiinputwithoutbatchdim
from . import aipubt_dataset_NumpyZipped
from . import aipubt_dataset_tusimple
from . import aipubt_dataset_numpymultiinputNCHW
from . import aipubt_dataset_tensorfromnumpymultiinput
from . import aipubt_dataset_stable_diffusion_unet
from . import aipubt_dataset_bevformer
from . import aipubt_dataset_bevformer_static
from . import aipubt_dataset_llama2
from . import aipubt_dataset_cocokp

# for built in metric
from . import aipubt_metric_CosDistance
from . import aipubt_metric_MaskRcnnCOCOmAP
from . import aipubt_metric_mIoU
from . import aipubt_metric_topk
from . import aipubt_metric_mAP
from . import aipubt_metric_SSDmAP
from . import aipubt_metric_fasterrcnnmAP
from . import aipubt_metric_YOLOmAP
from . import aipubt_metric_WER
from . import aipubt_metric_KeywordSpotting
from . import aipubt_metric_IWSLT_BLEU
from . import aipubt_metric_psnr
from . import aipubt_metric_EachCosDistance
from . import aipubt_metric_IWSLT_BLEU_2_gram
from . import aipubt_metric_Ocr
from . import aipubt_metric_CosDistance_with_seqlen
from . import aipubt_metric_fcos_mAP
from . import aipubt_metric_facebox
from . import aipubt_metric_centerface
from . import aipubt_metric_lightface
from . import aipubt_metric_delta1
from . import aipubt_metric_pckh
from . import aipubt_metric_retinafacebox
from . import aipubt_metric_sphereface
from . import aipubt_metric_MaxAbsError
from . import aipubt_metric_FlattenCosDistance
from . import aipubt_metric_poly_lanenet
from . import aipubt_metric_mobiledetSSDmAP
from . import aipubt_metric_centernet
from . import aipubt_metric_retinanetmAP
from . import aipubt_metric_f1mesure
from . import aipubt_metric_MaxAbsError_with_seqlen
from . import aipubt_metric_roc
from . import aipubt_metric_imdb
from . import aipubt_metric_LMHead
from . import aipubt_metric_cocokeypoint
from . import aipubt_metric_RMSE

try:
    from AIPUBuilder.Optimizer.plugins.aipubt_metric_bevformer import BEVFormerMetric
except Exception as e:
    pass
    # from AIPUBuilder.Optimizer.logger import OPT_DEBUG
    # OPT_DEBUG(f"import bevformermetric failed")

# for built in op
from . import aipubt_op_tile

# # for test debug
from . import aipubt_dataset_OpTestNumpyZipped
from . import aipubt_metric_OpTestCosDistance

# for qconfig
from . import aipubt_qconfig_bevformer
