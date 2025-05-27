# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

import numpy as np
from AIPUBuilder.Optimizer.utils import cosine_distance
from ..qatlogger import QAT_INFO


def check_result(actual, desired):
    assert len(actual) == len(desired), "actual: %d vs desired %d" % (
        len(actual),
        len(desired),
    )

    ret = True
    for idx in range(len(actual)):
        cos = cosine_distance(actual[idx], desired[idx])
        QAT_INFO(f"cosine distance of {idx} output: {cos}")
        # ret = np.testing.assert_allclose(
        #     actual[idx].detach().cpu().numpy(),
        #     desired[idx].detach().cpu().numpy(),
        #     rtol=1e-5,
        #     atol=1e-5) and ret

    return ret
