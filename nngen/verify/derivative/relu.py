from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np


def relu(features, dtype=None, name=None, par=1,
         features_dtype=None):

    out_point = 0 if dtype is None else dtype.point

    comp = features >= 0

    zeros = np.zeros_like(features, dtype=np.int64)
    ones = np.ones_like(features, dtype=np.int64) << out_point

    ret = np.where(comp, ones, zeros)
    return ret

