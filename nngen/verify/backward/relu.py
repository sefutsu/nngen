from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

def relu(ctx):
    x, = ctx.saved_tensors
    return np.where(x > 0, np.ones_like(x, dtype=np.int64), np.zeros_like(x, dtype=np.int64))
