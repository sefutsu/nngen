from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

def relu(ctx, grad):
    x, = ctx.saved_tensors
    return np.where(x > 0, grad, np.zeros_like(grad, dtype=np.int64)), 1
