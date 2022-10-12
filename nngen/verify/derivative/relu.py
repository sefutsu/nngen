from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

def relu(grad_output, features, dtype=None, pg_dtype=None):

    rshift_out = 0 if dtype is None or pg_dtype is None else pg_dtype.point - dtype.point
    grad_output = grad_output >> rshift_out if rshift_out >= 0 \
                        else grad_output << (-rshift_out)

    zeros = np.zeros_like(features, dtype=np.int64)
    grad_output = np.where(features > 0, grad_output, zeros)
    
    return grad_output
