from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

def relu(grad_output, features, dtype=None, features_dtype=None):
    # grad_output: dtype
    # return: features_dtype
    grad_point = 0 if dtype is None else dtype.point
    features_point = 0 if features_dtype is None else features_dtype.point
    rshift_out = grad_point - features_point
    grad_output = grad_output >> rshift_out if rshift_out >= 0 \
                        else grad_output << (-rshift_out)

    zeros = np.zeros_like(features, dtype=np.int64)
    grad_output = np.where(features > 0, grad_output, zeros)
    
    return grad_output
