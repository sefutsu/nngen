from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

def relu(propagated_gradient, features, dtype=None, pg_dtype=None):

    rshift_out = 0 if dtype is None or pg_dtype is None else pg_dtype.point - dtype.point
    propagated_gradient = propagated_gradient >> rshift_out if rshift_out >= 0 \
                        else propagated_gradient << (-rshift_out)

    zeros = np.zeros_like(features, dtype=np.int64)
    propagated_gradient = np.where(features > 0, propagated_gradient, zeros)
    
    return propagated_gradient
