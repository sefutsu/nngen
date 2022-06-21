from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

def multiply(a, b,
           dtype=None, asymmetric_clip=False,
           a_dtype=None, b_dtype=None,
           **kwargs):

    a_point = 0 if a_dtype is None else a_dtype.point
    b_point = 0 if b_dtype is None else b_dtype.point
    c_point = max(a_point, b_point) if dtype is None else dtype.point
    c_width = 32 if dtype is None else dtype.width

    c = a.astype(np.int64) * b.astype(np.int64)
    c_rshift = a_point + b_point - c_point
    if c_rshift > 0:
        c >> c_rshift
    else:
        c << (-c_rshift)
    
    p_th = (1 << (c_width - 1)) - 1
    if asymmetric_clip:
        n_th = -1 * p_th - 1
    else:
        n_th = -1 * p_th

    p_th = p_th >> c_point
    n_th = n_th >> c_point
    c = np.where(c > p_th, p_th, np.where(c < n_th, n_th, c))

    return c
    