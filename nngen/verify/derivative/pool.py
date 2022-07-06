from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
from nngen.util import divide

# (N, H, W, C)
def avg_pool(propagated_gradient, ksize, strides, padding=None, stored_input=None,
            dtype=None, pg_dtype=None, **kwargs):

    # padding: (up, right, down, left)
    if padding is None or (isinstance(padding, str) and padding.lower() == "same"):
        padding = stored_input["padding"]
    if isinstance(padding, str) and padding.lower() == "valid":
        padding = 0
    if isinstance(padding, int):
        padding = (padding,) * 4
    elif isinstance(padding, (tuple, list)):
        if len(padding) == 1:
            padding *= 4
        elif len(padding) == 2: # (up&down, left&right)
            padding *= 2
        elif len(padding) == 3: # (up, left&right, down)
            padding = (*padding, padding[1])
    else:
        raise ValueError("padding options must be 'SAME', 'VALID', int, tuple, or list.")

    if isinstance(ksize, int):
        ksize = (ksize,) * 2
    if isinstance(strides, int):
        strides = (strides,) * 2

    input_shape = list(stored_input["input_shape"])
    input_shape[1] += padding[0] + padding[2]
    input_shape[2] += padding[1] + padding[3]

    output_shape = propagated_gradient.shape

    res = np.zeros(input_shape, dtype=np.int64)

    for n in range(input_shape[0]):
        for k in range(input_shape[3]):
            for h in range(output_shape[1]):
                h_in = h * strides[0]
                for w in range(output_shape[2]):
                    w_in = w * strides[1]
                    for i in range(ksize[0]):
                        for j in range(ksize[1]):
                            res[n, h_in+i, w_in+j, k] += propagated_gradient[n, h, w, k]

    # remove padding
    propagated_gradient = propagated_gradient[:, 
        padding[0] : output_shape[1] - padding[2],
        padding[3] : output_shape[2] - padding[1],
        :]

    res = divide(res, ksize[0] * ksize[1])

    pg_point = 0 if pg_dtype is None else pg_dtype.point
    res_point = pg_point if dtype is None else dtype.point
    lshift = res_point - pg_point

    res = res << lshift if lshift > 0 else res >> (-lshift)

    return res
    