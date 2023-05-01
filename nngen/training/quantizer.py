from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

def stochastic_rounding(array):
    floor = np.floor(array)
    frac = array - floor
    rand = np.random.rand(array.shape)
    ceil = rand < frac
    return (floor + ceil).astype(np.int64)

def quantize_from_float(array, dtype, scale_factor):
    if dtype.signed:
        array_abs_max = np.max(np.abs(array))
        quantized_abs_max = (1 << (dtype.width - 1)) - 1
    else:
        array = np.clip(array, 0, None)
        array_abs_max = np.max(array)
        quantized_abs_max = (1 << dtype.width) - 1
    array = stochastic_rounding(array / array_abs_max * quantized_abs_max)
    scale_factor = scale_factor * array_abs_max / quantized_abs_max
    return array, scale_factor
