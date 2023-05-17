from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

def stochastic_rounding_float(array):
    floor = np.floor(array)
    frac = array - floor
    rand = np.random.rand(*array.shape)
    ceil = rand < frac
    return (floor + ceil).astype(np.int64)

def stochastic_rounding_int_shift(array, rshift):
    res = np.abs(array)
    frac = np.bitwise_and(array, (1 << rshift) - 1)
    res >>= rshift
    rand = np.random.randint(0, 1 << rshift, array.shape)
    res += rand < frac
    return res * np.sign(array)

def quantize_from_int(array, scale_factor, to_dtype):
    max_bit_pos = _max_bit_position(array)
    scaled_max_bit_pos = to_dtype.width - 1 - to_dtype.signed
    rshift_amount = max(0, max_bit_pos - scaled_max_bit_pos)

    shifted_array = stochastic_rounding_int_shift(array, rshift_amount)
    if to_dtype.signed:
        vmax = (1 << (to_dtype.width - 1) - 1)
        vmin = -vmax
    else:
        vmax = (1 << to_dtype.width - 1) - 1
        vmin = 0
    shifted_array = np.clip(shifted_array, vmin, vmax)
    shifted_scale_factor = scale_factor * 2 ** rshift_amount

    return shifted_array, shifted_scale_factor

def quantize_from_float(array, scale_factor, to_dtype):
    if to_dtype.signed:
        array_abs_max = np.max(np.abs(array))
        quantized_abs_max = (1 << (to_dtype.width - 1)) - 1
    else:
        array = np.clip(array, 0, None)
        array_abs_max = np.max(array)
        quantized_abs_max = (1 << to_dtype.width) - 1
    array = stochastic_rounding_float(array / array_abs_max * quantized_abs_max)
    scale_factor = scale_factor * array_abs_max / quantized_abs_max
    return array, scale_factor

def _max_bit_position(array):
    # 0-indexed from LSB
    max_val = np.abs(array).max()
    return int(np.log2(max_val))
