from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from .. import matmul as matmul_forward

import numpy as np
import nngen as ng

def matmul(propagated_gradient, a, b, deriv_by_a=True,
        saved_tensors=None, scale=None,
        transposed_a=False, transposed_b=False,
        rshift_mul=None, rshift_sum=None, rshift_aaa=None,
        a_dtype=None, b_dtype=None, scale_dtype=None, pg_dtype=None,
        asymmetric_clip=False,
        act_func=None, dtype=None,
        **kwargs):

    c_shape = propagated_gradient.shape
    
    if scale is None:
        scale = np.ones([c_shape[-1]], dtype=np.int64)
    elif not isinstance(scale, np.ndarray):
        new_scale = np.zeros([c_shape[-1]], dtype=np.int64)
        for i in range(new_scale.shape[-1]):
            new_scale[i] = scale
        scale = new_scale
    elif len(scale.shape) == 1 and scale.shape[0] == 1:
        new_scale = np.zeros([c_shape[-1]], dtype=np.int64)
        for i in range(new_scale.shape[-1]):
            new_scale[i] = scale[0]
        scale = new_scale


    if act_func is not None:
        try:
            activated_value = saved_tensors["activated_value"]
        except KeyError:
            raise ValueError("No input value of activation function")
        act_deriv_method = act_func.get_deriv_method()
        propagated_gradient = act_deriv_method(propagated_gradient, activated_value)

    if pg_dtype is None:
        pg_dtype = ng.int32

    scale_point = 0 if scale_dtype is None else scale_dtype.point
    scl_point = max(pg_dtype.point, scale_point)
    scl_shift = min(pg_dtype.point, scl_point)
    shifted_scale = np.right_shift(scale, scl_shift)

    propagated_gradient = propagated_gradient * shifted_scale[None]

    if deriv_by_a:
        propagated_gradient = matmul_forward(propagated_gradient, b,
                    transposed_a=False, transposed_b=not transposed_b,
                    a_dtype=pg_dtype, b_dtype=b_dtype, dtype=ng.int64)
        if transposed_a:
            propagated_gradient = propagated_gradient.transpose()
        if dtype is None: dtype = a_dtype
    else:
        propagated_gradient = matmul_forward(a, propagated_gradient,
                    transposed_a=not transposed_a, transposed_b=False,
                    a_dtype=a_dtype, b_dtype=pg_dtype, dtype=ng.int64)
        if transposed_b:
            propagated_gradient = propagated_gradient.transpose()
        if dtype is None: dtype = b_dtype
    if dtype is None: dtype = ng.int32

    c_shape = propagated_gradient.shape
    if rshift_mul is None:
        rshift_mul = np.zeros([c_shape[-1]], dtype=np.int64)
    elif not isinstance(rshift_mul, np.ndarray):
        new_rshift_mul = np.zeros([c_shape[-1]], dtype=np.int64)
        for i in range(new_rshift_mul.shape[-1]):
            new_rshift_mul[i] = rshift_mul
        rshift_mul = new_rshift_mul
    elif len(rshift_mul.shape) == 1 and rshift_mul.shape[0] == 1:
        new_rshift_mul = np.zeros([c_shape[-1]], dtype=np.int64)
        for i in range(new_rshift_mul.shape[-1]):
            new_rshift_mul[i] = rshift_mul[0]
        rshift_mul = new_rshift_mul

    if rshift_sum is None:
        rshift_sum = np.zeros([c_shape[-1]], dtype=np.int64)
    elif not isinstance(rshift_sum, np.ndarray):
        new_rshift_sum = np.zeros([c_shape[-1]], dtype=np.int64)
        for i in range(new_rshift_sum.shape[-1]):
            new_rshift_sum[i] = rshift_sum
        rshift_sum = new_rshift_sum
    elif len(rshift_sum.shape) == 1 and rshift_sum.shape[0] == 1:
        new_rshift_sum = np.zeros([c_shape[-1]], dtype=np.int64)
        for i in range(new_rshift_sum.shape[-1]):
            new_rshift_sum[i] = rshift_sum[0]
        rshift_sum = new_rshift_sum

    if rshift_aaa is None:
        rshift_aaa = np.zeros([c_shape[-1]], dtype=np.int64)
    elif not isinstance(rshift_aaa, np.ndarray):
        new_rshift_out = np.zeros([c_shape[-1]], dtype=np.int64)
        for i in range(new_rshift_out.shape[-1]):
            new_rshift_out[i] = rshift_out
        rshift_out = new_rshift_out
    elif len(rshift_out.shape) == 1 and rshift_out.shape[0] == 1:
        new_rshift_out = np.zeros([c_shape[-1]], dtype=np.int64)
        for i in range(new_rshift_out.shape[-1]):
            new_rshift_out[i] = rshift_out[0]
        rshift_aaa = new_rshift_out

    # dynamic quantization
    max_absolute_value = np.abs(propagated_gradient).max()
    if max_absolute_value:
        highest_bit = np.floor(np.log2(max_absolute_value)).astype(np.int8)
    else:
        highest_bit = 0
    available_highest_bit = dtype.width - 2
    rshift = max(0, highest_bit - available_highest_bit)
    if rshift > 0:
        propagated_gradient += 1 << (rshift - 1)
        propagated_gradient >>= rshift

    p_th = (1 << (dtype.width - 1)) - 1
    if asymmetric_clip:
        n_th = -1 * p_th - 1
    else:
        n_th = -1 * p_th

    p_th = p_th >> dtype.point
    n_th = n_th >> dtype.point

    propagated_gradient = np.where(propagated_gradient > p_th, p_th, 
        np.where(propagated_gradient < n_th, n_th, propagated_gradient))

    return propagated_gradient
    