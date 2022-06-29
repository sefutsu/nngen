from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from .. import matmul as matmul_forward

import numpy as np

def matmul(propagated_gradient, a, b, deriv_by_a=True,
        stored_input=None, scale=None,
        transposed_a=False, transposed_b=False,
        rshift_mul=None, rshift_sum=None, rshift_out=None,
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

    if rshift_out is None:
        rshift_out = np.zeros([c_shape[-1]], dtype=np.int64)
    elif not isinstance(rshift_out, np.ndarray):
        new_rshift_out = np.zeros([c_shape[-1]], dtype=np.int64)
        for i in range(new_rshift_out.shape[-1]):
            new_rshift_out[i] = rshift_out
        rshift_out = new_rshift_out
    elif len(rshift_out.shape) == 1 and rshift_out.shape[0] == 1:
        new_rshift_out = np.zeros([c_shape[-1]], dtype=np.int64)
        for i in range(new_rshift_out.shape[-1]):
            new_rshift_out[i] = rshift_out[0]
        rshift_out = new_rshift_out


    if act_func is not None:
        try:
            activated_value = stored_input["activated_value"]
        except KeyError:
            raise ValueError("No input value of activation function")
        act_deriv_method = act_func.get_derivative_method()
        propagated_gradient = act_deriv_method(propagated_gradient, activated_value)


    rshift_mul_pow = np.where(rshift_mul > np.zeros_like(rshift_mul, dtype=np.int64),
                              rshift_mul - 1,
                              np.zeros_like(rshift_mul))
    rshift_mul_round = np.where(rshift_mul > np.zeros_like(rshift_mul, dtype=np.int64),
                                np.power(np.ones_like(rshift_mul, dtype=np.int64) * 2,
                                         rshift_mul_pow),
                                np.zeros_like(rshift_mul, dtype=np.int64))

    rshift_sum_pow = np.where(rshift_sum > np.zeros_like(rshift_sum, dtype=np.int64),
                              rshift_sum - 1,
                              np.zeros_like(rshift_sum))
    rshift_sum_round = np.where(rshift_sum > np.zeros_like(rshift_sum, dtype=np.int64),
                                np.power(np.ones_like(rshift_sum, dtype=np.int64) * 2,
                                         rshift_sum_pow),
                                np.zeros_like(rshift_sum, dtype=np.int64))

    rshift_out_pow = np.where(rshift_out > np.zeros_like(rshift_out, dtype=np.int64),
                              rshift_out - 1,
                              np.zeros_like(rshift_out))
    rshift_out_round = np.where(rshift_out > np.zeros_like(rshift_out, dtype=np.int64),
                                np.power(np.ones_like(rshift_out, dtype=np.int64) * 2,
                                         rshift_out_pow),
                                np.zeros_like(rshift_out, dtype=np.int64))

    frac = np.where(rshift_out!=0, np.where(propagated_gradient>=0, rshift_out_round, rshift_out_round - 1),
            np.zeros_like(rshift_out, dtype=np.int64))

    propagated_gradient += frac
    propagated_gradient >>= rshift_out

    pg_point = 0 if pg_dtype is None else pg_dtype.point
    scale_point = 0 if scale_dtype is None else scale_dtype.point
    scl_point = max(pg_point, scale_point)
    scl_shift = min(pg_point, scl_point)
    shifted_scale = np.right_shift(scale, scl_shift)

    propagated_gradient = propagated_gradient * shifted_scale[None]

    propagated_gradient += rshift_sum_round
    propagated_gradient >>= rshift_sum

    propagated_gradient >>= rshift_mul_round
    propagated_gradient >>= rshift_mul

    if deriv_by_a:
        propagated_gradient = matmul_forward(propagated_gradient, b,
                    transposed_a=False, transposed_b=not transposed_b)
        if transposed_a:
            propagated_gradient = propagated_gradient.transpose()
        if dtype is None: dtype = a_dtype
    else:
        propagated_gradient = matmul_forward(a, propagated_gradient,
                    transposed_a=not transposed_a, transposed_b=False)
        if transposed_b:
            propagated_gradient = propagated_gradient.transpose()
        if dtype is None: dtype = b_dtype

    res_point = 0 if dtype is None else dtype.point
    res_width = 32 if dtype is None else dtype.width

    p_th = (1 << (res_width - 1)) - 1
    if asymmetric_clip:
        n_th = -1 * p_th - 1
    else:
        n_th = -1 * p_th

    p_th = p_th >> res_point
    n_th = n_th >> res_point

    propagated_gradient = np.where(propagated_gradient > p_th, p_th, 
        np.where(propagated_gradient < n_th, n_th, propagated_gradient))

    return propagated_gradient
    