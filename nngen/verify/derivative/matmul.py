from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from .. import matmul as matmul_forward

import numpy as np
import nngen as ng

def matmul(grad_output, a, b, deriv_by_a=True,
        saved_tensors=None, scale=None,
        transposed_a=False, transposed_b=False,
        a_dtype=None, b_dtype=None, scale_dtype=None,
        mul_dtype=None, sum_dtype=None, dtype=None,
        asymmetric_clip=False, act_func=None,
        **kwargs):

    c_shape = grad_output.shape
    
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
        input_of_act_func = saved_tensors[0]
        act_deriv_method = act_func.get_deriv_method()
        grad_output = act_deriv_method(grad_output, input_of_act_func)

    if dtype is None:
        dtype = ng.int32
    scale_point = 0 if scale_dtype is None else scale_dtype.point
    scl_point = max(dtype.point, scale_point)
    scl_shift = min(dtype.point, scl_point)
    shifted_scale = np.right_shift(scale, scl_shift)

    grad_output = grad_output * shifted_scale[None]

    if deriv_by_a:
        grad_output = matmul_forward(grad_output, b,
                    transposed_a=False, transposed_b=not transposed_b,
                    a_dtype=dtype, b_dtype=b_dtype, dtype=ng.int64)
        if transposed_a:
            grad_output = grad_output.transpose()
        if dtype is None: dtype = a_dtype
    else:
        grad_output = matmul_forward(a, grad_output,
                    transposed_a=not transposed_a, transposed_b=False,
                    a_dtype=a_dtype, b_dtype=dtype, dtype=ng.int64)
        if transposed_b:
            grad_output = grad_output.transpose()
    ret_dtype = a_dtype if deriv_by_a else b_dtype

    # dynamic quantization
    max_absolute_value = np.abs(grad_output).max()
    if max_absolute_value:
        highest_bit = np.floor(np.log2(max_absolute_value)).astype(np.int8)
    else:
        highest_bit = 0
    available_highest_bit = ret_dtype.width - 2
    rshift = max(0, highest_bit - available_highest_bit)
    if rshift > 0:
        grad_output += 1 << (rshift - 1)
        grad_output >>= rshift

    p_th = (1 << (ret_dtype.width - 1)) - 1
    if asymmetric_clip:
        n_th = -1 * p_th - 1
    else:
        n_th = -1 * p_th

    p_th = p_th >> ret_dtype.point
    n_th = n_th >> ret_dtype.point

    grad_output = np.where(grad_output > p_th, p_th, 
        np.where(grad_output < n_th, n_th, grad_output))

    return grad_output
    