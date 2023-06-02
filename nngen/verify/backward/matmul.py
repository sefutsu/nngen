from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

def matmul(ctx, grad, scale_a, scale_b, act_func=None):
    act_func_scale = 1
    if act_func:
        act_func_backward = act_func.get_backward_method()
        grad, act_func_scale = act_func_backward(ctx.act_func_ctx, grad)
    # transpose_a = False, transposed_b = True
    a, b = ctx.saved_tensors
    grad_a = np.matmul(grad, b), scale_b * act_func_scale
    grad_b = np.matmul(grad.T, a), scale_a * act_func_scale
    grad_bias = grad.sum(axis=0), act_func_scale
    return grad_a, grad_b, grad_bias
