from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

def matmul(ctx, grad, act_func=None):
    if act_func:
        act_func_backward = act_func.get_backward_method()
        grad *= act_func_backward(ctx.act_func_ctx)
    a, b = ctx.saved_tensors
    grad_a = np.matmul(b, grad.T)
    grad_b = np.matmul(a, grad.T)
    grad_bias = np.ones((a.shape[0],), dtype=np.int64)
    return grad_a, grad_b, grad_bias
