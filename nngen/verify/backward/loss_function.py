from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

def cross_entropy_loss(ctx, grad, reduction='mean'):
    diff, = ctx.saved_tensors
    delta = diff * grad

    if reduction == 'mean':
        return delta / len(delta)
    elif reduction == 'sum' or reduction == "none":
        return delta
    else:
        raise ValueError("reduction must be 'mean', 'sum' or 'none'")

def mse_loss(ctx, grad, reduction='mean'):
    diff, = ctx.saved_tensors
    delta = diff * grad * 2

    if reduction == 'mean':
        return delta / delta.size
    elif reduction == 'sum' or reduction == "none":
        return delta
    else:
        raise ValueError("reduction must be 'mean', 'sum' or 'none'")
