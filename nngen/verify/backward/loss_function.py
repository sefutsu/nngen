from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

def cross_entropy_loss(ctx, grad, reduction='mean'):
    softmax, target = ctx.saved_tensors
    delta = (softmax - target) * grad

    if reduction == 'mean':
        return delta / len(delta)
    elif reduction == 'sum':
        return delta
    elif reduction == "none":
        raise NotImplementedError
    else:
        raise ValueError("reduction must be 'mean', 'sum' or 'none'")
