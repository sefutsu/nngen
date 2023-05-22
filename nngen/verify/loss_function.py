from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

def cross_entropy_loss(weight, target, reduction='mean', ctx=None):
    weight -= weight.max(axis=1, keepdims=True)
    exp_sum = np.exp(weight).sum(axis=1, keepdims=True)
    exp_sum = np.clip(exp_sum, 1e-10, None)
    log_softmax = weight - np.log(exp_sum)
    if ctx:
        ctx.save_for_backward(np.exp(log_softmax) - target)

    loss = -(log_softmax * target).sum(axis=1)
    if reduction == 'mean':
        return loss.sum() / loss.size
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError("reduction must be 'mean', 'sum' or 'none'")

def mse_loss(weight, target, reduction='maen', ctx=None):
    diff = weight - target
    if ctx:
        ctx.save_for_backward(diff)

    loss = diff * diff
    if reduction == 'mean':
        return loss.sum() / loss.size
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError("reduction must be 'mean', 'sum' or 'none'")
