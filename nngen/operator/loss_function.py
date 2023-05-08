from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import nngen.basic_types as bt
import numpy as np
from nngen.training import quantizer

class cross_entropy_loss(bt._Operator):
    def __init__(self, weight, target, dtype=None, reduction='mean', name=None):
        # `dtype` must be float
        if dtype is None:
            dtype = np.float32
        bt._Operator.__init__(self, weight, target, dtype=dtype, shape=weight.shape, name=name)
        self.reduction = reduction
        self.weight_dtype = weight.dtype

    def eval(self, memo, input_dict, **kwargs):
        if id(self) in memo:
            return memo[id(self)]
        
        weight = self.args[0].eval(memo, input_dict)
        weight_scale_factor = self.args[0].scale_factor
        float_weight = weight.astype(self.dtype) * weight_scale_factor
        target = self.args[1].eval(memo, input_dict).astype(self.dtype)

        method = self.get_eval_method()
        ret = method(self.ctx, float_weight, target, self.reduction)
        memo[id(self)] = ret

        return ret

    def backward(self, grad, scale_factor):
        self.grad = grad
        self.grad_scale_factor = scale_factor

        method = self.get_backward_method()
        delta = method(self.ctx, grad, self.reduction)

        delta, scale_factor = quantizer.quantize_from_float(delta, self.weight_dtype, scale_factor)
        self.args[0].backward(delta, scale_factor)
