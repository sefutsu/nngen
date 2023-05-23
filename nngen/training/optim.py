from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import nngen.basic_types as bt
import nngen.storage as storage
import nngen.training.quantizer as quantizer

import numpy as np

class _Optimizer:
    def __init__(self, lr):
        self.lr = lr
    def step(self, node):
        if isinstance(node, bt._Operator):
            for arg in node.args:
                self.step(arg)
        elif isinstance(node, storage.variable) and node.requires_grad:
            node.value -= self.scaled_grad(node)
    def scaled_grad(self, node):
        raise NotImplementedError()

class power2_sgd(_Optimizer):
    def __init__(self, lr, grad_bitwidth_low=1, grad_bitwidth_high=5):
        _Optimizer.__init__(self, lr)
        self.grad_bitwidth_low = grad_bitwidth_low
        self.grad_bitwidth_high = grad_bitwidth_high
    def scaled_grad(self, node):
        grad_scale_factor = node.grad_scale_factor * self.lr
        weight_scale_factor = node.scale_factor
        scale_factor_diff = grad_scale_factor / weight_scale_factor
        scale_factor_diff_exp = np.round(np.log2(scale_factor_diff))
        grad_bitwidth = quantizer.bit_width(node.grad)
        scaled_grad_bitwidth = grad_bitwidth + scale_factor_diff_exp
        scaled_grad_bitwidth = np.clip(scaled_grad_bitwidth, 
                                       self.grad_bitwidth_low, self.grad_bitwidth_high)
        grad_rshift = np.maximum(0, grad_bitwidth - scaled_grad_bitwidth)
        shifted_grad = quantizer.stochastic_rounding_int_shift(node.grad, grad_rshift)
        return shifted_grad

class niti_sgd(_Optimizer):
    def __init__(self, lr, mu=5):
        _Optimizer.__init__(self, lr)
        self.mu = mu
    def scaled_grad(self, node):
        grad_bitwidth = quantizer.bit_width(node.grad)
        scaled_grad_bitwidth = self.mu
        grad_rshift = np.maximum(0, grad_bitwidth - scaled_grad_bitwidth)
        shifted_grad = quantizer.stochastic_rounding_int_shift(node.grad, grad_rshift)
        return shifted_grad
