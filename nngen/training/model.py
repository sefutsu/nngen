from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import nngen as ng
from nngen.training.util import set_with_default

class _default_dtype:
    act = ng.int8
    weight = ng.int8
    bias = ng.int32

class model:
    def __init__(self, layers, criterion, input_shape, 
                 input_dtype=None, act_dtype=None, weight_dtype=None, bias_dtype=None, grad_dtype=None, name=None):
        self.layers = layers
        self.input_shape = input_shape
        self.criterion = criterion

        self.act_dtype = set_with_default(act_dtype, _default_dtype.act)
        self.weight_dtype = set_with_default(weight_dtype, _default_dtype.weight)
        self.bias_dtype = set_with_default(bias_dtype, _default_dtype.bias)

        self.input_dtype = set_with_default(input_dtype, self.act_dtype)
        self.grad_dtype = set_with_default(grad_dtype, act_dtype)

        self.name = set_with_default(name, "main")

    def create_forward_layers(self):
        self.input_name = self.name + ".input"
        self.input = ng.placeholder(dtype=self.input_dtype, shape=self.input_shape, name=self.input_name)
        x = self.input
        for layer in self.layers:
            layer.set_type_if_none(act_dtype=self.act_dtype, weight_dtype=self.weight_dtype, bias_dtype=self.bias_dtype)
            x = layer.forward(x)
        self.forward = x

    def create_backward_layers(self):
        

