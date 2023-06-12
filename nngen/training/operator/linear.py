from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import nngen as ng

class linear:
    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x):
        self.weight = ng.variable(shape=(self.out_dim, self.in_dim), dtype=self.weight_dtype)
        self.bias = ng.variable(shape=(1, self.out_dim), dtype=self.bias_dtype)
        
