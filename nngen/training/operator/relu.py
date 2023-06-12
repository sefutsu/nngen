from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import nngen.operator as op

class relu:
    def forward(self, x):
        return op.relu(x)
    def backward(self, x):
        return op.deriv_relu(x)
