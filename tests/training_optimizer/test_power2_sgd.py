from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import nngen as ng
from nngen.training import optim, quantizer

class DummyOptimizer(optim._Optimizer):
    def scaled_grad(self, node):
        return node.grad

def test_update_value():
    a = ng.variable(dtype=ng.int8, shape=(1,))
    a.set_value([0])
    b = ng.add(a, a)
    optimizer = DummyOptimizer()
    for i in range(3):
        optimizer.zero_grad([b])
        a.grad = [-1]
        optimizer.step([a])
        assert a.value == [i + 1]

shape = (3, 4)
def test_niti_sgd():
    a = ng.variable(dtype=ng.int8, shape=shape)
    optimizer = optim.niti_sgd()
    for mu in range(1, 6):
        optimizer.mu = mu
        a.grad = np.random.randint(-1000, 1000, shape)
        scaled_grad = optimizer.scaled_grad(a)
        # room for stochastic rounding
        assert mu - 1 <= quantizer.bit_width(scaled_grad) <= mu + 1

def test_power2_sgd():
    optimizer = optim.power2_sgd(0.5)
    dummy = np.random.randint(-31, 32, shape)
    a = ng.variable(dtype=ng.int8, shape=dummy.shape)
    a.grad = dummy << 2
    a.grad_scale_factor = 3
    a.scale_factor = 6
    scaled_grad = optimizer.scaled_grad(a)
    assert (scaled_grad == dummy).all()
