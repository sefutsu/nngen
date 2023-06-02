from __future__ import absolute_import
from __future__ import print_function

from nngen.training.quantizer import *

cnt = 1000

def test_stochastic_rounding_float():
    array = np.random.uniform(-1000, 1000, cnt)
    res = stochastic_rounding_float(array)
    correct = (res == np.floor(array)).sum() + (res == np.ceil(array)).sum()
    assert correct == cnt

def test_stochastic_rounding_int_shift():
    array = np.random.randint(-(1 << 30), 1 << 30, cnt)
    for rshift in range(30):
        res = stochastic_rounding_int_shift(array, rshift)
        correct = (res == array >> rshift).sum() + (res == (array >> rshift) + 1).sum()
        assert correct == cnt
