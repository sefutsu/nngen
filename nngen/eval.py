from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

def eval(objs, **input_dict):
    memo = {}
    return [obj.eval(memo, input_dict) for obj in objs]

def backward(objs):
    for obj in objs:
        obj.backward(np.array(1.0), 1.0)
