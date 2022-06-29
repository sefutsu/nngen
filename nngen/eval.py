from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

def eval(objs, **input_dict):
    memo = {}
    return [obj.eval(memo, input_dict) for obj in objs]

def gradient(loss, input_var, shape=(1,)):
    return loss.gradient(input_var, np.ones(shape=shape, dtype=np.int64))
