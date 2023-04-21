from collections import OrderedDict

from . import basic_types as bt
from . import basic

def matmul(a, b, bias=None, scale=None,
           transposed_a=False, transposed_b=False,
           rshift_out=None,
           act_func=None,
           dtype=None, sum_dtype=None,
           name=None):
    if transposed_a:
        a = basic.transpose(a)
    if transposed_b:
        b = basic.transpose(b)
    res = _matmul(a, b, bias, scale, rshift_out, dtype=dtype, name=name)
    if act_func:
        return act_func(res, name=None if name is None else name + ".act")
    else:
        return res

class _matmul(bt._Operator):
    def __init__(self, a, b, bias=None, scale=None,
                rshift_out=None,
                dtype=None,
                name=None):
        
        shape = (a.shape[0], b.shape[1])
        args = [a, b]

        if bias is not None:
            args.append(bias)
            self.has_bias = True
        else:
            self.has_bias = False

        if scale is not None:
            args.append(scale)
            self.has_scale = True
        else:
            self.has_scale = False

        #TODO: when `vshamt_out` is constant
        vshamt_out = rshift_out
        if vshamt_out is not None:
            args.append(vshamt_out)
            self.has_vshamt_out = True
        else:
            self.has_vshamt_out = False

        bt._Operator.__init__(self, *args, dtype=dtype, shape=shape, name=name)


class relu(bt._ActFuncOperator):
    def act_func(self, features):
        pass
    def deriv_act_func(self, features):
        pass
