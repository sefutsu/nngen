from collections import OrderedDict

from . import basic_types as bt
from . import basic

class matmul(bt._Operator):
    def __init__(self, a, b, bias=None, scale=None,
                transposed_a=False, transposed_b=False,
                rshift_out=None,
                act_func=None,
                dtype=None,
                name=None):
        if transposed_a:
            a = basic.transpose(a)
        if transposed_b:
            b = basic.transpose(b)

        self.transposed_a = a
        self.transposed_b = b

        shape = (transposed_a.shape[0], transposed_b.shape[1])
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

        self.act_func = act_func(self) if act_func is not None else None
