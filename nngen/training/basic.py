from . import basic_types as bt

class transpose(bt._Operator):
    def __init__(self, a, perm=None, dtype=None, name=None):
        if len(a.shape) == 1:
            a_shape = tuple([1, a.shape[0]])
        else:
            a_shape = a.shape

        if perm is None:
            perm = tuple(reversed(range(len(a_shape))))

        self.transpose_perm = perm

        shape = (a_shape[p] for p in perm)

        bt._Operator.__init__(self, a, dtype=dtype, shape=shape, name=name)
