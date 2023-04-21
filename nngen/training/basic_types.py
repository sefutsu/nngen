import math

from nngen import dtype_list
from nngen.basic_types import same_shape, max_shape

class _Numeric:
    def __init__(self, dtype=None, shape=None, name=None):
        if shape is None:
            pass
        elif isinstance(shape, tuple):
            pass
        elif isinstance(shape, list):
            shape = tuple(shape)
        elif isinstance(shape, int):
            shape = tuple([shape])
        else:
            raise ValueError("illegal shape type: '%s'" % str(type(shape)))

        # zero-sized shape check
        if shape is not None:
            for s in shape:
                if s == 0:
                    raise ValueError("shape contains '0': %s" % str(shape))

        self.name = name

        self.shape = shape
        self.dtype = dtype

class _Storage(_Numeric):
    def __init__(self, dtype=None, shape=None, name=None, is_input=False):
        _Numeric.__init__(self, dtype=dtype, shape=shape, name=name)
        self.is_input = is_input

class _Operator(_Numeric):
    def __init__(self, *args, **opts):
        dtype = opts['dtype'] if 'dtype' in opts else None
        shape = opts['shape'] if 'shape' in opts else None
        name = opts['name'] if 'name' in opts else None
        par = opts['par'] if 'par' in opts else 1

        if dtype is None:
            dtype = dtype_list.get_max_dtype(*args)

        if shape is None:
            shape = same_shape(*args)

        if par is not None and 2 ** int(math.ceil(math.log(par, 2))) != par:
            raise ValueError('par must be power of 2.')

        _Numeric.__init__(self, dtype=dtype, shape=shape, name=name)
        self.args = args

class _StreamingOperator(_Operator):
    def __init__(self, *args, **opts):
        dtype = opts['dtype'] if 'dtype' in opts else None
        shape = opts['shape'] if 'shape' in opts else None
        name = opts['name'] if 'name' in opts else None

        if shape is None:
            shape = max_shape(*args)

        _Operator.__init__(self, *args, dtype=dtype, shape=shape, name=name)

class _ElementwiseOperator(_StreamingOperator):
    pass

class _ActFuncOperator(_ElementwiseOperator):
    def __init__(self, features, dtype=None, name=None):
        if dtype is None:
            dtype = features.dtype
        shape = features.shape
        _ElementwiseOperator.__init__(self, features, dtype=dtype, shape=shape, name=name)
