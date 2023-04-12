from . import basic_types as bt

class placeholder(bt._Storage):
    def __init__(self, dtype, shape=None, name=None):
        bt._Storage.__init__(self, dtype, shape, name, is_input=True)

class variable(bt._Storage):
    def __init__(self, dtype, shape=None, name=None):
        bt._Storage.__init__(self, dtype, shape, name, is_input=False)
