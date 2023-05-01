from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

class Context:
    def __init__(self):
        self.saved_tensors = []
    def save_for_backward(self, *tensors):
        self.saved_tensors = [tensor.copy() for tensor in tensors]
