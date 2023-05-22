from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

class Context:
    def __init__(self):
        self._saved_tensors = []
    def save_for_backward(self, *tensors):
        self._saved_tensors = [tensor.copy() for tensor in tensors]
    @property
    def saved_tensors(self):
        if not self._saved_tensors:
            raise ValueError("no saved tensors")
        return self._saved_tensors
