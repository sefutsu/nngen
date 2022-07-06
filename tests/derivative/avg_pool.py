from __future__ import absolute_import
from __future__ import print_function

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

import nngen as ng
import numpy as np
import torch

# from nngen.util import divide
# x = np.random.randint(0, 100, size=(3, 3))
# print(x)
# print(divide(x, 9))

### nngen.verify

dtype = ng.int64
input_shape = (2, 8, 8, 3)

input_layer = ng.placeholder(dtype=dtype, shape=input_shape, name="input_layer")
out = ng.avg_pool(input_layer, ksize=(1, 3, 3, 1), strides=(1, 1, 1, 1), padding=0)

input_value = np.random.randint(0, 200, size=input_shape, dtype=np.int64)

eval_res = ng.eval([out], input_layer=input_value)
res = ng.gradient(out, input_layer, shape=eval_res[0].shape)

### torch.autograd

dtype = torch.double

input_value = torch.tensor(input_value, dtype=dtype, requires_grad=True)
v = input_value.permute(0, 3, 1, 2)
out = torch.nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))(v)

s = out.sum()
s.backward()

torch_res = input_value.grad.numpy()

if (abs(res - torch_res) < 1e-5).all():
    print("# verify: PASSED")
else:
    print("# verify: FAILED")
    print("Result:", res)
    print("PyTorch Result:", torch_res)

