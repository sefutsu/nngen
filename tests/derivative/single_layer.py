from __future__ import absolute_import
from __future__ import print_function

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

import nngen as ng
import numpy as np
import torch

### nngen.verify

dtype = ng.int64
input_layer = ng.placeholder(dtype=dtype, shape=(1, 3), name="input_layer")
w0 = ng.variable(dtype=dtype, shape=(2, 3), name="w0")
b0 = ng.variable(dtype=dtype, shape=(2,), name="b0")
scale = ng.variable(dtype=dtype, shape=(2,), name="scale")
s0 = ng.matmul(input_layer, w0, bias=b0, scale=scale, transposed_b=True, act_func=ng.relu, dtype=dtype)

w0_value = np.random.randint(-100, 100, size=(2, 3), dtype=np.int64)
w0.set_value(w0_value)

b0_value = np.random.randint(-100, 100, size=(1, 2), dtype=np.int64)
b0.set_value(b0_value)

scale_value = np.random.randint(-100, 100, size=(2,), dtype=np.int64)
scale.set_value(scale_value)

input_value = np.random.randint(-100, 100, size=(1, 3), dtype=np.int64)

eval_res = ng.eval([s0], input_layer=input_value)
res = ng.gradient(s0, input_layer)

### torch.autograd

dtype = torch.double

input_value = torch.tensor(input_value, dtype=dtype, requires_grad=True)
w0 = torch.tensor(w0_value, dtype=dtype)
b0 = torch.tensor(b0_value, dtype=dtype)
scale = torch.tensor(scale_value, dtype=dtype)

a = input_value @ w0.T + b0
k = a * scale
h = torch.relu(k)
# k.retain_grad()
s = h.sum()

s.backward()
torch_res = input_value.grad.numpy()

if (abs(res - torch_res) < 1e-5).all():
    print("# verify: PASSED")
else:
    print("# verify: FAILED")
    print("Result:", res)
    print("PyTorch Result:", torch_res)

