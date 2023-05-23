from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import torch
from generate_input import *
from nngen.training.util import Context
import nngen as ng
import nngen.verify as verify
import nngen.verify.backward as backward

batch_size = 4
num_classes = 10
reduction = ["mean", "sum", "none"]
eps = 1e-4

def test_verify_forward():    
    for r in reduction:
        weight = generate_weight(batch_size, num_classes)
        target = generate_target(batch_size, num_classes)

        torch_res = torch.nn.CrossEntropyLoss(reduction=r)(torch.tensor(weight), torch.tensor(target)).numpy()

        ctx = Context()
        nngen_res = verify.cross_entropy_loss(weight, target, reduction=r, ctx=ctx)

        relative_eps = eps * np.abs(torch_res).max()
        assert (np.abs(nngen_res - torch_res) < relative_eps).all()

def test_verify_backward():
    for r in reduction:
        weight = generate_weight(batch_size, num_classes)
        target = generate_target(batch_size, num_classes)

        torch_weight = torch.tensor(weight, requires_grad=True)
        celoss = torch.nn.CrossEntropyLoss(reduction=r)(torch_weight, torch.tensor(target))
        if r == "none":
            batch_weight = generate_weight(batch_size, 1)
            celoss = torch.matmul(celoss, torch.tensor(batch_weight))
        else:
            batch_weight = 1
        celoss.backward()
        torch_res = torch_weight.grad.numpy()

        ctx = Context()
        verify.cross_entropy_loss(weight, target, reduction=r, ctx=ctx)
        nngen_res = backward.cross_entropy_loss(ctx, batch_weight, r)

        relative_eps = eps * np.abs(torch_res).max()
        assert (np.abs(nngen_res - torch_res) < relative_eps).all()

def test_forward():
    for r in reduction:
        weight = ng.placeholder(dtype=ng.int8, shape=(batch_size, num_classes), name="weight")
        target = ng.placeholder(dtype=np.float32, shape=(batch_size, num_classes), name="target")
        weight_value = generate_int8_weight(batch_size, num_classes)
        target_value = generate_target(batch_size, num_classes)
        scale_factor = np.random.uniform(1, 1000)
        weight_float_tensor = torch.tensor(weight_value / scale_factor)
        weight.scale_factor = scale_factor

        torch_res = torch.nn.CrossEntropyLoss(reduction=r)(weight_float_tensor, torch.tensor(target_value)).numpy()

        celoss = ng.cross_entropy_loss(weight, target, reduction=r)
        nngen_res = ng.eval([celoss], weight=weight_value, target=target_value)[0]

        relative_eps = eps * np.abs(torch_res).max()
        assert (np.abs(nngen_res - torch_res) < relative_eps).all()

def _test_backward(weight_dtype, eps):
    reduction = ["mean", "sum"]
    for r in reduction:
        weight = ng.variable(dtype=weight_dtype, shape=(batch_size, num_classes), name="weight")
        target = ng.variable(dtype=np.float32, shape=(batch_size, num_classes), name="target")
        weight_value = generate_int8_weight(batch_size, num_classes)
        target_value = generate_target(batch_size, num_classes)
        weight.set_value(weight_value)
        target.set_value(target_value)
        scale_factor = np.random.uniform(1, 1000)
        weight_float_tensor = torch.tensor(weight_value / scale_factor, requires_grad=True)
        weight.scale_factor = scale_factor

        torch.nn.CrossEntropyLoss(reduction=r)(weight_float_tensor, torch.tensor(target_value)).backward()
        torch_res = weight_float_tensor.grad.numpy()

        celoss = ng.cross_entropy_loss(weight, target, reduction=r)
        ng.eval([celoss])
        ng.backward([celoss])
        nngen_res = weight.grad.astype(np.float32) / weight.grad_scale_factor
        
        relative_eps = eps * np.abs(torch_res).max()
        assert (np.abs(nngen_res - torch_res) < relative_eps).all()

def test_backward_int8():
    _test_backward(ng.int8, 1e-2)
def test_backward_int16():
    _test_backward(ng.int16, 1e-4)
