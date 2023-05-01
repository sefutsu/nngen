from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import torch
from cross_entropy_loss import *
from nngen.training.util import Context
import nngen as ng
import nngen.verify as verify
import nngen.verify.backward as backward

def test_verify_forward():
    batch_size = 4
    num_classes = 10
    
    reduction = ["mean", "sum", "none"]

    for r in reduction:
        weight = generate_weight(batch_size, num_classes)
        target = generate_target(batch_size, num_classes)

        torch_res = torch.nn.CrossEntropyLoss(reduction=r)(torch.tensor(weight), torch.tensor(target)).numpy()

        ctx = Context()
        nngen_res = verify.cross_entropy_loss(ctx, weight, target, reduction=r)
        assert (abs(nngen_res - torch_res) < 1e-5).all()

def test_verify_backward():
    batch_size = 4
    num_classes = 10
    
    reduction = ["mean", "sum", "none"]

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
        verify.cross_entropy_loss(ctx, weight, target, reduction=r)
        nngen_res = backward.cross_entropy_loss(ctx, batch_weight, r)

        assert (abs(nngen_res - torch_res) < 1e-5).all()

def test_forward():
    batch_size = 4
    num_classes = 10
    
    reduction = ["mean"]

    for r in reduction:
        weight = ng.placeholder(dtype=ng.int8, shape=(batch_size, num_classes), name="weight")
        target = ng.placeholder(dtype=np.float32, shape=(batch_size, num_classes), name="target")
        weight_value = generate_int8_weight(batch_size, num_classes)
        target_value = generate_target(batch_size, num_classes)
        scale_factor = np.random.uniform(1e-3, 1)
        weight_float_value = weight_value * scale_factor
        weight.scale_factor = scale_factor

        torch_res = torch.nn.CrossEntropyLoss(reduction=r)(torch.tensor(weight_float_value), torch.tensor(target_value)).numpy()

        celoss = ng.cross_entropy_loss(weight, target, reduction=r)
        nngen_res = ng.eval([celoss], weight=weight_value, target=target_value)[0]

        assert (abs(nngen_res - torch_res) < 1e-5).all()
