from __future__ import absolute_import
from __future__ import print_function

import torch
from cross_entropy_loss import *
from nngen.training.util import Context
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
    
    reduction = ["mean", "sum"]

    for r in reduction:
        weight = generate_weight(batch_size, num_classes)
        target = generate_target(batch_size, num_classes)

        torch_weight = torch.tensor(weight, requires_grad=True)
        torch.nn.CrossEntropyLoss(reduction=r)(torch_weight, torch.tensor(target)).backward()
        torch_res = torch_weight.grad.numpy()

        ctx = Context()
        verify.cross_entropy_loss(ctx, weight, target, reduction=r)
        nngen_res = backward.cross_entropy_loss(ctx, 1, r)

        print(torch_res, nngen_res)
        assert (abs(nngen_res - torch_res) < 1e-5).all()