from __future__ import absolute_import
from __future__ import print_function

import torch
from cross_entropy_loss import *
import nngen.verify as verify

def test_cross_entropy_loss():
    batch_size = 4
    num_classes = 10
    
    reduction = ["mean", "sum", "none"]

    for r in reduction:
        weight = generate_weight(batch_size, num_classes)
        target = generate_target(batch_size, num_classes)
        torch_res = torch.nn.CrossEntropyLoss(reduction=r)(torch.tensor(weight), torch.tensor(target)).numpy()
        nngen_res = verify.cross_entropy_loss(weight, target, reduction=r)
        assert (abs(nngen_res - torch_res) < 1e-5).all()
