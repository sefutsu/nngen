from __future__ import absolute_import
from __future__ import print_function

import nngen as ng
import numpy as np

def generate_target(batch_size, num_classes):
    labels = np.random.randint(0, num_classes, size=(batch_size,))
    one_hot_vectors = np.eye(num_classes)[labels]
    return one_hot_vectors

def generate_weight(batch_size, num_classes):
    return np.random.uniform(-1e5, 1e5, (batch_size, num_classes))

def generate_int8_weight(batch_size, num_classes):
    return np.random.randint(-127, 128, (batch_size, num_classes))
