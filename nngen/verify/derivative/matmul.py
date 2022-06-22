from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from multiprocessing.sharedctypes import Value

import numpy as np
from .. import matmul as matmul_forward


def matmul(propagated_gradient, a, b, deriv_by_a=True, stored_input=None,
        transposed_a=False, transposed_b=False,
        a_dtype=None, b_dtype=None,
        act_func=None, dtype=None):

    if act_func is not None:
        try:
            activated_value = stored_input["activated_value"]
        except KeyError:
            raise ValueError("No input value of activation function")
        act_deriv_method = act_func.get_derivative_method()
        propagated_gradient = act_deriv_method(propagated_gradient, activated_value)


    if deriv_by_a:
        propagated_gradient = matmul_forward(propagated_gradient, b,
                    transposed_a=False, transposed_b=not transposed_b)
        if transposed_a:
            propagated_gradient = propagated_gradient.transpose()
    else:
        propagated_gradient = matmul_forward(a, propagated_gradient,
                    transposed_a=not transposed_a, transposed_b=False)
        if transposed_b:
            propagated_gradient = propagated_gradient.transpose()

    return propagated_gradient
    