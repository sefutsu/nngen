from double_layer import *

def test_double_layer():
    inputs = generate_inputs()
    weights = generate_weights()

    nngen_res = nngen_gradient(inputs, weights)
    torch_res = torch_gradient(inputs, weights)

    assert (abs(nngen_res - torch_res) < 1e-5).all()
