 test_forward():
    fc_stack = FCStack(
        first_layer_input_size=2,
        layers=[
            {"input_size": 2, "output_size": 4},
            {"output_size": 4, "use_bias": False},
        ],
    )
    input_tensor = torch.randn(2, 2)
    output_tensor = fc_stack(input_tensor)
    assert output_tensor.shape == torch.Size([2, 4])

# Unit test forward method
def test_forward_residual():
    fc_stack = FCStack(
        first_layer_input_size=2,
        layers=[
            {"input_size": 2, "output_size": 4},
            {"output_size": 4, "use_bias": False},
        ],
        residual=True,
    )
    input_tensor = torch.randn(2, 2)
    output_tensor = fc_stack(input_tensor)
    assert output_tensor.shape == torch.Size([2, 4])

# Unit test forward method
def test_forward_no_layers():
    fc_stack = FCStack(
        first_layer_input_size=2,
        layers=None,
        num_layers=0,
    )
    input_tensor = torch.randn(2, 2)
    output_tensor = fc_stack(input_tensor)
    assert output_tensor.shape == torch.Size([2, 2])

# Unit test forward method
def test_forward_no_layers_residual():
    fc_stack = FCStack(
        first_layer_input_size=2,
        layers=None,
        num_layers=0,
        residual=True,
    )
    input_tensor = torch.randn(2, 2)
    output_tensor = fc_stack(input_tensor)
    assert output_tensor.shape == torch.Size([2, 2])

# Unit test forward method
def test_forward_no_layers_residual_norm():