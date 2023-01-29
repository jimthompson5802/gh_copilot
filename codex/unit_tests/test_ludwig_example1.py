 test_forward():
    # Test forward method
    encoder = Stacked2DCNN(
        height=84,
        width=84,
        num_conv_layers=2,
        num_channels=3,
        out_channels=32,
        kernel_size=3,
        stride=1,
        padding="valid",
        dilation=1,
        conv_use_bias=True,
        padding_mode="zeros",
        conv_norm=None,
        conv_norm_params=None,
        conv_activation="relu",
        conv_dropout=0,
        pool_function="max",
        pool_kernel_size=2,
        pool_stride=None,
        pool_padding=0,
        pool_dilation=1,
        groups=1,
        fc_layers=None,
        num_fc_layers=1,
        output_size=128,
        fc_use_bias=True,
        fc_weights_initializer="xavier_uniform",
        fc_bias_initializer="zeros",
        fc_norm=None,
        fc_norm_params=None,
        fc_activation="relu",
        fc_dropout=0,
        encoder_config=None,
    )

    # Create a random input
    input_shape = encoder.input_shape
    inputs = torch.randint(0, 255, input_shape, dtype=torch.uint8)

    # Run forward method
    outputs = encoder.forward(inputs)

    # Check output shape
    assert outputs["encoder_output"].shape == encoder.output_shape

# Unit test output shape
def test_output_shape():
    # Test output shape
    encoder = Stacked2DCNN(
        height=84,
        width=84,
        num_conv_layers=2,
        num_channels=3,
        out_channels=32,
        kernel_size=3,
        stride=1,
        padding="valid",
        dilation=1,
        conv_use