Summarizes results using Copilot to generate unit tests for Ludwig code.

## First iteration of generating unit tests
These unit tests were generated with the Codex recommmended prompt:
```python
# Unit test
def
```

Generated test for [`ludwig_example0.py`](https://github.com/jimthompson5802/gh_copilot/blob/main/codex/sample_code/ludwig_example0.py).  

Noted issues in generated code: 
* The `forward()` method is not tested.  Only attributes created in the constructor are validated.
* Missing required imports
* Truncated code

```python
 test_fc_stack():
    fc_stack = FCStack(
        first_layer_input_size=2,
        layers=[
            {"input_size": 2, "output_size": 4},
            {"output_size": 4, "use_bias": False},
        ],
    )
    assert fc_stack.num_layers == 2
    assert fc_stack.input_shape == torch.Size([2])
    assert fc_stack.output_shape == torch.Size([4])

    fc_stack = FCStack(
        first_layer_input_size=2,
        num_layers=2,
        default_output_size=4,
        default_use_bias=False,
    )
    assert fc_stack.num_layers == 2
    assert fc_stack.input_shape == torch.Size([2])
    assert fc_stack.output_shape == torch.Size([4])

    fc_stack = FCStack(
        first_layer_input_size=2,
        num_layers=0,
    )
    assert fc_stack.num_layers == 0
    assert fc_stack.input_shape == torch.Size([2])
    assert fc_stack.output_shape == torch.Size([2])

    fc_stack = FCStack(
        first_layer_input_size=2,
        layers=None,
    )
    assert fc_stack.num_layers == 1
    assert fc_stack.input_shape == torch.Size([2])
    assert fc_stack.output_shape == torch.Size([256])

    fc_stack = FCStack(
        first_layer_input_size=2,
        layers=None,
        num_layers=0,
    )
    assert fc_stack.num_layers == 0
    assert fc_stack.input_shape == torch.Size([2])
    assert fc_stack.output_shape == torch.Size([2])

    fc_stack = FCStack(
        first_layer_input_size=2,
        layers=[
            {"input
```

Generated test for [`ludwig_example1.py`](https://github.com/jimthompson5802/gh_copilot/blob/main/codex/sample_code/ludwig_example1.py). 

Noted issues in generated code: 
* The `forward()` method is not tested.
* no assertions are generated
* Missing required imports
* truncated code

```python

 test_stacked_cnn():
    from ray.rllib.utils.test_utils import check, framework_iterator
    from ray.rllib.utils.framework import try_import_torch
    from ray.rllib.utils.torch_ops import convert_to_torch_tensor

    torch, nn = try_import_torch()

    for _ in framework_iterator(frameworks=("torch", "tf")):
        # Test default config
        config = Stacked2DCNN.get_schema_cls().sample_config()
        encoder = Stacked2DCNN(**config)
        check(encoder.input_shape, (3, 84, 84))
        check(encoder.output_shape, (128,))

        # Test with custom config
        config = Stacked2DCNN.get_schema_cls().sample_config()
        config["conv_layers"] = [
            {"out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1},
            {"out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1},
            {"out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1},
        ]
        config["num_conv_layers"] = None
        config["num_channels"] = 3
        config["out_channels"] = None
        config["kernel_size"] = None
        config["stride"] = None
        config["padding"] = None
        config["dilation"] = None
        config["conv_use_bias"] = None
        config["padding_mode"] = None
        config["conv_norm"] = None
        config["conv_norm_params"] = None
        config["conv_activation"] = None
        config["conv_dropout"] = None
        config["pool_function"] = None
        config["pool_kernel_size"] = None
        config["pool_stride"] = None
        config["pool_padding"] = None
        config["pool_dilation"] = None
        config["groups"] = None
        config["
```

## Second iteration of generating unit tests
Upon changing the prompt to:
```python
# Unit test forward method
def
```
and changing Codex api parameters:
* `temperature=0` to `temperature=0.1`
* `max_tokens=256` to `max_tokens=512`

the generated unit tests are better but still require some manual fix-up.

Generated test for [`ludwig_example0.py`](https://github.com/jimthompson5802/gh_copilot/blob/main/codex/sample_code/ludwig_example0.py) with updated prompt.  This new test performs these expected operations:
* generates synthetic data for testing
* Executes the `forward()` method with the synthetic data
* tests the output shape

Noted issues in generated code: 
* generated code is truncated
* Multiple tests generated, could be handled via `@pytest.mark.parametrize`
* Missing required imports

```python 
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
```

Generated test for [`ludwig_example1.py`](https://github.com/jimthompson5802/gh_copilot/blob/main/codex/sample_code/ludwig_example1.py) with updated prompt.  This new test performs these expected operations:
* generates synthetic data for testing
* Executes the `forward()` method with the synthetic data
* tests the output shape

Noted issues in generated code: 
* generated code is truncated
* Wrong dtype for synthetic data
* Does not account for `batch_size` when generating synthetic data or testing output shape
* missing required imports

```python
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

```