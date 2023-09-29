# Using Codex API

Example code for using Codex API

## Documentation:
* [Codex model](https://beta.openai.com/docs/models/codex)
* [Codex User Guide](https://beta.openai.com/docs/guides/code/code-completion-limited-beta)
* [Codex API documentation](https://beta.openai.com/docs/api-reference/code-completion)

## Contents
| File                 | Description                                                                                                   |
|----------------------|---------------------------------------------------------------------------------------------------------------|
| codex_api_example.py | Demonstrate invoking the Codex api to generate code from a prompt                                             |
| create_unit_test.py  | For all source modules in a specified directory, generate unit test and save in a specified output directory. |
| ludwig_unit_test_generation_results.md | Results of generating unit tests for Ludwig code using Codex API.                                             |
| translate_sas_to_python.py | Translate SAS code to Python code using Codex API.                                                            |

## Setup
`Dockerfile` in directory `docker` contains the setup for running the example code.  Run `build_image.sh` to create docker image and run example code in the container.  `run_codex.sh` starts a docker container with openai environment.


## Unit test example with Copilot X Chat Feature

Unit test for this Module: https://github.com/jimthompson5802/gh_copilot/blob/main/codex/sample_code/ludwig_example0.py

![](./unit_tests/images/copilot_x_unit_test_chat.png)

### Prompt Unit Test: FCStack
```
generate pytest unit test for FCStack. Parameterize for different input and output sizes and test one or more layers
```

### Generated code 1st Draft for FCStack

This version failed to run.  Instantiation of `FCStack` object contained invalid parameter specification.

```python
import torch
import pytest
from ludwig.modules.fully_connected_modules import FCStack

@pytest.mark.parametrize("layers", [
    [{"input_size": 10, "output_size": 5}],
    [{"input_size": 20, "output_size": 10}, {"output_size": 5}],
    [{"input_size": 30, "output_size": 15}, {"output_size": 10}, {"output_size": 5}]
])
def test_output_shape(layers):
    fc_stack = FCStack(layers)
    input_tensor = torch.randn(1, layers[0]["input_size"])
    output_tensor = fc_stack(input_tensor)
    expected_output_shape = torch.Size([1, layers[-1]["output_size"]])
    assert output_tensor.shape == expected_output_shape
```

### Modified code to run

A small change to the generated code is needed to run the unit test.  The first layer input size is not included in the layers list.  The first layer input size is needed to create the first layer of the FCStack.  The modified code is shown below.

```python
import torch
import pytest
from ludwig.modules.fully_connected_modules import FCStack

@pytest.mark.parametrize("layers", [
    [{"input_size": 10, "output_size": 5}],
    [{"input_size": 20, "output_size": 10}, {"output_size": 5}],
    [{"input_size": 30, "output_size": 15}, {"output_size": 10}, {"output_size": 5}]
])
def test_output_shape(layers):
    fc_stack = FCStack(first_layer_input_size=layers[0]["input_size"], layers=layers) # <== modified
    input_tensor = torch.randn(1, layers[0]["input_size"])
    output_tensor = fc_stack(input_tensor)
    expected_output_shape = torch.Size([1, layers[-1]["output_size"]])
    assert output_tensor.shape == expected_output_shape
```

### Second Prompt Unit Test: FCStack

```text
generate pytest unit test for FCStack. Parameterize for different input and output sizes and test one or more layers. Test combinations of the other parameters.
```

### Generated code 1st Draft for the second unit test prompt for FCStack

Generated code ran w/o issue.  No modifications were needed.  

```python
import torch
import pytest
from ludwig.modules.fully_connected_modules import FCStack

@pytest.mark.parametrize("input_size, output_size, layers, use_bias, activation", [
    (10, 5, [{"output_size": 5}], True, "relu"),
    (20, 5, [{"output_size": 10}, {"output_size": 5}], False, "sigmoid"),
    (30, 5, [{"output_size": 15}, {"output_size": 10}, {"output_size": 5}], True, "tanh")
])
def test_output_shape(input_size, output_size, layers, use_bias, activation):
    fc_stack = FCStack(input_size, layers=layers, default_output_size=output_size, default_use_bias=use_bias, default_activation=activation)
    input_tensor = torch.randn(1, input_size)
    output_tensor = fc_stack(input_tensor)
    expected_output_shape = torch.Size([1, output_size])
    assert output_tensor.shape == expected_output_shape
```

### Prompt Unit Test: FCLayer

```text
generate pytest unit test for FCLayer. Parameterize the parameters to create the object to test different combinations of values.
```

### Generated code 1st Draft for FCLayer
Generated 1st Draft Code worked w/o issue.  No modifications were needed.

```python
import torch
import pytest
from ludwig.modules.fully_connected_modules import FCLayer

@pytest.mark.parametrize("input_size, output_size, use_bias, activation", [
    (10, 5, True, "relu"),
    (20, 10, False, "sigmoid"),
    (30, 15, True, "tanh")
])
def test_output_shape(input_size, output_size, use_bias, activation):
    fc_layer = FCLayer(input_size=input_size, output_size=output_size, use_bias=use_bias, activation=activation)
    input_tensor = torch.randn(1, input_size)
    output_tensor = fc_layer(input_tensor)
    expected_output_shape = torch.Size([1, output_size])
    assert output_tensor.shape == expected_output_shape
```

### Comments about the generated unit tests

* Generated function name `test_output_shape` should be changed to something more descriptive, such as `test_fc_stack` and `test_fc_layer`.
* To make the tests more robust manually add a few more test cases by copy/paste the existing boiler plate code and changing parameters values.
* Since Ludwig is an open-source project, it may have been in the training data used to train Codex LLM that underlies GH Copilot.  This may explain why some of the generated code worked w/o issue.


### Actual Unit Test Code for FCLayer and FCStack in Ludwig

Following are the actual Ldwig unit tests for the two classes.

```python
BATCH_SIZE = 2
DEVICE = get_torch_device()
RANDOM_SEED = 1919


@pytest.mark.parametrize("input_size", [2, 3])
@pytest.mark.parametrize("output_size", [3, 4])
@pytest.mark.parametrize("activation", ["relu", "sigmoid", "tanh"])
@pytest.mark.parametrize("dropout", [0.0, 0.6])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("norm", [None, "layer", "batch", "ghost"])
def test_fc_layer(
    input_size: int,
    output_size: int,
    activation: str,
    dropout: float,
    batch_size: int,
    norm: Optional[str],
):
    set_random_seed(RANDOM_SEED)  # make repeatable
    fc_layer = FCLayer(
        input_size=input_size, output_size=output_size, activation=activation, dropout=dropout, norm=norm
    ).to(DEVICE)
    input_tensor = torch.randn(batch_size, input_size, device=DEVICE)
    output_tensor = fc_layer(input_tensor)
    assert output_tensor.shape[1:] == fc_layer.output_shape


@pytest.mark.parametrize(
    "first_layer_input_size,layers,num_layers",
    [
        (2, None, 3),
        (2, [{"output_size": 4}, {"output_size": 8}], None),
        (2, [{"input_size": 2, "output_size": 4}, {"output_size": 8}], None),
    ],
)
def test_fc_stack(
    first_layer_input_size: Optional[int],
    layers: Optional[List],
    num_layers: Optional[int],
):
    set_random_seed(RANDOM_SEED)
    fc_stack = FCStack(first_layer_input_size=first_layer_input_size, layers=layers, num_layers=num_layers).to(DEVICE)
    input_tensor = torch.randn(BATCH_SIZE, first_layer_input_size, device=DEVICE)
    output_tensor = fc_stack(input_tensor)
    assert output_tensor.shape[1:] == fc_stack.output_shape


def test_fc_stack_input_size_mismatch_fails():
    first_layer_input_size = 10
    layers = [{"input_size": 2, "output_size": 4}, {"output_size": 8}]

    fc_stack = FCStack(
        first_layer_input_size=first_layer_input_size,
        layers=layers,
    ).to(DEVICE)
    input_tensor = torch.randn(BATCH_SIZE, first_layer_input_size, device=DEVICE)

    with pytest.raises(RuntimeError):
        fc_stack(input_tensor)


def test_fc_stack_no_layers_behaves_like_passthrough():
    first_layer_input_size = 10
    layers = None
    num_layers = 0
    output_size = 15

    fc_stack = FCStack(
        first_layer_input_size=first_layer_input_size,
        layers=layers,
        num_layers=num_layers,
        default_output_size=output_size,
    ).to(DEVICE)
    input_tensor = torch.randn(BATCH_SIZE, first_layer_input_size, device=DEVICE)
    output_tensor = fc_stack(input_tensor)

    assert list(output_tensor.shape[1:]) == [first_layer_input_size]
    assert output_tensor.shape[1:] == fc_stack.output_shape
    assert np.all(np.isclose(input_tensor, output_tensor))
```
