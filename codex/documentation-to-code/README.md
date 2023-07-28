# Documentation-to-Code

Set of experiments to convert documentation to code using `gpt-engineer` package.

## Setup
* Setup environment variable `OPENAI_API_KEY` with your OpenAI API key.
* Install `gpt-engineer` package.

## Examples
|Directory| Description                                                                                                              |
|---|--------------------------------------------------------------------------------------------------------------------------|
|gpt-engineer-example| Example of using `gpt-engineer` package to generate code from documentation.                                             |
|generate-synthetic-data| Example of using `gpt-engineer` package to generate code to generate synthetic regression data.                          |
|torch-training| Example of using `gpt-engineer` package to generate code to train an PyTorch model on the synthetic regression data set. |


## Sample execution


```
$ cd documentation-to-code
$ gpt-engineer --temperature 0.01 torch-training/ True gpt-3.5-turbo
```

## Observations

* Need to use `gpt-3.5-turbo-16k` model.  Depending on amount of code required, models with smaller token size will not generate all the required modules. Update: this may not be relevant.  see following note about `temperature` option.

* default temperature 0.1 sometimes generates non-working code even with `gpt-3.5-turbo-16k` model.  setting temperature to 0.01.

* `generate-synthetic-data` example ran w/o issue.

* `torch-training` example encountered this error:
```text
root@4f2974789963:/opt/project/codex/documentation-to-code/torch-training/workspace# python main.py
/usr/local/lib/python3.9/site-packages/torch/nn/modules/loss.py:536: UserWarning: Using a target size (torch.Size([32])) that is different to the input size (torch.Size([32, 1])). This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.
  return F.mse_loss(input, target, reduction=self.reduction)
/usr/local/lib/python3.9/site-packages/torch/nn/modules/loss.py:536: UserWarning: Using a target size (torch.Size([4])) that is different to the input size (torch.Size([4, 1])). This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.
  return F.mse_loss(input, target, reduction=self.reduction)
Traceback (most recent call last):
  File "/opt/project/codex/documentation-to-code/torch-training/workspace/main.py", line 24, in <module>
    model.save_model(neural_network, "trained_model.pt")
AttributeError: module 'model' has no attribute 'save_model'
```
