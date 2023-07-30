# Documentation-to-Code

Set of experiments to convert documentation to code using [`gpt-engineer` package](https://github.com/AntonOsika/gpt-engineer).

## Setup
* Setup environment variable `OPENAI_API_KEY` with your OpenAI API key.
* Install `gpt-engineer` package.

## Examples
|Directory| Description                                                                                                              |
|---|--------------------------------------------------------------------------------------------------------------------------|
|gpt-engineer-example| Example of using `gpt-engineer` package to generate code from documentation.                                             |
|generate-synthetic-data| Example of using `gpt-engineer` package to generate code to generate synthetic regression data.                          |
| ordinary-least-square| Example of using `gpt-engineer` package to generate code to train an OLS model.                                          |
|torch-training| Example of using `gpt-engineer` package to generate code to train an PyTorch model on the synthetic regression data set. |
| web-application| Example of using `gpt-engineer` package to generate code to create a web application.  Not Working.                      |


## Sample execution


```
$ cd documentation-to-code
$ gpt-engineer --temperature 0.01 torch-training/ True gpt-3.5-turbo
```

## Observations

* Need to use `gpt-3.5-turbo-16k` model.  Depending on amount of code required, models with smaller token size will not generate all the required modules. 

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
  resolved by changing `model.save_model()` to `train.save_model()`.  Looks like `gpt-engineer` selected the wrong Python module for the `save_model()` function.

* `torch-training` revised prompt to follwing, worked w/o issue when using `gpt-3.5-turbo-16k` LLM
```text
write a python program to do the following:
read data from ../../generate-synthetic-data/workspace/data/synthetic_regression.csv into a pandas dataframe
Convert the pandas dataframe to a PyTorch dataset.  Create a 80% training and 20% test split.
Define a 4 layer neural network and train it on the dataset.  Target variable is "target".  print loss value at the end of every epoch.
Save the model.
```

model training run output
```text
root@4f2974789963:/opt/project/codex/documentation-to-code/torch-training/workspace# python main.py
/usr/local/lib/python3.9/site-packages/torch/nn/modules/loss.py:536: UserWarning: Using a target size (torch.Size([32])) that is different to the input size (torch.Size([32, 1])). This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.
  return F.mse_loss(input, target, reduction=self.reduction)
/usr/local/lib/python3.9/site-packages/torch/nn/modules/loss.py:536: UserWarning: Using a target size (torch.Size([16])) that is different to the input size (torch.Size([16, 1])). This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.
  return F.mse_loss(input, target, reduction=self.reduction)
Epoch 1: Loss = 2.9905717372894287
Epoch 2: Loss = 2.735131104787191
Epoch 3: Loss = 2.236255327860514
Epoch 4: Loss = 1.9517639875411987
Epoch 5: Loss = 1.4809595346450806
Epoch 6: Loss = 0.9442115823427836
Epoch 7: Loss = 0.47655657927195233
Epoch 8: Loss = 0.33064985275268555
Epoch 9: Loss = 0.49060186743736267
Epoch 10: Loss = 0.4889020025730133
/usr/local/lib/python3.9/site-packages/torch/nn/modules/loss.py:536: UserWarning: Using a target size (torch.Size([20])) that is different to the input size (torch.Size([20, 1])). This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.
  return F.mse_loss(input, target, reduction=self.reduction)
Test Loss: 0.32672378420829773
root@4f2974789963:/opt/project/codex/documentation-to-code/torch-training/workspace#
```

* `ordinary-least-square` example ran w/ some issues.  Had to minor code corrections, e.g, add missing imports and adjust synthetic data to be non-singular.
  * Used ChatGPT to generate the `gpt-engineer` prompt, "write a gpt-engineer prompt to create a python program to perform ordinary least square model." 
  * modified code to use same data set as `torch-training` example.  Results of the OLS model training with the synthetic data set:
```text
01d5073512b1:python -u /opt/project/codex/documentation-to-code/ordinary-least-square/workspace/main.py
Predicted values: [1.90100206 2.31704975 2.06071171 1.87331451 2.10176641]
Mean Squared Error: 0.17820806931665634
R-squared: 0.3322654901993485

Process finished with exit code 0
```