# Module:`attention_modules.py` Overview

The code is a module that implements various attention mechanisms commonly used in transformer models. 

The module includes the following classes:

1. `FeedForwardAttentionReducer`: Implements a feed-forward attention reducer that reduces the input sequence by applying attention weights to each element of the sequence.

2. `MultiHeadSelfAttention`: Implements a multi-head self-attention mechanism that applies attention to the input sequence using multiple attention heads.

3. `TransformerBlock`: Implements a transformer block that consists of a self-attention layer followed by a feed-forward neural network layer.

4. `TransformerStack`: Implements a stack of transformer blocks to create a transformer model.

Each class is a subclass of `LudwigModule`, which is a custom base class that provides some utility functions for working with PyTorch models.

## Class **`FeedForwardAttentionReducer`** Overview
The `FeedForwardAttentionReducer` class is a Python class that extends the `LudwigModule` class. It is used for reducing the input sequence using feed-forward attention mechanism.

The class has the following attributes:
- `fc_layer1`: A linear layer that maps the input size to the hidden size.
- `fc_layer1_activation`: An activation function applied to the output of `fc_layer1`.
- `fc_layer2`: A linear layer that maps the hidden size to 1, with no bias.
- `input_shape_var`: A variable to store the shape of the input.
- `output_shape_var`: A variable to store the shape of the output.

The class has the following methods:
- `forward`: Performs the forward pass of the attention reducer. It takes the inputs and an optional mask as input and returns the reduced inputs. It applies the feed-forward attention mechanism to the inputs, computes the attention weights, and applies the weights to the inputs to obtain the reduced inputs.
- `input_shape`: A property that returns the shape of the input.
- `output_shape`: A property that returns the shape of the output.

### Method **`__init__`** Overview
The `__init__` method is a special method in Python classes that is automatically called when an object of the class is created. It is used to initialize the attributes of the object.

In this specific code snippet, the `__init__` method is defined with three parameters: `input_size`, `hidden_size`, and `activation`. Here is the purpose of each parameter:

- `input_size`: It represents the size of the input to the neural network model.
- `hidden_size`: It represents the number of neurons in the hidden layer of the neural network model. It has a default value of 256 if not provided.
- `activation`: It represents the activation function to be used in the neural network model. It has a default value of "tanh" if not provided.

Now, let's discuss the mathematical operations or procedures performed in this `__init__` method:

1. `self.fc_layer1 = nn.Linear(input_size, hidden_size)`: This line creates a fully connected linear layer (`nn.Linear`) with `input_size` as the input dimension and `hidden_size` as the output dimension. It assigns this layer to the attribute `self.fc_layer1`.

2. `self.fc_layer1_activation = get_activation(activation)`: This line assigns the activation function specified by the `activation` parameter to the attribute `self.fc_layer1_activation`. The `get_activation` function is used to retrieve the appropriate activation function based on the provided activation name.

3. `self.fc_layer2 = nn.Linear(hidden_size, 1, bias=False)`: This line creates another fully connected linear layer (`nn.Linear`) with `hidden_size` as the input dimension and 1 as the output dimension. It assigns this layer to the attribute `self.fc_layer2`. The `bias` parameter is set to `False`, indicating that no bias term should be added to this layer.

4. `self.input_shape_var = None`: This line initializes the attribute `self.input_shape_var` to `None`. This attribute can be used to store the shape of the input data.

5. `self.output_shape_var = None`: This line initializes the attribute `self.output_shape_var` to `None`. This attribute can be used to store the shape of the output data.

Here is the LaTex code for the equations represented by the mathematical operations:

1. $self.fc\_layer1 = nn.Linear(input\_size, hidden\_size)$
2. $self.fc\_layer1\_activation = get\_activation(activation)$
3. $self.fc\_layer2 = nn.Linear(hidden\_size, 1, bias=False)$
4. $self.input\_shape\_var = None$
5. $self.output\_shape\_var = None$

### Method **`forward`** Overview
The `forward` method is a method defined in a Python class. It takes two parameters: `inputs` and `mask`. 

The purpose of the `forward` method is to perform a series of mathematical operations on the `inputs` and return the result. Here is a breakdown of the mathematical operations performed:

1. `self.input_shape_var = inputs.size()[1:]`: This line of code assigns the shape of the `inputs` tensor to the `input_shape_var` variable. The shape is obtained using the `size()` method of the `inputs` tensor.

2. `hidden = self.fc_layer1(inputs)`: This line of code applies a fully connected layer (`fc_layer1`) to the `inputs` tensor. The result is assigned to the `hidden` variable.

3. `hidden = self.fc_layer1_activation(hidden)`: This line of code applies an activation function to the `hidden` tensor. The specific activation function is not specified in the code snippet.

4. `hidden = self.fc_layer2(hidden)`: This line of code applies another fully connected layer (`fc_layer2`) to the `hidden` tensor. The result is assigned back to the `hidden` variable.

5. `attention = F.softmax(hidden, dim=1)`: This line of code applies the softmax function to the `hidden` tensor along the second dimension (`dim=1`). The result is assigned to the `attention` variable.

6. `gated_inputs = torch.sum(attention * inputs, dim=1)`: This line of code performs element-wise multiplication between the `attention` tensor and the `inputs` tensor, and then sums the result along the second dimension (`dim=1`). The result is assigned to the `gated_inputs` variable.

7. `self.output_shape_var = gated_inputs.size()[1:]`: This line of code assigns the shape of the `gated_inputs` tensor to the `output_shape_var` variable. The shape is obtained using the `size()` method of the `gated_inputs` tensor.

8. `return gated_inputs`: This line of code returns the `gated_inputs` tensor as the output of the `forward` method.

Here is the LaTex code for the mathematical operations:

1. $self.input\_shape\_var = inputs.size()[1:]$
2. $hidden = self.fc\_layer1(inputs)$
3. $hidden = self.fc\_layer1\_activation(hidden)$
4. $hidden = self.fc\_layer2(hidden)$
5. $attention = F.softmax(hidden, dim=1)$
6. $gated\_inputs = torch.sum(attention * inputs, dim=1)$
7. $self.output\_shape\_var = gated\_inputs.size()[1:]$
8. $return gated\_inputs$

### Method **`input_shape`** Overview
The `input_shape` method in Python is used to retrieve the shape of the input data for a neural network model. It is typically used in deep learning frameworks such as PyTorch.

The method does not take any parameters. It is a member function of a class, and the `self` parameter refers to the instance of the class.

The purpose of the `input_shape` method is to provide information about the shape of the input data to the model. This is important because the shape of the input determines the number of input neurons in the neural network.

The method returns the `input_shape_var`, which is a variable that holds the shape of the input data. The shape is represented as a `torch.Size` object, which is a subclass of the Python `tuple` type.

The `input_shape` method does not perform any mathematical operations or procedures. It simply returns the value of the `input_shape_var` variable.

Here is an example of how the `input_shape` method can be used:

```python
import torch

class MyModel:
    def __init__(self, input_shape):
        self.input_shape_var = input_shape

    def input_shape(self) -> torch.Size:
        return self.input_shape_var

# Create an instance of the model
model = MyModel(torch.Size([32, 32, 3]))

# Get the input shape
shape = model.input_shape()

print(shape)  # Output: torch.Size([32, 32, 3])
```

In this example, the `MyModel` class has an `input_shape` method that returns the value of the `input_shape_var` variable. The `input_shape_var` is initialized with the shape of the input data, which is `[32, 32, 3]` in this case. The `input_shape` method is then called on an instance of the model to retrieve the input shape. The shape is printed, resulting in `torch.Size([32, 32, 3])`.

### Method **`output_shape`** Overview
The `output_shape` method in Python is used to retrieve the output shape of a tensor or a variable. It returns the output shape as a `torch.Size` object.

Parameters:
- `self`: It represents the instance of the class that the method belongs to.

Return:
- `torch.Size`: It is a class representing the size of a tensor or a variable.

Mathematical Operations:
The `output_shape` method does not perform any mathematical operations or procedures. It simply returns the `output_shape_var`, which is a variable that holds the output shape of the tensor or variable.

LaTeX Code:
The `output_shape` method does not involve any mathematical operations, so there is no need for LaTeX code to display equations.

## Class **`MultiHeadSelfAttention`** Overview
The `MultiHeadSelfAttention` class is a Python class that extends the `LudwigModule` class. It is used to implement multi-head self-attention in a neural network model.

The class has the following attributes:
- `embedding_size`: The size of the hidden state or embedding.
- `num_heads`: The number of attention heads.
- `projection_dim`: The dimension of each attention head.
- `query_dense`, `key_dense`, `value_dense`: Linear layers used to project the input to the query, key, and value spaces.
- `combine_heads`: Linear layer used to combine the outputs of the attention heads.

The class has the following methods:
- `attention`: Performs the attention mechanism by calculating the attention scores, applying a mask if provided, and computing the weighted sum of the values.
- `separate_heads`: Reshapes the inputs to separate the heads and permutes the dimensions.
- `forward`: Implements the forward pass of the multi-head self-attention layer. It projects the inputs to the query, key, and value spaces, separates the heads, applies the attention mechanism, combines the outputs, and returns the projected outputs.
- `output_shape`: Returns the output shape of the layer.

Overall, the `MultiHeadSelfAttention` class provides a modular implementation of multi-head self-attention that can be easily integrated into a larger neural network model.

### Method **`__init__`** Overview
The `__init__` method is a special method in Python classes that is automatically called when an object of the class is created. In the context of the provided code snippet, this method is used to initialize the parameters and layers of a transformer model.

The method takes three parameters:
- `self`: It is a reference to the instance of the class itself. It is used to access the attributes and methods of the class.
- `input_size`: It represents the size of the input to the transformer model.
- `hidden_size`: It represents the size of the hidden layers in the transformer model.
- `num_heads`: It represents the number of attention heads in the transformer model. It has a default value of 8.

The purpose of the `__init__` method is to set up the initial state of the object. In this case, it initializes various attributes and layers of the transformer model. Here is a breakdown of the mathematical operations or procedures performed in the `__init__` method:

1. Check if `hidden_size` is divisible by `num_heads`:
   - If `hidden_size` is not divisible by `num_heads`, a `ValueError` is raised with an error message indicating that the `hidden_size` should be divisible by `num_heads`.

2. Calculate the `projection_dim`:
   - The `projection_dim` is calculated by dividing `hidden_size` by `num_heads`. This represents the size of each attention head.

3. Initialize the query, key, and value linear layers:
   - The `query_dense`, `key_dense`, and `value_dense` layers are initialized using the `nn.Linear` class from the PyTorch library. These layers are used to project the input to the transformer model into the query, key, and value spaces, respectively.

4. Initialize the combine heads linear layer:
   - The `combine_heads` layer is initialized using the `nn.Linear` class. This layer is used to combine the outputs of the attention heads into a single output.

The provided code snippet does not include any mathematical operations or procedures beyond the initialization of the layers and attributes.

### Method **`attention`** Overview
The `attention` method in Python is used to perform attention mechanism calculations. It takes four parameters: `query`, `key`, `value`, and an optional `mask`.

- `query`: This parameter represents the query tensor. It is used to calculate the similarity between the query and the key.
- `key`: This parameter represents the key tensor. It is used to calculate the similarity between the query and the key.
- `value`: This parameter represents the value tensor. It is used to calculate the weighted sum of values based on the attention weights.
- `mask` (optional): This parameter represents an optional mask tensor. It is used to mask certain elements of the attention scores.

The method performs the following mathematical operations:

1. Calculate the attention scores by performing matrix multiplication between the `query` and the transpose of `key`:
   
$$
   \text{{score}} = \text{{torch.matmul}}(query, key.\text{{permute}}(0, 1, 3, 2))
   $$

2. Calculate the dimension of the `key` tensor and convert it to a float tensor:
   
$$
   \text{{dim\_key}} = \text{{torch.tensor}}(key.\text{{shape}}[-1]).\text{{type}}(\text{{torch.float32}})
   $$

3. Scale the attention scores by dividing them by the square root of the dimension of the `key` tensor:
   
$$
   \text{{scaled\_score}} = \frac{{\text{{score}}}}{{\sqrt{{\text{{dim\_key}}}}}}
   $$

4. If a `mask` tensor is provided, element-wise multiply the `mask` tensor with the `scaled_score` tensor:
   
$$
   \text{{scaled\_score}} = \text{{mask}} \times \text{{scaled\_score}}
   $$

5. Apply the softmax function along the last dimension of the `scaled_score` tensor to obtain the attention weights:
   
$$
   \text{{weights}} = \text{{F.softmax}}(\text{{scaled\_score}}, \text{{dim}}=-1)
   $$

6. Calculate the weighted sum of the `value` tensor using the attention weights:
   
$$
   \text{{output}} = \text{{torch.matmul}}(\text{{weights}}, \text{{value}})
   $$

7. Return the `output` tensor and the `weights` tensor as the result of the method.

### Method **`separate_heads`** Overview
The `separate_heads` method in Python is used to separate the heads of a multi-head attention mechanism. It takes three parameters:

1. `self`: This parameter refers to the instance of the class that the method belongs to. It is used to access the attributes and methods of the class.

2. `inputs`: This parameter represents the input tensor that needs to be separated into heads. It is expected to be a 3-dimensional tensor of shape `(batch_size, sequence_length, hidden_size)`.

3. `batch_size`: This parameter specifies the size of the batch in the input tensor.

The method performs the following mathematical operations:

1. Reshaping the input tensor: The `inputs` tensor is reshaped using the `torch.reshape` function. The reshaping operation converts the tensor from a 3-dimensional shape `(batch_size, sequence_length, hidden_size)` to a 4-dimensional shape `(batch_size, sequence_length, num_heads, projection_dim)`. The `num_heads` and `projection_dim` are attributes of the class to which the method belongs.

2. Permuting the dimensions: The `torch.permute` function is used to permute the dimensions of the reshaped tensor. The dimensions are rearranged according to the provided permutation `(0, 2, 1, 3)`. This permutation swaps the second and third dimensions of the tensor, effectively separating the heads. The resulting tensor has a shape of `(batch_size, num_heads, sequence_length, projection_dim)`.

The LaTex code for the mathematical operations can be represented as follows:

1. Reshaping the input tensor:

$$
\text{{inputs}} = \text{{torch.reshape}}(\text{{inputs}}, (\text{{batch\_size}}, -1, \text{{self.num\_heads}}, \text{{self.projection\_dim}}))
$$

2. Permuting the dimensions:

$$
\text{{return torch.permute}}(\text{{inputs}}, (0, 2, 1, 3))
$$

### Method **`forward`** Overview
The `forward` method is a part of a Python class and is used to perform forward pass computations in a neural network model. It takes two parameters: `inputs` and `mask`. 

- `inputs` is a tensor representing the input data to the model. It has a shape of `[batch_size, seq_len, embedding_dim]`, where `batch_size` is the number of samples in a batch, `seq_len` is the length of the input sequence, and `embedding_dim` is the dimensionality of the input embeddings.
- `mask` is an optional parameter that can be used to mask certain elements of the input sequence. It is used to prevent the model from attending to padded or masked positions in the input.

The method performs the following mathematical operations or procedures:

1. It applies three linear transformations (`self.query_dense`, `self.key_dense`, `self.value_dense`) to the input tensor `inputs` to obtain query, key, and value tensors. These transformations project the input embeddings into a higher-dimensional space. The resulting tensors have a shape of `(batch_size, seq_len, h)`, where `h` is the projection dimension.

2. It separates the query, key, and value tensors into multiple heads using the `separate_heads` method. This is done to enable parallel processing and capture different aspects of the input. The resulting tensors have a shape of `(batch_size, num_heads, seq_len, projection_dim)`, where `num_heads` is the number of attention heads.

3. It applies the attention mechanism (`self.attention`) to the query, key, and value tensors to compute the attention weights and the attended outputs. The attention mechanism calculates the importance of each element in the key tensor with respect to the query tensor and uses these weights to compute a weighted sum of the value tensor. The resulting tensors have a shape of `(batch_size, seq_len, num_heads, projection_dim)`.

4. It permutes the dimensions of the attended outputs using `torch.permute` to obtain a tensor with a shape of `(batch_size, seq_len, num_heads, projection_dim)`.

5. It reshapes the tensor obtained in the previous step using `torch.reshape` to obtain a tensor with a shape of `(batch_size, seq_len, h)`.

6. It applies the `combine_heads` method to the reshaped tensor to combine the multiple heads back into a single tensor. The resulting tensor has a shape of `(batch_size, seq_len, h)`.

7. It returns the projected outputs tensor.

The mathematical operations or procedures performed by the `forward` method can be represented using LaTeX code as follows:


$$
\text{{query}} = \text{{self.query\_dense}}(\text{{inputs}})
$$

$$
\text{{key}} = \text{{self.key\_dense}}(\text{{inputs}})
$$

$$
\text{{value}} = \text{{self.value\_dense}}(\text{{inputs}})
$$

$$
\text{{query}} = \text{{separate\_heads}}(\text{{query}}, \text{{batch\_size}})
$$

$$
\text{{key}} = \text{{separate\_heads}}(\text{{key}}, \text{{batch\_size}})
$$

$$
\text{{value}} = \text{{separate\_heads}}(\text{{value}}, \text{{batch\_size}})
$$

$$
\text{{outputs}}, \text{{weights}} = \text{{attention}}(\text{{query}}, \text{{key}}, \text{{value}}, \text{{mask}})
$$

$$
\text{{outputs}} = \text{{torch.permute}}(\text{{outputs}}, (0, 2, 1, 3))
$$

$$
\text{{concat\_outputs}} = \text{{torch.reshape}}(\text{{outputs}}, (\text{{batch\_size}}, -1, \text{{self.embedding\_size}}))
$$

$$
\text{{projected\_outputs}} = \text{{combine\_heads}}(\text{{concat\_outputs}})
$$

$$
\text{{return projected\_outputs}}
$$

### Method **`output_shape`** Overview
The `output_shape` method in Python returns the output shape of a tensor. It is a method of a class and is defined as follows:

```python
def output_shape(self):
    return torch.Size([self.embedding_size])
```

The purpose of this method is to provide the shape of the output tensor. The `self.embedding_size` parameter represents the size of the embedding dimension.

The method returns a `torch.Size` object, which is a tuple representing the shape of the tensor. In this case, the output shape is a 1-dimensional tensor with a size equal to `self.embedding_size`.

The mathematical operations or procedures performed by this method are minimal. It simply returns the shape of the tensor without any mathematical calculations. Therefore, there is no need for LaTex code to display equations in this case.

## Class **`TransformerBlock`** Overview
The `TransformerBlock` class is a subclass of `LudwigModule` in Python. It is used to implement a single block of the Transformer model. 

The class has the following attributes:
- `input_size`: an integer representing the size of the input.
- `max_sequence_length`: an integer representing the maximum sequence length.
- `hidden_size`: an integer representing the size of the hidden layer.
- `num_heads`: an integer representing the number of attention heads.
- `output_size`: an integer representing the size of the output.
- `dropout`: a float representing the dropout rate.

The class has the following methods:
- `__init__()`: initializes the TransformerBlock object with the given parameters.
- `input_shape()`: returns the shape of the input tensor.
- `forward()`: performs the forward pass of the TransformerBlock, applying self-attention, dropout, layer normalization, and fully connected layers.
- `output_shape()`: returns the shape of the output tensor.

### Method **`__init__`** Overview
The `__init__` method is a special method in Python classes that is automatically called when an object of the class is created. It is used to initialize the attributes of the object.

Parameters:
- `self`: The first parameter of any method in a class, which refers to the instance of the class itself.
- `input_size`: An integer representing the size of the input.
- `max_sequence_length`: An integer representing the maximum length of the input sequence.
- `hidden_size`: An integer representing the size of the hidden layer.
- `num_heads`: An integer representing the number of attention heads.
- `output_size`: An integer representing the size of the output.
- `dropout`: A float representing the dropout rate (default value is 0.1).

Mathematical operations or procedures:
1. Initialize the attributes `input_size`, `max_sequence_length`, and `hidden_size` with the corresponding parameter values.
2. Create an instance of the `MultiHeadSelfAttention` class called `self_attention`, passing the `input_size`, `hidden_size`, and `num_heads` parameters.
3. Create an instance of the `nn.Dropout` class called `dropout1`, passing the `dropout` parameter.
4. Create an instance of the `nn.LayerNorm` class called `layernorm1`, passing the `hidden_size` and `eps` parameters.
5. Create a sequential neural network called `fully_connected` with the following layers:
   - Linear layer with input size `input_size`, output size `output_size`.
   - Activation function (ReLU).
   - Linear layer with input size `output_size`, output size `hidden_size`.
6. Create an instance of the `nn.Dropout` class called `dropout2`, passing the `dropout` parameter.
7. Create an instance of the `nn.LayerNorm` class called `layernorm2`, passing the `hidden_size` and `eps` parameters.

The `__init__` method sets up the initial state of the object by initializing its attributes and creating instances of other classes that will be used in the main functionality of the object.

### Method **`input_shape`** Overview
The `input_shape` method in Python is used to determine the shape of the input tensor for a neural network model. It is typically used in deep learning frameworks like PyTorch.

The method does not take any parameters. It is a member function of a class and is called on an instance of that class.

The purpose of the `input_shape` method is to return the shape of the input tensor. In this specific example, the method returns a `torch.Size` object with two dimensions: `self.max_sequence_length` and `self.input_size`.

The `self.max_sequence_length` parameter represents the maximum length of the input sequence. It is an integer value that determines the number of time steps in the input tensor.

The `self.input_size` parameter represents the size of each input element. It is an integer value that determines the number of features or dimensions in each time step of the input tensor.

The method performs the following mathematical operations or procedures:

1. Create a `torch.Size` object: The method creates a `torch.Size` object using the `torch.Size()` constructor.

2. Set the dimensions of the `torch.Size` object: The method sets the dimensions of the `torch.Size` object to `[self.max_sequence_length, self.input_size]`. This means that the resulting shape of the input tensor will have `self.max_sequence_length` time steps and `self.input_size` features or dimensions in each time step.

Here is the LaTex code to display the equations in a markdown document:


$$
\text{{input\_shape}}(self) \rightarrow \text{{torch.Size}}([self.max\_sequence\_length, self.input\_size])
$$

### Method **`forward`** Overview
The `forward` method is a part of a Python class and is used to perform forward propagation in a neural network. It takes two parameters: `inputs` and `mask`. 

- `inputs`: A tensor representing the input data to the neural network. It has a shape of `[b, s, h]`, where `b` is the batch size, `s` is the sequence length, and `h` is the hidden size.
- `mask`: An optional tensor representing a mask for the input data. It has the same shape as `inputs` and is used to mask certain elements of the input.

The method performs the following mathematical operations:

1. `attn_output = self.self_attention(inputs)`: This line applies self-attention mechanism to the input data. It computes the attention weights for each element in the sequence and applies them to the input data. The resulting tensor `attn_output` has the same shape as `inputs` (`[b, s, h]`).

2. `attn_output = self.dropout1(attn_output)`: This line applies dropout regularization to the `attn_output` tensor. Dropout randomly sets a fraction of the tensor elements to zero, which helps prevent overfitting. The resulting tensor `attn_output` has the same shape as before (`[b, s, h]`).

3. `ln1_output = self.layernorm1(inputs + attn_output)`: This line applies layer normalization to the sum of `inputs` and `attn_output`. Layer normalization normalizes the values of the tensor along the hidden dimension (`h`). The resulting tensor `ln1_output` has the same shape as `inputs` (`[b, s, h]`).

4. `fc_output = self.fully_connected(ln1_output)`: This line applies a fully connected layer to the `ln1_output` tensor. A fully connected layer applies a linear transformation to the tensor, followed by an activation function. The resulting tensor `fc_output` has the same shape as `ln1_output` (`[b, s, h]`).

5. `fc_output = self.dropout2(fc_output)`: This line applies dropout regularization to the `fc_output` tensor, similar to step 2. The resulting tensor `fc_output` has the same shape as before (`[b, s, h]`).

6. `return self.layernorm2(ln1_output + fc_output)`: This line applies layer normalization to the sum of `ln1_output` and `fc_output`. The resulting tensor is returned as the output of the `forward` method and has the same shape as `ln1_output` (`[b, s, h]`).

Here is the LaTex code for the equations performed in the `forward` method:


$$
\text{{attn\_output}} = \text{{self.self\_attention}}(\text{{inputs}})
$$

$$
\text{{attn\_output}} = \text{{self.dropout1}}(\text{{attn\_output}})
$$

$$
\text{{ln1\_output}} = \text{{self.layernorm1}}(\text{{inputs}} + \text{{attn\_output}})
$$

$$
\text{{fc\_output}} = \text{{self.fully\_connected}}(\text{{ln1\_output}})
$$

$$
\text{{fc\_output}} = \text{{self.dropout2}}(\text{{fc\_output}})
$$

$$
\text{{output}} = \text{{self.layernorm2}}(\text{{ln1\_output}} + \text{{fc\_output}})
$$

### Method **`output_shape`** Overview
The `output_shape` method in Python returns the shape of the output tensor produced by a model or layer. It is defined as a method within a class and takes no parameters other than the instance itself (`self`). The method returns a `torch.Size` object, which represents the shape of a tensor in PyTorch.

The purpose of the `output_shape` method is to provide information about the shape of the output tensor produced by a model or layer. This can be useful for understanding the dimensions of the output and for performing subsequent operations or calculations based on the output shape.

The method itself does not perform any mathematical operations or procedures. It simply returns a `torch.Size` object that represents the shape of the output tensor. The shape is defined as `[self.max_sequence_length, self.hidden_size]`, where `self.max_sequence_length` and `self.hidden_size` are attributes or variables specific to the model or layer. The `max_sequence_length` represents the maximum length of a sequence or input, while the `hidden_size` represents the size or dimensionality of the hidden state or output.

To display the equation in LaTeX format, the output shape can be represented as:


$$
\text{{output\_shape}} = [\text{{max\_sequence\_length}}, \text{{hidden\_size}}]
$$

This equation represents the shape of the output tensor as a list of two elements, where the first element is the maximum sequence length and the second element is the hidden size.

## Class **`TransformerStack`** Overview
The `TransformerStack` class is a subclass of `LudwigModule` in Python. It is used to create a stack of transformer blocks for sequence processing tasks. 

The class has the following attributes:
- `input_size`: an integer representing the size of the input features.
- `max_sequence_length`: an integer representing the maximum sequence length.
- `hidden_size`: an integer representing the size of the hidden layers.
- `num_heads`: an integer representing the number of attention heads.
- `output_size`: an integer representing the size of the output features.
- `num_layers`: an integer representing the number of transformer blocks in the stack.
- `dropout`: a float representing the dropout rate.
- `supports_masking`: a boolean indicating whether the class supports masking.
- `layers`: a `nn.ModuleList` containing the transformer blocks in the stack.

The class has the following methods:
- `__init__()`: initializes the `TransformerStack` object and creates the transformer blocks based on the specified parameters.
- `input_shape()`: returns the shape of the input tensor.
- `forward()`: performs the forward pass of the transformer stack, applying each transformer block to the input tensor.
- `output_shape()`: returns the shape of the output tensor.

The `TransformerStack` class is designed to be used in sequence processing tasks, where the input is a sequence of features and the output is a transformed sequence.

### Method **`__init__`** Overview
The `__init__` method is a constructor method in Python that is used to initialize the attributes of an object when it is created. In the given code snippet, the `__init__` method is defined for a class that represents a Transformer model.

The purpose of each parameter in the `__init__` method is as follows:

- `self`: It is a reference to the instance of the class.
- `input_size`: An integer representing the size of the input.
- `max_sequence_length`: An integer representing the maximum sequence length.
- `hidden_size`: An integer representing the size of the hidden layers in the Transformer model. The default value is 256.
- `num_heads`: An integer representing the number of attention heads in the Transformer model. The default value is 8.
- `output_size`: An integer representing the size of the output. The default value is 256.
- `num_layers`: An integer representing the number of layers in the Transformer model. The default value is 1.
- `dropout`: A float representing the dropout rate. The default value is 0.1.
- `**kwargs`: Additional keyword arguments.

The mathematical operations or procedures performed in the `__init__` method are as follows:

1. Set `self.supports_masking` to `True`.
2. Assign the values of `max_sequence_length`, `input_size`, and `hidden_size` to the corresponding attributes of the object.
3. Create an empty list `self.layers` to store the Transformer blocks.
4. Initialize `prior_input_size` with the value of `input_size`.
5. Iterate `num_layers` times and create a `TransformerBlock` object with the given parameters. Append the created object to `self.layers` list. Update `prior_input_size` with the output shape of the current layer.
6. Iterate over `self.layers` and log the name of each layer using the `logger.debug` method.

The mathematical operations or procedures in the `__init__` method do not involve any explicit mathematical equations.

### Method **`input_shape`** Overview
The `input_shape` method in Python is used to determine the shape of the input tensor for a neural network model. It is typically used in deep learning frameworks like PyTorch.

The method does not take any parameters. It is a member function of a class and is called on an instance of that class.

The purpose of the `input_shape` method is to return the shape of the input tensor. In this specific example, the method returns a `torch.Size` object with two dimensions: `self.max_sequence_length` and `self.input_size`.

The `self.max_sequence_length` parameter represents the maximum length of the input sequence. It is an integer value that determines the number of time steps in the input tensor.

The `self.input_size` parameter represents the size of each input element. It is an integer value that determines the number of features or dimensions in each time step of the input tensor.

The method performs the following mathematical operations or procedures:

1. Create a `torch.Size` object: The method creates a `torch.Size` object using the `torch.Size()` constructor.

2. Set the dimensions of the `torch.Size` object: The method sets the dimensions of the `torch.Size` object to `[self.max_sequence_length, self.input_size]`. This means that the shape of the input tensor will have `self.max_sequence_length` time steps and `self.input_size` features or dimensions in each time step.

The mathematical operations or procedures can be represented using LaTeX code as follows:


$$
\text{{input\_shape}}() \rightarrow \text{{torch.Size}}([self.max\_sequence\_length, self.input\_size])
$$

This LaTeX code can be used to display the equations in a markdown document.

### Method **`forward`** Overview
The `forward` method in Python is a method used in neural networks to perform forward propagation. It takes two parameters: `inputs` and `mask`. 

The `inputs` parameter represents the input data that is fed into the neural network. It can be a single input or a batch of inputs, depending on the implementation.

The `mask` parameter is an optional parameter that is used for masking certain inputs during the forward propagation process. It is commonly used in tasks such as sequence modeling, where some elements in the input sequence need to be ignored.

The method starts by assigning the `inputs` to a variable called `hidden`. This variable represents the output of the previous layer or the input data if it is the first layer.

Then, it iterates over each layer in the neural network using a for loop. For each layer, it calls the layer's `__call__` method, passing the `hidden` variable as the input and the `mask` parameter if provided. This allows the layer to perform its operations on the input and produce an output.

The output of each layer becomes the input for the next layer in the loop. This process continues until all layers have been processed.

Finally, the method returns the final output, which represents the output of the last layer in the neural network.

As for the mathematical operations or procedures performed in the `forward` method, it depends on the specific implementation of the layers used in the neural network. Each layer can have its own set of mathematical operations, such as matrix multiplications, activation functions, and bias additions.

To document the mathematical operations or procedures performed in the `forward` method, you can provide specific details about the layers used in the neural network and their mathematical operations. For example, if the neural network includes a fully connected layer, you can describe the matrix multiplication and bias addition operations performed by the layer using LaTex code in a markdown document.

### Method **`output_shape`** Overview
The `output_shape` method in Python returns the shape of the output tensor produced by a model or layer. It is defined as a method within a class and takes no parameters other than the instance itself (`self`). The method returns a `torch.Size` object, which represents the shape of a tensor in PyTorch.

The purpose of the `output_shape` method is to provide information about the shape of the output tensor produced by a model or layer. This can be useful for understanding the dimensions of the output and for performing subsequent operations or calculations based on the output shape.

The method itself does not perform any mathematical operations or procedures. It simply returns a `torch.Size` object that represents the shape of the output tensor. The shape is defined as `[self.max_sequence_length, self.hidden_size]`, where `self.max_sequence_length` and `self.hidden_size` are attributes or variables specific to the instance of the class.

To display the equation in a markdown document using LaTeX, the following code can be used:


$$
\text{{output\_shape}}() \rightarrow \text{{torch.Size}}([\text{{self.max\_sequence\_length}}, \text{{self.hidden\_size}}])
$$

