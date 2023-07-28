# Module:`sequence_encoders.py` Overview

## **Error in generating module level documentation**

## Class **`SequenceEncoder`** Overview
The `SequenceEncoder` class is a subclass of the `Encoder` class in Python. It does not have any additional methods or attributes defined in it.

## Class **`SequencePassthroughEncoder`** Overview
The `SequencePassthroughEncoder` class is a subclass of the `SequenceEncoder` class in Python. It is used to encode input sequences and pass them through without any modification.

The class has the following attributes:
- `reduce_output`: A string that defines how to reduce the output tensor along the sequence length dimension if the rank of the tensor is greater than 2. Available values are: `sum`, `mean` or `avg`, `max`, `concat`, `last`, and `None` or `null`.
- `max_sequence_length`: An integer that represents the maximum sequence length.
- `encoding_size`: An integer that represents the size of the encoding vector, or None if sequence elements are scalars.

The class has the following methods:
- `__init__()`: Initializes the class and sets the attributes.
- `forward()`: Takes an input sequence and an optional mask as input and returns the encoded sequence.
- `get_schema_cls()`: Returns the schema class for the encoder.
- `input_shape()`: Returns the shape of the input tensor.
- `output_shape()`: Returns the shape of the output tensor.

### Method **`__init__`** Overview
The `__init__` method is the constructor method of a Python class. It is called when an object of the class is created. In this case, the method is defined with the following parameters:

- `self`: It is a reference to the current instance of the class.
- `reduce_output` (optional): It defines how to reduce the output tensor along the `s` sequence length dimension if the rank of the tensor is greater than 2. The available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension), and `None` or `null` (which does not reduce and returns the full tensor).
- `max_sequence_length` (optional): It specifies the maximum sequence length.
- `encoding_size` (optional): It represents the size of the encoding vector, or `None` if sequence elements are scalars.
- `encoder_config` (optional): It is a configuration object for the encoder.
- `**kwargs` (optional): It allows passing additional keyword arguments.

The purpose of the `__init__` method is to initialize the object of the class. In this specific implementation, it performs the following mathematical operations or procedures:

1. It calls the `__init__` method of the superclass using `super().__init__()`.
2. It assigns the `encoder_config` to the `config` attribute of the object.
3. It assigns the `max_sequence_length` parameter to the `max_sequence_length` attribute of the object.
4. It logs a debug message with the name of the object.
5. It assigns the `reduce_output` parameter to the `reduce_output` attribute of the object.
6. It creates an instance of the `SequenceReducer` class and assigns it to the `reduce_sequence` attribute of the object. The `reduce_mode`, `max_sequence_length`, and `encoding_size` parameters are passed to the `SequenceReducer` constructor.
7. If the `reduce_output` is `None`, it sets the `supports_masking` attribute of the object to `True`.

The mathematical operations or procedures performed by the `__init__` method do not involve any explicit mathematical equations.

### Method **`forward`** Overview
The `forward` method in Python is a method defined within a class. It takes two parameters: `input_sequence` and `mask`. 

The `input_sequence` parameter represents the input sequence fed into the encoder. It can have two possible shapes: [batch x sequence length] if the input is of type `torch.int32`, or [batch x sequence length x encoding size] if the input is of type `torch.float32`. 

The `mask` parameter represents a sequence mask, but it is not yet implemented in this method. It has a shape of [batch x sequence length].

Inside the method, the `input_sequence` is first converted to type `torch.float32` using the `type` method of the `torch` module. This ensures that the input sequence is of the correct type for further processing.

Next, a while loop is used to check the number of dimensions in the `input_sequence` tensor. If the number of dimensions is less than 3, the `unsqueeze` method is used to add an additional dimension at the end of the tensor. This is done using the `-1` index, which represents the last dimension of the tensor.

After the necessary dimensions are added, the `input_sequence` tensor is passed to the `reduce_sequence` method, which performs some mathematical operations or procedures on the input sequence. However, the details of this method are not provided in the given code snippet.

Finally, the method returns a dictionary with a single key-value pair. The key is "encoder_output" and the value is the `hidden` tensor, which represents the output of the encoder.

To document the mathematical operations or procedures performed by the `reduce_sequence` method, more information or the implementation of that method is required. Without that information, it is not possible to generate LaTeX code to display the equations in a markdown document.

### Method **`get_schema_cls`** Overview
The `get_schema_cls` method in Python returns the `SequencePassthroughConfig` class. This method does not take any parameters.

The purpose of the `get_schema_cls` method is to provide access to the `SequencePassthroughConfig` class, which is used for configuring the sequence passthrough functionality in a program.

As for the mathematical operations or procedures performed by this method, there are none. The method simply returns the `SequencePassthroughConfig` class without performing any mathematical calculations.

Here is the LaTex code for displaying the equations in a markdown document:

```latex

$$
\text{{def get\_schema\_cls():}}
$$

$$
\quad \text{{return SequencePassthroughConfig}}
$$
```

### Method **`input_shape`** Overview
The `input_shape` method in Python is used to determine the shape of the input data for a neural network model. It is typically used in the context of deep learning frameworks such as PyTorch.

The method does not take any parameters. It is a member function of a class, and the `self` parameter refers to the instance of the class on which the method is called.

The purpose of the `input_shape` method is to return the shape of the input data as a `torch.Size` object. In the given code snippet, the shape is represented as `[self.max_sequence_length]`, where `self.max_sequence_length` is a variable that holds the length of the input sequence.

The mathematical operations or procedures performed by this method are minimal. It simply returns the shape of the input data as a `torch.Size` object, which is a tuple-like structure that represents the dimensions of a tensor. No mathematical operations or equations are involved in this particular method.

To display the equation `[self.max_sequence_length]` in LaTeX format, you can use the following code in a markdown document:

```

$$
\text{{input\_shape}} = [self.max\_sequence\_length]
$$
```

This will render the equation as:


$$ \text{input\_shape} = [self.max\_sequence\_length] $$

### Method **`output_shape`** Overview
The `output_shape` method in Python is a method that returns the output shape of a tensor. It is defined within a class and takes no parameters other than the `self` parameter, which refers to the instance of the class.

The purpose of the `output_shape` method is to provide information about the shape of the tensor that will be produced as output by a particular operation or layer. This can be useful for understanding the dimensions of the data being processed and for performing subsequent operations or calculations.

The method returns the `input_shape` attribute of the instance, which is assumed to be a tensor shape represented as a `torch.Size` object. The `torch.Size` object is a subclass of the Python `tuple` class and represents the shape of a tensor as a tuple of integers.

The `output_shape` method does not perform any mathematical operations or procedures itself. It simply returns the `input_shape` attribute, which is assumed to have been set previously. The mathematical operations or procedures that determine the `input_shape` attribute would be implemented elsewhere in the code.

Here is an example of how the `output_shape` method can be used:

```python
import torch

class MyLayer:
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def output_shape(self) -> torch.Size:
        return self.input_shape

# Create an instance of the MyLayer class
layer = MyLayer(torch.Size([32, 64, 128]))

# Get the output shape of the layer
output_shape = layer.output_shape()

print(output_shape)  # Output: torch.Size([32, 64, 128])
```

In this example, the `MyLayer` class has an `input_shape` attribute that is set to `torch.Size([32, 64, 128])`. The `output_shape` method simply returns this `input_shape` attribute. When the method is called on an instance of the `MyLayer` class, it returns the `input_shape` attribute, which is then printed to the console.

## Class **`SequenceEmbedEncoder`** Overview
The `SequenceEmbedEncoder` class is a subclass of the `SequenceEncoder` class. It is used to encode input sequences using embeddings. 

The class has the following attributes:
- `vocab`: A list representing the vocabulary of the input feature to encode.
- `max_sequence_length`: An integer representing the maximum sequence length.
- `representation`: A string indicating the type of representation for the embeddings. It can be either "dense" or "sparse".
- `embedding_size`: An integer representing the maximum embedding size. The actual size will be the minimum of `vocabulary_size` and `embedding_size` for dense representations, and exactly `vocabulary_size` for sparse encoding.
- `embeddings_trainable`: A boolean indicating whether the embeddings should be trained during the training process.
- `pretrained_embeddings`: A string representing the path to a file containing pre-trained embeddings in the GloVe format.
- `embeddings_on_cpu`: A boolean indicating whether the embedding matrices should be stored on the CPU instead of the GPU.
- `weights_initializer`: A string or dictionary specifying the initializer to use for the weights.
- `dropout`: A float representing the dropout probability.
- `reduce_output`: A string indicating how to reduce the output tensor along the sequence length dimension if the rank of the tensor is greater than 2.
- `encoder_config`: A dictionary containing additional configuration parameters.

The class has the following methods:
- `__init__()`: Initializes the `SequenceEmbedEncoder` object with the given parameters.
- `forward()`: Performs the forward pass of the encoder. It takes the input sequence and optional input mask as input and returns the encoded sequence.
- `get_schema_cls()`: Returns the schema class for the encoder.
- `input_shape()`: Returns the input shape of the encoder.
- `output_shape()`: Returns the output shape of the encoder.

### Method **`__init__`** Overview
The `__init__` method is the constructor of a Python class. It is called when an object of the class is created. In this case, the method is part of a class that represents an encoder for sequence data.

The purpose of each parameter in the `__init__` method is as follows:

- `vocab`: Vocabulary of the input feature to encode. It is a list.
- `max_sequence_length`: The maximum sequence length. It is an integer.
- `representation`: The representation of the embeddings. It can be either "dense" or "sparse". If "dense", the embeddings are initialized randomly. If "sparse", the embeddings are initialized as one-hot encodings. It is a string.
- `embedding_size`: The maximum embedding size. The actual size will be the minimum between the vocabulary size and the embedding size for "dense" representations, and exactly the vocabulary size for "sparse" encoding. It is an integer.
- `embeddings_trainable`: A boolean value indicating whether the embeddings should be trained during the training process. If `True`, the embeddings are trainable. If `False`, the embeddings are fixed. This parameter only has an effect when the representation is "dense", as one-hot encodings are not trainable.
- `pretrained_embeddings`: A path to a file containing embeddings in the GloVe format. By default, dense embeddings are initialized randomly, but this parameter allows specifying a file with pre-trained embeddings. Only the embeddings with labels present in the vocabulary are kept, and the others are discarded. If the vocabulary contains strings that have no match in the embeddings file, their embeddings are initialized with the average of all other embeddings plus some random noise. This parameter only has an effect when the representation is "dense". It is a string (filepath).
- `embeddings_on_cpu`: A boolean value indicating whether the embedding matrices should be stored on CPU memory instead of GPU memory. This can be useful when the embedding matrix is very large. It slightly slows down the process due to data transfer between CPU and GPU memory. It is a boolean value.
- `weights_initializer`: The initializer to use for the weights. If `None`, the default initializer of each variable is used. There are several options for the initializer, such as `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. It is also possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters. It is a string or a dictionary.
- `dropout`: The dropout probability. It is a tensor of type `torch.float`.
- `reduce_output`: Defines how to reduce the output tensor along the sequence length dimension if the rank of the tensor is greater than 2. Available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension), and `None` or `null` (which does not reduce and returns the full tensor). It is a string.

The mathematical operations or procedures performed in the `__init__` method are as follows:

1. Set the `config` attribute of the class to the value of `encoder_config`.
2. Set the `embedding_size` attribute of the class to the value of `embedding_size`.
3. Set the `max_sequence_length` attribute of the class to the value of `max_sequence_length`.
4. Set the `reduce_output` attribute of the class to the value of `reduce_output`. If `reduce_output` is `None`, set the `supports_masking` attribute of the class to `True`.
5. Create an instance of the `EmbedSequence` class and assign it to the `embed_sequence` attribute of the class. Pass the appropriate parameters to the `EmbedSequence` constructor.
6. Create an instance of the `SequenceReducer` class and assign it to the `reduce_sequence` attribute of the class. Pass the appropriate parameters to the `SequenceReducer` constructor.

### Method **`forward`** Overview
The `forward` method in Python is a method defined within a class. In this case, it takes two parameters: `inputs` and `mask`. 

The `inputs` parameter represents the input sequence that is fed into the encoder. It is a tensor of shape `[batch x sequence length]` and has a data type of `torch.int32`.

The `mask` parameter is an optional input mask that is currently unused and not yet implemented in the `EmbedSequence` class.

Within the method, the `inputs` tensor is passed to the `embed_sequence` method, along with the `mask` parameter. This method is responsible for embedding the input sequence.

The embedded sequence is then passed to the `reduce_sequence` method, which performs some mathematical operations or procedures to reduce the sequence. However, the specific details of these operations are not provided in the given code snippet.

Finally, the output of the `reduce_sequence` method, which is stored in the `hidden` variable, is returned as a dictionary with the key "encoder_output".

Unfortunately, without further information about the `embed_sequence` and `reduce_sequence` methods, it is not possible to generate LaTeX code for the mathematical operations or procedures performed in this `forward` method.

### Method **`get_schema_cls`** Overview
The `get_schema_cls` method in Python returns the `SequenceEmbedConfig` class. This method does not take any parameters.

The purpose of the `get_schema_cls` method is to provide access to the `SequenceEmbedConfig` class, which is used for configuring the embedding of sequences in a machine learning model.

As for the mathematical operations or procedures performed by this method, there are none. The method simply returns the `SequenceEmbedConfig` class, which is a configuration class and does not involve any mathematical operations.

Here is the LaTex code for displaying the equations in a markdown document:

```latex

$$
\text{{def get\_schema\_cls():}}
$$

$$
\quad \text{{return SequenceEmbedConfig}}
$$
```

### Method **`input_shape`** Overview
The `input_shape` method in Python is used to determine the shape of the input data for a neural network model. It is typically used in the context of deep learning frameworks such as PyTorch.

The method does not take any parameters. It is a member function of a class, and the `self` parameter refers to the instance of the class on which the method is called.

The purpose of the `input_shape` method is to return the shape of the input data as a `torch.Size` object. In the given code snippet, the shape is represented as `[self.max_sequence_length]`, where `self.max_sequence_length` is a variable that holds the length of the input sequence.

The mathematical operations or procedures performed by this method are minimal. It simply returns the shape of the input data as a `torch.Size` object, which is a tuple-like object that represents the dimensions of a tensor. The shape is determined solely based on the value of `self.max_sequence_length`.

To represent the equation in LaTeX code, we can write:


$$
\text{{input\_shape}}() \rightarrow \text{{torch.Size}}([\text{{self.max\_sequence\_length}}])
$$

This equation shows that the `input_shape` method returns a `torch.Size` object with a single dimension, where the size of that dimension is equal to the value of `self.max_sequence_length`.

### Method **`output_shape`** Overview
The `output_shape` method in Python is used to retrieve the output shape of a sequence reduction operation. It is defined within a class and returns a `torch.Size` object.

Parameters:
- `self`: The instance of the class that the method is called on.

Purpose:
The purpose of the `output_shape` method is to provide information about the shape of the output tensor after performing a sequence reduction operation.

Mathematical Operations:
The `output_shape` method does not perform any mathematical operations itself. Instead, it retrieves the output shape from the `reduce_sequence` attribute of the class instance and returns it.

LaTeX Code:
No mathematical equations are involved in the `output_shape` method, so there is no need for LaTeX code to display equations in a markdown document.

## Class **`ParallelCNN`** Overview
The `ParallelCNN` class is a subclass of `SequenceEncoder` and represents a parallel convolutional neural network (CNN) encoder. It is used to encode input sequences into a fixed-length representation.

The class has several parameters that can be set during initialization, including options for embedding the input sequence, specifying the vocabulary, choosing the representation type (dense or sparse), setting the size of the embedding vectors, and controlling the training of the embeddings. Other parameters allow customization of the convolutional layers, fully connected layers, activation functions, dropout, and more.

The `ParallelCNN` class overrides the `forward` method to define the forward pass of the encoder. It first processes the input sequence by embedding it if necessary, then applies parallel 1D convolutional layers to extract features from the sequence. The output of the convolutional layers is then reduced along the sequence length dimension using a specified reduction method. If specified, fully connected layers are applied to further process the reduced output. The final output of the encoder is returned as a dictionary with the key "encoder_output".

The class also provides methods for getting the schema class, which is used for configuration, and for getting the input and output shapes of the encoder.

Overall, the `ParallelCNN` class provides a flexible and customizable implementation of a parallel CNN encoder for sequence encoding tasks.

### Method **`__init__`** Overview
The `__init__` method is the constructor of a Python class. It is called when an object of the class is created. In this case, the method is defined with multiple parameters that control the configuration of the class instance.

Here is a description of each parameter:

- `should_embed`: A boolean parameter that determines whether the input sequence should be embedded into dense vectors. If `True`, the input sequence is expected to be made of integers and will be mapped into embeddings. Default value is `True`.
- `vocab`: A list that represents the vocabulary of the input feature to encode. Default value is `None`.
- `representation`: A string parameter that specifies the representation of the embeddings. The possible values are `'dense'` and `'sparse'`. If `'dense'`, the embeddings are initialized randomly. If `'sparse'`, the embeddings are initialized to be one-hot encodings. Default value is `'dense'`.
- `embedding_size`: An integer parameter that represents the maximum embedding size. The actual size will be `min(vocabulary_size, embedding_size)` for `'dense'` representations and exactly `vocabulary_size` for the `'sparse'` encoding, where `vocabulary_size` is the number of different strings appearing in the training set in the column the feature is named after (plus 1 for `<UNK>`). Default value is `256`.
- `max_sequence_length`: An integer parameter that represents the maximum length of the input sequence. Default value is `None`.
- `embeddings_trainable`: A boolean parameter that determines whether the embeddings are trainable during the training process. If `True`, embeddings are trained. If `False`, embeddings are fixed. Default value is `True`.
- `pretrained_embeddings`: A string parameter that specifies a path to a file containing embeddings in the GloVe format. When the file containing the embeddings is loaded, only the embeddings with labels present in the vocabulary are kept, and the others are discarded. If the vocabulary contains strings that have no match in the embeddings file, their embeddings are initialized with the average of all other embeddings plus some random noise to make them different from each other. This parameter has effect only if `representation` is `'dense'`. Default value is `None`.
- `embeddings_on_cpu`: A boolean parameter that determines whether the embedding matrix is stored on CPU memory instead of GPU memory. This may be useful when the embedding matrix is really big and storing it on GPU memory slows down the process due to data transfer between CPU and GPU memory. Default value is `False`.
- `conv_layers`: A list of dictionaries that contains the parameters of all the convolutional layers. The length of the list determines the number of parallel convolutional layers, and the content of each dictionary determines the parameters for a specific layer. The available parameters for each layer are: `filter_size`, `num_filters`, `pool`, `norm`, and `activation`. If any of those values is missing from the dictionary, the default one specified as a parameter of the encoder will be used instead. If both `conv_layers` and `num_conv_layers` are `None`, a default list will be assigned to `conv_layers` with the value `[{filter_size: 2}, {filter_size: 3}, {filter_size: 4}, {filter_size: 5}]`. Default value is `None`.
- `num_conv_layers`: An integer parameter that represents the number of parallel convolutional layers. This parameter has effect only if `conv_layers` is `None`. Default value is `None`.
- `filter_size`: An integer parameter that represents the default filter size that will be used for each convolutional layer if a `filter_size` is not already specified in `conv_layers`. It indicates how wide is the 1D convolutional filter. Default value is `3`.
- `num_filters`: An integer parameter that represents the default number of filters that will be used for each convolutional layer if a `num_filters` is not already specified in `conv_layers`. It indicates the number of filters, and by consequence the output channels of the 1D convolution. Default value is `256`.
- `pool_function`: A string parameter that represents the default pooling function that will be used for each convolutional layer if a `pool_function` is not already specified in `conv_layers`. It indicates the type of pooling operation to be performed. Default value is `'max'`.
- `pool_size`: An integer parameter that represents the default pool size that will be used for each convolutional layer if a `pool_size` is not already specified in `conv_layers`. It indicates the size of the max pooling that will be performed along the sequence dimension after the convolution operation. Default value is `None`.
- `fc_layers`: A list of dictionaries that contains the parameters of all the fully connected layers. The length of the list determines the number of stacked fully connected layers, and the content of each dictionary determines the parameters for a specific layer. The available parameters for each layer are: `output_size`, `norm`, and `activation`. If any of those values is missing from the dictionary, the default one specified as a parameter of the encoder will be used instead. If both `fc_layers` and `num_fc_layers` are `None`, a default list will be assigned to `fc_layers` with the value `[{output_size: 512}, {output_size: 256}]` (only applies if `reduce_output` is not `None`). Default value is `None`.
- `num_fc_layers`: An integer parameter that represents the number of stacked fully connected layers. This parameter has effect only if `fc_layers` is `None` and `reduce_output` is not `None`. Default value is `None`.
- `output_size`: An integer parameter that represents the default output size that will be used for each fully connected layer if an `output_size` is not already specified in `fc_layers`. It indicates the size of the output of a fully connected layer. Default value is `256`.
- `use_bias`: A boolean parameter that determines whether to include a bias term in the convolutional and fully connected layers. Default value is `True`.
- `weights_initializer`: A string parameter that represents the default initializer to use for the weights of the convolutional and fully connected layers. If `None`, it uses `'xavier_uniform'`. Other options include: `'constant'`, `'identity'`, `'zeros'`, `'ones'`, `'orthogonal'`, `'normal'`, `'uniform'`, `'truncated_normal'`, `'variance_scaling'`, `'xavier_normal'`, `'xavier_uniform'`, `'xavier_normal'`, `'he_normal'`, `'he_uniform'`, `'lecun_normal'`, `'lecun_uniform'`. Alternatively, it is possible to specify a dictionary with a key `'type'` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. Default value is `'xavier_uniform'`.
- `bias_initializer`: A string parameter that represents the default initializer to use for the biases of the convolutional and fully connected layers. If `None`, it uses `'zeros'`. The available options are the same as for `weights_initializer`. Default value is `'zeros'`.
- `norm`: A string parameter that represents the default normalization method to use for the convolutional and fully connected layers. If a `norm` is not already specified in `conv_layers` or `fc_layers`, this default `norm` will be used for each layer. Default value is `None`.
- `norm_params`: A dictionary parameter that represents the default parameters for the normalization method. This parameter has effect only if `norm` is not `None`. Default value is `None`.
- `activation`: A string parameter that represents the default activation function to use for the convolutional and fully connected layers. Default value is `'relu'`.
- `dropout`: A float parameter that determines if there should be a dropout layer before returning the encoder output. Default value is `0`.
- `reduce_output`: A string parameter that defines how to reduce the output tensor of the convolutional layers along the sequence length dimension if the rank of the tensor is greater than 2. Available values are: `'sum'`, `'mean'` or `'avg'`, `'max'`, `'concat'` (concatenates along the first dimension), `'last'` (returns the last vector of the first dimension), and `None` or `'null'` (which does not reduce and returns the full tensor). Default value is `'max'`.
- `encoder_config`: A dictionary parameter that represents the configuration of the encoder. Default value is `None`.
- `**kwargs`: Additional keyword arguments that can be passed to the method.

The method initializes the class instance by setting the values of the parameters and creating the necessary objects for the encoder. It also logs debug information about the initialization process.

The mathematical operations or procedures performed by the method include:
- Creating an instance of the `EmbedSequence` class if `should_embed` is `True`, which handles the embedding of the input sequence.
- Creating an instance of the `ParallelConv1D` class, which performs parallel 1D convolutions on the input sequence.
- Creating an instance of the `SequenceReducer` class, which reduces the output tensor of the convolutional layers along the sequence length dimension.
- Creating an instance of the `FCStack` class if `reduce_output` is not `None`, which represents a stack of fully connected layers.

The LaTex code for the equations in the markdown document would depend on the specific mathematical operations or procedures performed by the class. Since the code provided does not include the implementation details of the classes used, it is not possible to generate the exact LaTex code for the equations.

### Method **`forward`** Overview
The `forward` method in Python is a method that is used in PyTorch models to define the forward pass of the model. It takes in two parameters: `inputs` and `mask`. 

The `inputs` parameter represents the input sequence that is fed into the encoder. It has a shape of [batch x sequence length] and its type is `torch.int32`. 

The `mask` parameter is an optional input mask that is currently unused and not yet implemented. 

The purpose of the `forward` method is to perform the forward pass of the model, which involves several steps:

1. Embeddings: If the `should_embed` flag is set to `True`, the `embed_sequence` method is called to embed the input sequence. Otherwise, the input sequence is used as is. If the shape of the embedded sequence is not 3-dimensional, it is unsqueezed along the last dimension and converted to `torch.float` dtype.

2. Conv Layers: The `parallel_conv1d` method is called to apply convolutional layers to the embedded sequence. The output of this step is stored in the `hidden` variable.

3. Sequence Reduction: If the `reduce_output` flag is not `None`, the `reduce_sequence` method is called to reduce the sequence dimension of the `hidden` variable.

4. FC Layers: The `fc_stack` method is called to apply fully connected layers to the `hidden` variable.

Finally, the method returns a dictionary with the key "encoder_output" and the value of the `hidden` variable.

Here is the LaTex code for the equations performed in the `forward` method:

1. Embeddings: No mathematical operations are performed.

2. Conv Layers: No mathematical operations are performed.

3. Sequence Reduction: No mathematical operations are performed.

4. FC Layers: No mathematical operations are performed.

### Method **`get_schema_cls`** Overview
The `get_schema_cls` method in Python returns the `ParallelCNNConfig` class. This method does not take any parameters.

The purpose of the `get_schema_cls` method is to provide access to the `ParallelCNNConfig` class, which is likely a configuration class for a parallel convolutional neural network (CNN) model.

As for the mathematical operations or procedures performed by this method, there are none mentioned in the provided code snippet. Therefore, no LaTeX code is required to display any equations in a markdown document.

### Method **`input_shape`** Overview
The `input_shape` method in Python is used to determine the shape of the input data for a neural network model. It is typically used in the context of deep learning frameworks such as PyTorch.

The method does not take any parameters. It is a member function of a class, and the `self` parameter refers to the instance of the class on which the method is called.

The purpose of the `input_shape` method is to return the shape of the input data as a `torch.Size` object. In the given code snippet, the shape is represented as `[self.max_sequence_length]`, where `self.max_sequence_length` is a variable that holds the length of the input sequence.

The mathematical operations or procedures performed by this method are minimal. It simply returns the shape of the input data as a `torch.Size` object, which is a tuple-like object that represents the dimensions of a tensor. No mathematical operations or procedures are involved in this specific method.

To display the equation `[self.max_sequence_length]` in LaTeX code, you can use the following syntax:


$$
\text{{input\_shape}} = \text{{torch.Size}}([self.max\_sequence\_length])
$$

This LaTeX code can be used in a markdown document to represent the equation.

### Method **`output_shape`** Overview
The `output_shape` method in Python returns the output shape of a neural network layer. It is defined within a class and takes no parameters other than the `self` reference. The method returns a `torch.Size` object, which represents the shape of the output tensor.

The purpose of the `output_shape` method is to provide information about the shape of the output tensor produced by a layer in a neural network. This information is useful for understanding the dimensions of the output and for configuring subsequent layers in the network.

The method first checks if the `reduce_output` attribute of the current instance is not `None`. If it is not `None`, it means that the layer is a fully connected layer (`fc_stack`) and the `output_shape` method returns the output shape of the fully connected layer by accessing the `output_shape` attribute of the `fc_stack` object.

If the `reduce_output` attribute is `None`, it means that the layer is a parallel convolutional layer (`parallel_conv1d`). In this case, the `output_shape` method returns the output shape of the parallel convolutional layer by accessing the `output_shape` attribute of the `parallel_conv1d` object.

The mathematical operations or procedures performed by the `output_shape` method are simply accessing the `output_shape` attribute of either the `fc_stack` or `parallel_conv1d` object, depending on the value of the `reduce_output` attribute. No mathematical operations or calculations are performed within the method itself.

Here is the LaTex code to display the equations in a markdown document:


$$
\text{{def output\_shape(self) -> torch.Size:}}
$$

$$
\quad \text{{if self.reduce\_output is not None:}}
$$

$$
\quad \quad \text{{return self.fc\_stack.output\_shape}}
$$

$$
\quad \text{{return self.parallel\_conv1d.output\_shape}}
$$

## Class **`StackedCNN`** Overview
The `StackedCNN` class is a subclass of `SequenceEncoder` and represents a stacked convolutional neural network (CNN) for sequence encoding. It takes a sequence of inputs and applies a series of convolutional layers followed by optional fully connected layers to encode the sequence.

The class has several parameters that can be customized, including options for embedding the input sequence, specifying the vocabulary, setting the representation type (dense or sparse), configuring the convolutional layers, specifying the number of filters and filter size, setting the pooling function and size, configuring the fully connected layers, and more.

The class initializes the encoder by setting the configuration parameters and creating the necessary layers. It also handles the logic for generating default convolutional and fully connected layers if they are not provided by the user.

The `forward` method performs the forward pass of the encoder. It first applies the embedding layer if necessary, then passes the embedded sequence through the convolutional layers. If a sequence reduction method is specified, it applies the reduction layer and then passes the result through the fully connected layers. The output of the encoder is returned as a dictionary with the key "encoder_output".

Overall, the `StackedCNN` class provides a flexible and customizable implementation of a stacked CNN for sequence encoding.

### Method **`__init__`** Overview
The `__init__` method is the constructor of a Python class. It is called when an object of the class is created. In this case, the `__init__` method is defined for a class that represents an encoder.

The purpose of each parameter in the `__init__` method is as follows:

- `should_embed`: If `True`, the input sequence is expected to be made of integers and will be mapped into embeddings.
- `vocab`: Vocabulary of the input feature to encode.
- `representation`: The possible values are `dense` and `sparse`. `dense` means the embeddings are initialized randomly, `sparse` means they are initialized to be one-hot encodings.
- `embedding_size`: The maximum embedding size, the actual size will be `min(vocabulary_size, embedding_size)` for `dense` representations and exactly `vocabulary_size` for the `sparse` encoding, where `vocabulary_size` is the number of different strings appearing in the training set in the column the feature is named after (plus 1 for `<UNK>`).
- `max_sequence_length`: The maximum length of the input sequence.
- `embeddings_trainable`: If `True`, embeddings are trained during the training process. If `False`, embeddings are fixed. It may be useful when loading pretrained embeddings for avoiding fine-tuning them. This parameter has effect only for `representation` is `dense` as `sparse` one-hot encodings are not trainable.
- `pretrained_embeddings`: By default, `dense` embeddings are initialized randomly, but this parameter allows specifying a path to a file containing embeddings in the GloVe format. When the file containing the embeddings is loaded, only the embeddings with labels present in the vocabulary are kept, the others are discarded. If the vocabulary contains strings that have no match in the embeddings file, their embeddings are initialized with the average of all other embeddings plus some random noise to make them different from each other. This parameter has effect only if `representation` is `dense`.
- `embeddings_on_cpu`: By default, embeddings matrices are stored on GPU memory if a GPU is used, as it allows for faster access. But in some cases, the embedding matrix may be really big and this parameter forces the placement of the embedding matrix in regular memory and the CPU is used to resolve them, slightly slowing down the process as a result of data transfer between CPU and GPU memory.
- `conv_layers`: It is a list of dictionaries containing the parameters of all the convolutional layers. The length of the list determines the number of parallel convolutional layers, and the content of each dictionary determines the parameters for a specific layer. The available parameters for each layer are: `filter_size`, `num_filters`, `pool`, `norm`, and `activation`. If any of those values is missing from the dictionary, the default one specified as a parameter of the encoder will be used instead. If both `conv_layers` and `num_conv_layers` are `None`, a default list will be assigned to `conv_layers` with the value `[{filter_size: 2}, {filter_size: 3}, {filter_size: 4}, {filter_size: 5}]`.
- `num_conv_layers`: If `conv_layers` is `None`, this is the number of stacked convolutional layers.
- `num_filters`: If a `num_filters` is not already specified in `conv_layers`, this is the default `num_filters` that will be used for each layer. It indicates the number of filters, and by consequence the output channels of the 1d convolution.
- `filter_size`: If a `filter_size` is not already specified in `conv_layers`, this is the default `filter_size` that will be used for each layer. It indicates how wide is the 1d convolutional filter.
- `strides`: The stride of the convolutional filter.
- `padding`: The padding mode for the convolutional layers.
- `dilation_rate`: The dilation rate for the convolutional layers.
- `pool_function`: The pooling function to use after the convolution operation.
- `pool_size`: If a `pool_size` is not already specified in `conv_layers`, this is the default `pool_size` that will be used for each layer. It indicates the size of the max pooling that will be performed along the sequence dimension after the convolution operation.
- `pool_strides`: The stride of the pooling operation.
- `pool_padding`: The padding mode for the pooling operation.
- `fc_layers`: It is a list of dictionaries containing the parameters of all the fully connected layers. The length of the list determines the number of stacked fully connected layers, and the content of each dictionary determines the parameters for a specific layer. The available parameters for each layer are: `output_size`, `norm`, and `activation`. If any of those values is missing from the dictionary, the default one specified as a parameter of the encoder will be used instead. If both `fc_layers` and `num_fc_layers` are `None`, a default list will be assigned to `fc_layers` with the value `[{output_size: 512}, {output_size: 256}]` (only applies if `reduce_output` is not `None`).
- `num_fc_layers`: If `fc_layers` is `None`, this is the number of stacked fully connected layers (only applies if `reduce_output` is not `None`).
- `output_size`: If an `output_size` is not already specified in `fc_layers`, this is the default `output_size` that will be used for each layer. It indicates the size of the output of a fully connected layer.
- `use_bias`: Whether to use bias in the convolutional and fully connected layers.
- `weights_initializer`: The initializer to use for the weights of the layers.
- `bias_initializer`: The initializer to use for the biases of the layers.
- `norm`: If a `norm` is not already specified in `conv_layers` or `fc_layers`, this is the default `norm` that will be used for each layer. It indicates the norm of the output.
- `norm_params`: Additional parameters for the normalization layers.
- `activation`: Default activation function to use.
- `dropout`: Determines if there should be a dropout layer before returning the encoder output.
- `reduce_output`: Defines how to reduce the output tensor of the convolutional layers along the sequence length dimension if the rank of the tensor is greater than 2. Available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension), and `None` or `null` (which does not reduce and returns the full tensor).
- `encoder_config`: Additional configuration parameters for the encoder.

The `__init__` method performs the following mathematical operations or procedures:

1. Initializes the encoder configuration with the provided `encoder_config`.
2. Sets up the convolutional layers based on the provided `conv_layers` or `num_conv_layers`.
3. Sets up the fully connected layers based on the provided `fc_layers` or `num_fc_layers`.
4. Initializes the embedding sequence if `should_embed` is `True`.
5. Initializes the convolutional stack with the appropriate parameters.
6. Initializes the sequence reducer for reducing the output tensor of the convolutional layers.
7. Initializes the fully connected stack if `reduce_output` is not `None`.

### Method **`get_schema_cls`** Overview
The `get_schema_cls` method in Python returns the `StackedCNNConfig` class. This method does not take any parameters.

The purpose of the `get_schema_cls` method is to provide access to the `StackedCNNConfig` class, which is likely a configuration class for a stacked convolutional neural network (CNN) model.

As for the mathematical operations or procedures performed by this method, there are none mentioned in the provided code snippet. Therefore, no LaTeX code can be generated to display equations in a markdown document.

### Method **`input_shape`** Overview
The `input_shape` method in Python is used to determine the shape of the input data for a neural network model. It is typically used in the context of deep learning frameworks such as PyTorch.

The method does not take any parameters. It is a member function of a class, and the `self` parameter refers to the instance of the class on which the method is called.

The purpose of the `input_shape` method is to return the shape of the input data as a `torch.Size` object. In the given code snippet, the shape is represented as `[self.max_sequence_length]`, where `self.max_sequence_length` is a variable that holds the length of the input sequence.

The mathematical operations or procedures performed by this method are minimal. It simply returns the shape of the input data as a `torch.Size` object, which is a tuple-like object that represents the dimensions of a tensor. The shape is determined by the value of `self.max_sequence_length`, which is a scalar value representing the length of the input sequence.

To represent the equation in LaTeX code, we can write:


$$
\text{{input\_shape}}() \rightarrow \text{{torch.Size}}([\text{{self.max\_sequence\_length}}])
$$

This equation represents the return value of the `input_shape` method, which is a `torch.Size` object with a single dimension corresponding to the length of the input sequence.

### Method **`output_shape`** Overview
The `output_shape` method in Python is used to determine the shape of the output tensor produced by a particular layer or module in a neural network. It returns the size of the output tensor as a `torch.Size` object.

The method does not take any parameters. It accesses the `reduce_output` attribute of the current object. If `reduce_output` is `None`, it returns the `output_shape` attribute of the `conv1d_stack` object. Otherwise, it returns the `output_shape` attribute of the `fc_stack` object.

The purpose of the `output_shape` method is to provide information about the shape of the output tensor produced by a layer or module. This information is useful for understanding the dimensions of the output and for performing subsequent operations or calculations based on the output shape.

The mathematical operations or procedures performed by the `output_shape` method are simple attribute access and conditional statements. It checks the value of the `reduce_output` attribute and returns the `output_shape` attribute of the corresponding object. There are no mathematical equations involved in this method.

LaTeX code for displaying the equations in a markdown document is not applicable in this case, as there are no mathematical equations involved in the `output_shape` method.

### Method **`forward`** Overview
The `forward` method in Python is a method defined within a class. It takes two parameters: `inputs` and `mask`. 

The purpose of the `forward` method is to perform a series of mathematical operations or procedures on the input sequence `inputs` and return the result. 

Here is a breakdown of the method and its parameters:

- `inputs`: The input sequence fed into the encoder. It has a shape of [batch x sequence length] and is of type `torch.int32`.

- `mask`: Input mask (unused, not yet implemented). This parameter is optional and can be omitted. It is used to mask certain elements of the input sequence, but in this implementation, it is not yet implemented.

The method starts by checking if the `should_embed` flag is set. If it is, the input sequence is passed through an embedding layer using the `embed_sequence` method. If not, the input sequence is used as is. If the shape of the embedded sequence is not 3-dimensional, it is unsqueezed to make it 3-dimensional.

Next, the embedded sequence is assigned to the `hidden` variable.

Then, the `hidden` variable is passed through a stack of 1D convolutional layers using the `conv1d_stack` method.

If the `reduce_output` flag is not None, the `hidden` variable is further reduced using the `reduce_sequence` method.

Finally, if there are any fully connected layers specified in the `fc_stack` method, the `hidden` variable is passed through them.

The method returns a dictionary with the key "encoder_output" and the value of the `hidden` variable.

The mathematical operations or procedures performed in this method include embedding the input sequence, applying 1D convolutions, reducing the sequence if specified, and passing the result through fully connected layers. Unfortunately, the specific mathematical equations or formulas used in these operations are not provided in the code snippet.

## Class **`StackedParallelCNN`** Overview
The `StackedParallelCNN` class is a subclass of `SequenceEncoder` and represents a stacked parallel convolutional neural network (CNN) for sequence encoding. It takes a sequence of inputs and applies a series of convolutional layers in parallel, followed by optional fully connected layers.

The class has several parameters that can be customized, including whether to embed the input sequence, the vocabulary for the input feature, the representation type (dense or sparse), the size of the embeddings, whether the embeddings are trainable, and whether to use pretrained embeddings. Other parameters control the number and size of the convolutional layers, the pooling function, the number and size of the fully connected layers, the output size, the activation function, and the dropout rate.

The class initializes the encoder by setting the configuration parameters and creating the necessary layers. It also defines the input and output shapes of the encoder. The `forward` method performs the forward pass of the encoder, applying the embedding layer if necessary, followed by the convolutional layers, sequence reduction, and fully connected layers. The output of the encoder is a dictionary containing the encoder output.

### Method **`__init__`** Overview
The `__init__` method is the constructor of a Python class. It is called when an object of the class is created. In this case, the method is used to initialize the parameters and components of an encoder.

The purpose of each parameter in the `__init__` method is as follows:

- `should_embed`: If True, the input sequence is expected to be made of integers and will be mapped into embeddings.
- `vocab`: Vocabulary of the input feature to encode.
- `representation`: The possible values are `dense` and `sparse`. `dense` means the embeddings are initialized randomly, `sparse` means they are initialized to be one-hot encodings.
- `embedding_size`: The maximum embedding size. The actual size will be `min(vocabulary_size, embedding_size)` for `dense` representations and exactly `vocabulary_size` for the `sparse` encoding.
- `max_sequence_length`: The maximum length of the input sequence.
- `embeddings_trainable`: If True, embeddings are trained during the training process. If False, embeddings are fixed.
- `pretrained_embeddings`: Path to a file containing embeddings in the GloVe format.
- `embeddings_on_cpu`: If True, the embedding matrix is placed in regular memory and the CPU is used to resolve them.
- `stacked_layers`: A list of lists of dictionaries containing the parameters of the stack of parallel convolutional layers.
- `num_stacked_layers`: The number of elements in the stack of parallel convolutional layers.
- `filter_size`: The default filter size that will be used for each layer.
- `num_filters`: The default number of filters that will be used for each layer.
- `pool_function`: The default pooling function that will be used for each layer.
- `pool_size`: The default pool size that will be used for each layer.
- `fc_layers`: A list of dictionaries containing the parameters of all the fully connected layers.
- `num_fc_layers`: The number of stacked fully connected layers.
- `output_size`: The default output size that will be used for each layer.
- `use_bias`: If True, bias terms are used in the layers.
- `weights_initializer`: The initializer to use for the weights.
- `bias_initializer`: The initializer to use for the biases.
- `norm`: The default normalization method to use.
- `norm_params`: Parameters for the normalization method.
- `activation`: The default activation function to use.
- `dropout`: If True, a dropout layer is added before returning the encoder output.
- `reduce_output`: Defines how to reduce the output tensor of the convolutional layers along the sequence length dimension.

The mathematical operations or procedures performed in the `__init__` method include:

- Initializing the parameters and components of the encoder.
- Checking the validity of the layer parametrization.
- Creating an instance of the `EmbedSequence` class if `should_embed` is True.
- Creating an instance of the `ParallelConv1DStack` class.
- Creating an instance of the `SequenceReducer` class if `reduce_output` is not None.
- Creating an instance of the `FCStack` class if `reduce_output` is not None.

### Method **`get_schema_cls`** Overview
The `get_schema_cls` method in Python returns the `StackedParallelCNNConfig` class. This method does not take any parameters.

The purpose of the `get_schema_cls` method is to provide access to the `StackedParallelCNNConfig` class. This class is likely used to configure and define the schema for a stacked parallel convolutional neural network (CNN).

Unfortunately, without further information or access to the `StackedParallelCNNConfig` class, it is not possible to document the specific mathematical operations or procedures performed by this method. The mathematical operations or procedures would depend on the implementation of the `StackedParallelCNNConfig` class and its associated methods.

As for generating LaTeX code to display equations in a markdown document, it would require specific equations or mathematical operations to be provided. Without knowledge of the specific equations involved, it is not possible to generate LaTeX code for them.

### Method **`input_shape`** Overview
The `input_shape` method in Python is used to determine the shape of the input data for a neural network model. It is typically used in the context of deep learning frameworks such as PyTorch.

The method does not take any parameters. It is a member function of a class, and the `self` parameter refers to the instance of the class on which the method is called.

The purpose of the `input_shape` method is to return the shape of the input data as a `torch.Size` object. In the given code snippet, the shape is represented as `[self.max_sequence_length]`, where `self.max_sequence_length` is a variable that holds the length of the input sequence.

The mathematical operations or procedures performed by this method are minimal. It simply returns the shape of the input data as a `torch.Size` object, which is a tuple-like object that represents the dimensions of a tensor. No mathematical operations or procedures are involved in this specific method.

To display the equation `[self.max_sequence_length]` in LaTeX code, you can use the following syntax in a markdown document:

```

$$ [self.max\_sequence\_length] $$
```

This will render the equation as:


$$ [self.max\_sequence\_length] $$

### Method **`output_shape`** Overview
The `output_shape` method in Python returns the output shape of a neural network layer. It is defined within a class and takes no parameters other than the `self` reference. The method returns a `torch.Size` object.

The purpose of the `output_shape` method is to determine the shape of the output tensor produced by a layer in a neural network. It does this by checking if the `reduce_output` attribute is not `None`. If it is not `None`, it returns the output shape of the fully connected stack (`fc_stack.output_shape`). Otherwise, it returns the output shape of the parallel convolutional stack (`parallel_conv1d_stack.output_shape`).

The mathematical operations or procedures performed by this method are simply accessing the `output_shape` attribute of either the fully connected stack or the parallel convolutional stack, depending on the value of the `reduce_output` attribute. There are no mathematical operations or equations involved in this method.

### Method **`forward`** Overview
The `forward` method in Python is a method defined within a class. It takes two parameters: `inputs` and `mask`. 

The purpose of the `forward` method is to perform a series of mathematical operations or procedures on the input sequence `inputs` and return the result. 

Here is a breakdown of the method and its parameters:

- `inputs`: The input sequence fed into the encoder. It is a tensor of shape [batch x sequence length] and type torch.int32.

- `mask`: Input mask (unused, not yet implemented). This parameter is optional and can be omitted. It is a tensor of the same shape as `inputs`.

The method starts by checking if the `should_embed` flag is set. If it is, the input sequence is passed through an embedding layer using the `embed_sequence` method. If not, the input sequence is used as is. If the shape of the embedded sequence is not 3-dimensional, it is unsqueezed along the last dimension until it becomes 3-dimensional.

Next, the embedded sequence is passed through a parallel convolutional stack using the `parallel_conv1d_stack` method.

If the `reduce_output` flag is not None, the output sequence is further reduced using the `reduce_sequence` method.

Finally, the reduced sequence is passed through a fully connected (FC) layer stack using the `fc_stack` method.

The output of the method is a dictionary with a single key-value pair, where the key is "encoder_output" and the value is the final hidden state of the encoder.

The mathematical operations or procedures performed by the `forward` method can be summarized as follows:

1. Embedding the input sequence if `should_embed` is True.
2. Passing the embedded sequence through a parallel convolutional stack.
3. Reducing the output sequence if `reduce_output` is not None.
4. Passing the reduced sequence through a fully connected layer stack.

Here is the LaTex code for the equations in the markdown document:


$$
\text{{embedded\_sequence}} = \text{{embed\_sequence}}(\text{{inputs}}, \text{{mask}})
$$


$$
\text{{hidden}} = \text{{embedded\_sequence}}
$$


$$
\text{{hidden}} = \text{{parallel\_conv1d\_stack}}(\text{{hidden}}, \text{{mask}})
$$


$$
\text{{hidden}} = \text{{reduce\_sequence}}(\text{{hidden}})
$$


$$
\text{{hidden}} = \text{{fc\_stack}}(\text{{hidden}}, \text{{mask}})
$$


$$
\text{{return}} \{"encoder\_output": \text{{hidden}}\}
$$

## Class **`StackedRNN`** Overview
The `StackedRNN` class is a subclass of `SequenceEncoder` and is used for encoding sequential data using stacked recurrent neural networks (RNNs). 

The class has several parameters that can be set during initialization, including:
- `should_embed`: a boolean indicating whether the input sequence should be embedded into dense vectors.
- `vocab`: a list representing the vocabulary of the input feature to encode.
- `representation`: a string indicating the type of representation to use for the embeddings, either "dense" or "sparse".
- `embedding_size`: an integer representing the size of the embeddings.
- `embeddings_trainable`: a boolean indicating whether the embeddings should be trainable during the training process.
- `pretrained_embeddings`: a filepath to a file containing pre-trained embeddings in the GloVe format.
- `embeddings_on_cpu`: a boolean indicating whether the embeddings should be stored in regular memory and processed on the CPU.
- `num_layers`: an integer representing the number of stacked recurrent layers.
- `max_sequence_length`: an integer representing the maximum length of the input sequence.
- `state_size`: an integer representing the size of the hidden state of the RNN.
- `cell_type`: a string indicating the type of recurrent cell to use, either "rnn", "lstm", or "gru".
- `bidirectional`: a boolean indicating whether to use bidirectional RNNs.
- `activation`: a string indicating the activation function to use.
- `recurrent_activation`: a string indicating the activation function to use for the recurrent connections.
- `unit_forget_bias`: a boolean indicating whether to use a bias term in the forget gate of LSTM cells.
- `recurrent_initializer`: a string indicating the type of initializer to use for the recurrent weights.
- `dropout`: a float representing the dropout rate for the input sequence.
- `recurrent_dropout`: a float representing the dropout rate for the recurrent connections.
- `fc_layers`: a list of dictionaries representing the parameters for fully connected layers.
- `num_fc_layers`: an integer representing the number of fully connected layers.
- `output_size`: an integer representing the size of the output of the fully connected layers.
- `use_bias`: a boolean indicating whether to use bias terms in the fully connected layers.
- `weights_initializer`: a string indicating the type of initializer to use for the weights of the fully connected layers.
- `bias_initializer`: a string indicating the type of initializer to use for the biases of the fully connected layers.
- `norm`: a string indicating the type of normalization to use in the fully connected layers.
- `norm_params`: a dictionary containing the parameters for the normalization.
- `fc_activation`: a string indicating the activation function to use in the fully connected layers.
- `fc_dropout`: a float representing the dropout rate for the fully connected layers.
- `reduce_output`: a string indicating how to reduce the output tensor of the convolutional layers if its rank is greater than 2.
- `encoder_config`: a dictionary containing the configuration of the encoder.

The class has a `forward` method that takes an input tensor and an optional mask tensor as input and returns a dictionary containing the encoder output and the final state of the RNN. The input tensor should have shape `[batch x sequence length]` and type `torch.int32`. The mask tensor is currently unused and not yet implemented.

The class also has properties for the input shape and output shape of the encoder, as well as methods for getting the schema class and the input data type.

### Method **`__init__`** Overview
The `__init__` method is the constructor of a Python class. It is called when an object of the class is created. In this case, the `__init__` method is defined for a class that represents an encoder.

The purpose of each parameter in the `__init__` method is as follows:

- `should_embed`: If `True`, the input sequence is expected to be made of integers and will be mapped into embeddings.
- `vocab`: Vocabulary of the input feature to encode.
- `representation`: The possible values are `dense` and `sparse`. `dense` means the embeddings are initialized randomly, `sparse` means they are initialized to be one-hot encodings.
- `embedding_size`: The maximum embedding size, the actual size will be `min(vocabulary_size, embedding_size)` for `dense` representations and exactly `vocabulary_size` for the `sparse` encoding, where `vocabulary_size` is the number of different strings appearing in the training set in the column the feature is named after (plus 1 for `<UNK>`).
- `embeddings_trainable`: If `True`, embeddings are trained during the training process. If `False`, embeddings are fixed. It may be useful when loading pretrained embeddings for avoiding fine-tuning them. This parameter has effect only for `representation` is `dense` as `sparse` one-hot encodings are not trainable.
- `pretrained_embeddings`: By default, `dense` embeddings are initialized randomly, but this parameter allows specifying a path to a file containing embeddings in the GloVe format. When the file containing the embeddings is loaded, only the embeddings with labels present in the vocabulary are kept, the others are discarded. If the vocabulary contains strings that have no match in the embeddings file, their embeddings are initialized with the average of all other embeddings plus some random noise to make them different from each other. This parameter has effect only if `representation` is `dense`.
- `embeddings_on_cpu`: By default, embeddings matrices are stored on GPU memory if a GPU is used, as it allows for faster access. But in some cases, the embedding matrix may be really big and this parameter forces the placement of the embedding matrix in regular memory and the CPU is used to resolve them, slightly slowing down the process as a result of data transfer between CPU and GPU memory.
- `num_layers`: The number of stacked recurrent layers.
- `max_sequence_length`: The maximum length of the input sequence. If `None`, the length is not limited.
- `state_size`: The size of the state of the recurrent neural network (RNN).
- `cell_type`: The type of recurrent cell to use. Available values are: `rnn`, `lstm`, `gru`.
- `bidirectional`: If `True`, two recurrent networks will perform encoding in the forward and backward direction and their outputs will be concatenated.
- `activation`: The activation function to use in the RNN.
- `recurrent_activation`: The activation function to use for the recurrent step in the RNN.
- `unit_forget_bias`: If `True`, the forget gate bias of the LSTM cell is set to 1.0, otherwise it is set to 0.0.
- `recurrent_initializer`: The initializer to use for the recurrent weights matrix.
- `dropout`: Dropout rate for the input sequence.
- `recurrent_dropout`: Dropout rate for the recurrent stack.
- `fc_layers`: A list of dictionaries containing the parameters of all the fully connected (FC) layers.
- `num_fc_layers`: The number of stacked FC layers.
- `output_size`: The size of the output of the FC layers.
- `use_bias`: If `True`, the FC layers will have bias terms.
- `weights_initializer`: The initializer to use for the weights of the FC layers.
- `bias_initializer`: The initializer to use for the bias terms of the FC layers.
- `norm`: The normalization layer to use after each FC layer.
- `norm_params`: The parameters for the normalization layer.
- `fc_activation`: The activation function to use in the FC layers.
- `fc_dropout`: Dropout rate for the FC layers.
- `reduce_output`: Defines how to reduce the output tensor of the convolutional layers along the sequence length dimension if the rank of the tensor is greater than 2. Available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension), and `None` or `null` (which does not reduce and returns the full tensor).
- `encoder_config`: Additional configuration parameters for the encoder.

The mathematical operations or procedures performed in the `__init__` method include:

- Initializing the parent class of the encoder.
- Setting the configuration parameters.
- Initializing the maximum sequence length, hidden size, and embedding size.
- Checking if embedding is required and initializing the `EmbedSequence` object if necessary.
- Initializing the `RecurrentStack` object.
- Initializing the `SequenceReducer` object if the output needs to be reduced.
- Initializing the `FCStack` object if the output needs to be further processed with FC layers.

### Method **`get_schema_cls`** Overview
The `get_schema_cls` method in Python returns the `StackedRNNConfig` class. This method does not take any parameters.

The purpose of the `get_schema_cls` method is to provide access to the `StackedRNNConfig` class, which is likely a configuration class for a stacked recurrent neural network (RNN) model.

As for the mathematical operations or procedures performed by this method, there are none mentioned in the provided code snippet. Therefore, no LaTex code can be generated to display equations in a markdown document.

### Method **`input_shape`** Overview
The `input_shape` method in Python is used to determine the shape of the input data for a neural network model. It is typically used in the context of deep learning frameworks such as PyTorch.

The method does not take any parameters. It is a member function of a class, and the `self` parameter refers to the instance of the class on which the method is called.

The purpose of the `input_shape` method is to return the shape of the input data as a `torch.Size` object. In the given code snippet, the shape is represented as `[self.max_sequence_length]`, where `self.max_sequence_length` is a variable that holds the length of the input sequence.

The mathematical operations or procedures performed by this method are minimal. It simply returns the shape of the input data as a `torch.Size` object, which is a tuple-like structure that represents the dimensions of a tensor. The shape is determined solely based on the value of `self.max_sequence_length`.

To represent the equation in LaTeX code, we can write:


$$
\text{{input\_shape}}() \rightarrow \text{{torch.Size}}([\text{{self.max\_sequence\_length}}])
$$

This equation shows that the `input_shape` method returns a `torch.Size` object with a single dimension, where the size of that dimension is equal to the value of `self.max_sequence_length`.

### Method **`output_shape`** Overview
The `output_shape` method in Python returns the output shape of a neural network layer. It is defined within a class and takes no parameters other than the `self` reference. The method returns a `torch.Size` object, which represents the shape of the output tensor.

The purpose of the `output_shape` method is to provide information about the shape of the output tensor produced by a layer in a neural network. This can be useful for understanding the dimensions of the output and for configuring subsequent layers in the network.

The method first checks if the `reduce_output` attribute of the current instance is not `None`. If it is not `None`, it means that the layer is a fully connected (dense) layer, and the method returns the output shape of the fully connected stack (`fc_stack.output_shape`).

If the `reduce_output` attribute is `None`, it means that the layer is a recurrent layer, and the method returns the output shape of the recurrent stack (`recurrent_stack.output_shape`).

The `output_shape` method does not perform any mathematical operations or procedures. It simply returns the output shape of the layer based on the type of layer it is and the corresponding attributes of the layer instance.

Here is the LaTex code for the equations:


$$
\text{{output\_shape}}(self) \rightarrow \text{{torch.Size}}:
$$


$$
\text{{if }} \text{{self.reduce\_output}} \text{{ is not None:}}
$$


$$
\quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \

### Method **`input_dtype`** Overview
The `input_dtype` method in Python is used to specify the data type of the input tensor in a PyTorch model. It is a method of a class and is defined as follows:

```python
def input_dtype(self):
    return torch.int32
```

This method does not take any parameters. It simply returns the data type `torch.int32`, which is a 32-bit integer data type in PyTorch.

The purpose of this method is to provide information about the expected data type of the input tensor. This information is useful for various operations and computations performed on the tensor within the model.

As for the mathematical operations or procedures performed by this method, there are none. It only returns the specified data type without performing any calculations.

In LaTeX, the equation for the `input_dtype` method can be represented as:


$$
\text{{input\_dtype}}() = \text{{torch.int32}}
$$

### Method **`forward`** Overview
The `forward` method is a method in a Python class that takes in two parameters: `inputs` and `mask`. 

The purpose of the `inputs` parameter is to represent the input sequence that is fed into the encoder. It is a tensor of shape `[batch x sequence length]` and has a data type of `torch.int32`.

The purpose of the `mask` parameter is to represent an input mask, although it is currently unused and not yet implemented. It is an optional parameter, meaning it can be omitted when calling the method.

The method begins by checking if the `should_embed` flag is set to `True`. If it is, the `inputs` sequence is passed through the `embed_sequence` method, which embeds the sequence into a higher-dimensional space. If `should_embed` is `False`, the `inputs` sequence is used as is. If the `inputs` sequence has less than 3 dimensions, it is unsqueezed to have a shape of `[..., sequence_length, 1]`.

The embedded sequence is then assigned to the `hidden` variable.

Next, the `hidden` sequence is passed through a recurrent stack, represented by the `recurrent_stack` method. This stack applies recurrent layers to the `hidden` sequence, potentially capturing temporal dependencies in the data. The `mask` parameter is also passed to the `recurrent_stack` method, although its purpose is not specified in the code snippet.

After the recurrent layers, the `hidden` sequence is optionally reduced using the `reduce_output` parameter. If `reduce_output` is not `None`, the `hidden` sequence is passed through the `reduce_sequence` method, which reduces the sequence to a lower-dimensional representation.

Finally, if there are any fully connected (FC) layers specified in the `fc_stack` attribute, the `hidden` sequence is passed through them. The `mask` parameter is also passed to the `fc_stack` method, although its purpose is not specified in the code snippet.

The method returns a dictionary with two keys: "encoder_output" and "encoder_output_state". The value associated with the "encoder_output" key is the `hidden` sequence, potentially after being reduced and passed through FC layers. The value associated with the "encoder_output_state" key is the final state of the recurrent layers.

The mathematical operations or procedures performed in this method include embedding the input sequence, applying recurrent layers, reducing the sequence (if specified), and passing the sequence through FC layers. However, the specific mathematical equations or operations involved in these procedures are not provided in the code snippet.

## Class **`StackedCNNRNN`** Overview
The `StackedCNNRNN` class is a subclass of `SequenceEncoder` and represents a stacked convolutional neural network (CNN) followed by a recurrent neural network (RNN) encoder. 

The class has several parameters that can be set during initialization, including options for embedding the input sequence, specifying the vocabulary, setting the representation type (dense or sparse), configuring the CNN layers, specifying the RNN parameters, and defining the fully connected (FC) layers.

The `StackedCNNRNN` class implements the `forward` method, which takes an input sequence tensor and an optional mask tensor. The input sequence is first embedded if necessary, then passed through the CNN layers, followed by the RNN layers. The output of the RNN layers can be further reduced using a sequence reduction method, and then passed through the FC layers if specified. The final output is returned as a dictionary containing the encoder output and the final state of the RNN.

The class also provides methods for getting the input and output shapes of the encoder, as well as a method for getting the schema class associated with the encoder.

Overall, the `StackedCNNRNN` class provides a flexible and configurable implementation of a stacked CNN-RNN encoder for sequence data.

### Method **`__init__`** Overview
The `__init__` method is the constructor of a Python class. It is called when an object of the class is created. In this case, the `__init__` method is defined with multiple parameters and is used to initialize the attributes of the class.

Here is a description of each parameter and its purpose:

- `should_embed`: If True, the input sequence is expected to be made of integers and will be mapped into embeddings.
- `vocab`: Vocabulary of the input feature to encode.
- `max_sequence_length`: The maximum length of the input sequence.
- `representation`: The representation type of the embeddings. It can be either "dense" or "sparse".
- `embedding_size`: The size of the embeddings.
- `embeddings_trainable`: If True, embeddings are trained during the training process. If False, embeddings are fixed.
- `pretrained_embeddings`: Path to a file containing embeddings in the GloVe format.
- `embeddings_on_cpu`: If True, the embedding matrix is stored in regular memory and the CPU is used to resolve them.
- `conv_layers`: Custom-defined convolutional layers.
- `num_conv_layers`: The number of convolutional layers to generate with default parameters.
- `num_filters`: The number of filters in each convolutional layer.
- `filter_size`: The size of the filters in each convolutional layer.
- `strides`: The stride size in each convolutional layer.
- `padding`: The padding type in each convolutional layer.
- `dilation_rate`: The dilation rate in each convolutional layer.
- `conv_activation`: The activation function in each convolutional layer.
- `conv_dropout`: Dropout rate for the convolutional layers.
- `pool_function`: The pooling function to use.
- `pool_size`: The size of the pooling window.
- `pool_strides`: The stride size in the pooling layer.
- `pool_padding`: The padding type in the pooling layer.
- `num_rec_layers`: The number of stacked recurrent layers.
- `state_size`: The size of the state of the recurrent layers.
- `cell_type`: The type of recurrent cell to use (rnn, lstm, gru).
- `bidirectional`: If True, two recurrent networks will perform encoding in the forward and backward direction and their outputs will be concatenated.
- `activation`: The activation function in the recurrent layers.
- `recurrent_activation`: The activation function for the recurrent connections in the recurrent layers.
- `unit_forget_bias`: If True, the forget gate bias is initialized to 1.
- `recurrent_initializer`: The initializer to use for the recurrent weights.
- `dropout`: Dropout rate for the output of the recurrent layers.
- `recurrent_dropout`: Dropout rate for the recurrent connections in the recurrent layers.
- `fc_layers`: Custom-defined fully connected layers.
- `num_fc_layers`: The number of fully connected layers to generate with default parameters.
- `output_size`: The size of the output of the fully connected layers.
- `use_bias`: If True, bias terms are added to the fully connected layers.
- `weights_initializer`: The initializer to use for the weights of the fully connected layers.
- `bias_initializer`: The initializer to use for the bias terms of the fully connected layers.
- `norm`: The normalization layer to use.
- `norm_params`: Parameters for the normalization layer.
- `fc_activation`: The activation function in the fully connected layers.
- `fc_dropout`: Dropout rate for the fully connected layers.
- `reduce_output`: Defines how to reduce the output tensor of the convolutional layers along the sequence length dimension if the rank of the tensor is greater than 2.

The `__init__` method performs the following mathematical operations or procedures:

1. Initializes the attributes of the class based on the provided parameters.
2. Creates an instance of the `EmbedSequence` class if `should_embed` is True.
3. Creates an instance of the `Conv1DStack` class to stack multiple 1D convolutional layers.
4. Creates an instance of the `RecurrentStack` class to stack multiple recurrent layers.
5. Creates an instance of the `SequenceReducer` class to reduce the output sequence of the recurrent layers.
6. Creates an instance of the `FCStack` class to stack multiple fully connected layers.

The LaTex code for the equations in the markdown document would depend on the specific equations used in the mathematical operations or procedures of the `__init__` method.

### Method **`get_schema_cls`** Overview
The `get_schema_cls` method is a Python function that returns an instance of the `StackedCNNRNNConfig` class. This method does not take any parameters.

The purpose of the `get_schema_cls` method is to provide a way to obtain an instance of the `StackedCNNRNNConfig` class. This class is likely used for configuring a stacked convolutional neural network with recurrent layers.

Since the method does not perform any mathematical operations or procedures, there is no need to generate LaTeX code for equations.

### Method **`input_shape`** Overview
The `input_shape` method in Python is used to determine the shape of the input data for a neural network model. It is typically used in the context of deep learning frameworks such as PyTorch.

The method does not take any parameters. It is a member function of a class, and the `self` parameter refers to the instance of the class on which the method is called.

The purpose of the `input_shape` method is to return the shape of the input data as a `torch.Size` object. In the given code snippet, the shape is represented as `[self.max_sequence_length]`, where `self.max_sequence_length` is a variable that holds the length of the input sequence.

The mathematical operations or procedures performed by this method are minimal. It simply returns the shape of the input data as a `torch.Size` object, which is a tuple-like structure that represents the dimensions of a tensor. No mathematical operations or equations are involved in this particular method.

To display the equation `[self.max_sequence_length]` in LaTeX format, you can use the following code in a markdown document:

```

$$
\text{{input\_shape}} = [self.max\_sequence\_length]
$$
```

This will render the equation as:


$$ \text{input\_shape} = [self.max\_sequence\_length] $$

### Method **`output_shape`** Overview
The `output_shape` method in Python returns the output shape of a neural network layer. It is defined within a class and takes no parameters other than the `self` reference. The method returns a `torch.Size` object, which represents the shape of the output tensor.

The purpose of the `output_shape` method is to provide information about the shape of the output tensor produced by a layer in a neural network. This can be useful for understanding the dimensions of the output and for configuring subsequent layers in the network.

The method first checks if the `reduce_output` attribute of the current layer is not `None`. If it is not `None`, it means that the layer is a fully connected (dense) layer, and the method returns the output shape of the fully connected layer, which is obtained from the `output_shape` attribute of the `fc_stack` object.

If the `reduce_output` attribute is `None`, it means that the layer is a recurrent layer, and the method returns the output shape of the recurrent layer, which is obtained from the `output_shape` attribute of the `recurrent_stack` object.

In mathematical terms, the `output_shape` method does not perform any explicit mathematical operations. It simply returns the output shape of the layer, which is a representation of the dimensions of the output tensor. Therefore, there is no need to generate LaTeX code for mathematical equations in this case.

### Method **`forward`** Overview
The `forward` method is a method in a Python class that performs various mathematical operations on input sequences. Here is a breakdown of the purpose of each parameter and the mathematical operations performed:

- `inputs`: The input sequence fed into the encoder. It is a tensor of shape [batch x sequence length] and type torch.int32.

- `mask` (optional): Input mask. This parameter is currently unused and not yet implemented.

The method performs the following mathematical operations:

1. Embeddings: If the `should_embed` flag is True, the input sequence is embedded using the `embed_sequence` method. Otherwise, the input sequence is used as is. If the shape of the embedded sequence is not 3-dimensional, it is unsqueezed along the last dimension until it becomes 3-dimensional.

2. Convolutional Layers: The embedded sequence is passed through a stack of 1D convolutional layers using the `conv1d_stack` method.

3. Recurrent Layers: The output from the convolutional layers is passed through a stack of recurrent layers using the `recurrent_stack` method. The output hidden state and final state of the recurrent layers are returned.

4. Sequence Reduction: If the `reduce_output` flag is not None, the hidden sequence is reduced using the `reduce_sequence` method.

5. Fully Connected Layers: The reduced hidden sequence is passed through a stack of fully connected layers using the `fc_stack` method.

Finally, the method returns a dictionary containing the encoder output and the encoder output state. The encoder output has the shape [batch_size, seq_size, output_size] if reduction is applied, otherwise it has the shape [batch_size, seq_size, state_size]. The final state has the shape [batch_size, state_size] for RNN/GRU or ([batch_size, state_size], [batch_size, state_size]) for LSTM.

## Class **`StackedTransformer`** Overview
The `StackedTransformer` class is a subclass of `SequenceEncoder` and is used for encoding sequences of data using a stacked transformer architecture. 

The class has several parameters that can be set during initialization, including `max_sequence_length` (the maximum length of the input sequence), `should_embed` (whether the input sequence should be embedded), `vocab` (the vocabulary of the input feature to encode), `representation` (the type of representation to use for the embeddings), `embedding_size` (the size of the embeddings), `embeddings_trainable` (whether the embeddings should be trainable), `pretrained_embeddings` (a path to a file containing pretrained embeddings), `embeddings_on_cpu` (whether to store the embeddings on the CPU), `num_layers` (the number of transformer layers to stack), `hidden_size` (the size of the hidden state in the transformer), `num_heads` (the number of attention heads in the transformer), `transformer_output_size` (the size of the output of the transformer), `dropout` (the dropout rate), `fc_layers` (a list of dictionaries specifying the parameters for fully connected layers), `num_fc_layers` (the number of fully connected layers), `output_size` (the size of the output), `use_bias` (whether to use bias in the fully connected layers), `weights_initializer` (the initializer for the weights), `bias_initializer` (the initializer for the biases), `norm` (the normalization layer to use), `norm_params` (the parameters for the normalization layer), `fc_activation` (the activation function for the fully connected layers), `fc_dropout` (the dropout rate for the fully connected layers), `reduce_output` (how to reduce the output tensor of the convolutional layers), and `encoder_config` (a dictionary containing the configuration of the encoder).

The class has a `forward` method that takes an input tensor and an optional mask tensor as input and returns the encoded output. The input tensor should have shape `[batch x sequence length]` and type `torch.int32`. The output tensor has shape `[batch x sequence length x hidden]` if `reduce_output` is `None`, or shape `[batch x output_size]` if `reduce_output` is not `None`. The method first embeds the input sequence if `should_embed` is `True`, then applies the transformer layers, reduces the sequence if `reduce_output` is not `None`, and applies the fully connected layers if `reduce_output` is not `None`. The method returns a dictionary with the key "encoder_output" and the encoded output as the value.

The class also has properties `input_shape` and `output_shape` that return the shape of the input and output tensors, respectively. The class also has a static method `get_schema_cls` that returns the schema class for the encoder.

### Method **`__init__`** Overview
The `__init__` method is the constructor of a Python class. It is called when an object of the class is created. In this case, the `__init__` method is defined for a class that represents an encoder.

The purpose of each parameter in the `__init__` method is as follows:

- `max_sequence_length`: The maximum length of the input sequence.
- `should_embed`: If `True`, the input sequence is expected to be made of integers and will be mapped into embeddings.
- `vocab`: Vocabulary of the input feature to encode.
- `representation`: The representation of the embeddings. The possible values are `dense` and `sparse`. `dense` means the embeddings are initialized randomly, while `sparse` means they are initialized to be one-hot encodings.
- `embedding_size`: The maximum embedding size. The actual size will be `min(vocabulary_size, embedding_size)` for `dense` representations and exactly `vocabulary_size` for the `sparse` encoding, where `vocabulary_size` is the number of different strings appearing in the training set in the column the feature is named after (plus 1 for `<UNK>`).
- `embeddings_trainable`: If `True`, embeddings are trained during the training process. If `False`, embeddings are fixed. This parameter has effect only for `representation` is `dense` as `sparse` one-hot encodings are not trainable.
- `pretrained_embeddings`: Path to a file containing embeddings in the GloVe format. When the file containing the embeddings is loaded, only the embeddings with labels present in the vocabulary are kept, and the others are discarded. If the vocabulary contains strings that have no match in the embeddings file, their embeddings are initialized with the average of all other embeddings plus some random noise to make them different from each other. This parameter has effect only if `representation` is `dense`.
- `embeddings_on_cpu`: If `True`, the embedding matrix is placed in regular memory and the CPU is used to resolve them, slightly slowing down the process as a result of data transfer between CPU and GPU memory.
- `num_layers`: The number of stacked transformer layers.
- `hidden_size`: The size of the hidden state of the transformer.
- `num_heads`: The number of attention heads in the transformer.
- `transformer_output_size`: The output size of the transformer.
- `dropout`: The dropout rate to apply in the transformer.
- `fc_layers`: A list of dictionaries containing the parameters of all the fully connected (FC) layers. The length of the list determines the number of FC layers, and the content of each dictionary determines the parameters for a specific layer. The available parameters for each layer are: `output_size`, `use_bias`, `weights_initializer`, `bias_initializer`, `norm`, `norm_params`, `activation`, and `dropout`. If any of those values is missing from the dictionary, the default one specified as a parameter of the encoder will be used instead.
- `num_fc_layers`: The number of stacked FC layers.
- `output_size`: The output size of the FC layers.
- `use_bias`: If `True`, the FC layers will use bias.
- `weights_initializer`: The initializer to use for the FC layers. If `None`, it uses `xavier_uniform`. Other options are available, such as `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `xavier_normal`, `xavier_uniform`, `xavier_normal`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively, it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters.
- `bias_initializer`: The initializer to use for the bias of the FC layers. It follows the same options as `weights_initializer`.
- `norm`: The normalization layer to apply after each FC layer. It can be `None` or a string indicating the type of normalization layer to use, such as `batch_norm`, `layer_norm`, or `instance_norm`.
- `norm_params`: Parameters for the normalization layer. It can be `None` or a dictionary with the parameters for the chosen normalization layer.
- `fc_activation`: The activation function to use in the FC layers.
- `fc_dropout`: The dropout rate to apply in the FC layers.
- `reduce_output`: Defines how to reduce the output tensor of the convolutional layers along the `s` sequence length dimension if the rank of the tensor is greater than 2. The available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension), and `None` or `null` (which does not reduce and returns the full tensor).
- `encoder_config`: Additional configuration parameters for the encoder.

The mathematical operations or procedures performed in the `__init__` method are as follows:

- The method initializes the parent class using `super().__init__()`.
- It sets the `config` attribute of the class to the value of `encoder_config`.
- It sets the `max_sequence_length` attribute of the class to the value of `max_sequence_length`.
- It sets the `should_embed` attribute of the class to the value of `should_embed`.
- It sets the `should_project` attribute of the class to `False`.
- If `should_embed` is `True`, it creates an instance of the `TokenAndPositionEmbedding` class and assigns it to the `embed_sequence` attribute of the class. This class is responsible for mapping the input sequence into embeddings.
- If `should_embed` is `True` and the actual embedding size is different from `hidden_size`, it creates an instance of the `nn.Linear` class and assigns it to the `project_to_hidden_size` attribute of the class. This linear layer is used to project the embeddings to the hidden size.
- If `should_embed` is `False`, it creates an instance of the `nn.Linear` class and assigns it to the `project_to_hidden_size` attribute of the class. This linear layer is used to project the input sequence to the hidden size.
- It creates an instance of the `TransformerStack` class and assigns it to the `transformer_stack` attribute of the class. This class represents a stack of transformer layers.
- It sets the `reduce_output` attribute of the class to the value of `reduce_output`.
- It creates an instance of the `SequenceReducer` class and assigns it to the `reduce_sequence` attribute of the class. This class is responsible for reducing the output tensor of the convolutional layers along the `s` sequence length dimension.
- If `reduce_output` is `None`, it sets the `supports_masking` attribute of the class to `True`.
- If `reduce_output` is not `None`, it creates an instance of the `FCStack` class and assigns it to the `fc_stack` attribute of the class. This class represents a stack of fully connected layers.

### Method **`get_schema_cls`** Overview
The `get_schema_cls` method is a Python function that returns the `StackedTransformerConfig` class. This method does not take any parameters.

The purpose of the `get_schema_cls` method is to provide access to the `StackedTransformerConfig` class, which is likely a configuration class used for defining the schema or structure of a stacked transformer.

As for the mathematical operations or procedures performed by this method, there are none mentioned in the provided code snippet. Therefore, there is no need to generate LaTeX code for equations in this case.

### Method **`input_shape`** Overview
The `input_shape` method in Python is used to determine the shape of the input data for a neural network model. It is typically used in the context of deep learning frameworks such as PyTorch.

The method does not take any parameters. It is a member function of a class, and the `self` parameter refers to the instance of the class on which the method is called.

The purpose of the `input_shape` method is to return the shape of the input data as a `torch.Size` object. In the given code snippet, the shape is represented as `[self.max_sequence_length]`, where `self.max_sequence_length` is a variable that holds the length of the input sequence.

The mathematical operations or procedures performed by this method are minimal. It simply returns the shape of the input data as a `torch.Size` object, which is a tuple-like object that represents the dimensions of a tensor. The shape is determined by the value of `self.max_sequence_length`, which is a scalar value representing the length of the input sequence.

To represent the equation in LaTeX code, we can write:


$$
\text{{input\_shape}}() \rightarrow \text{{torch.Size}}([\text{{self.max\_sequence\_length}}])
$$

This equation represents the return value of the `input_shape` method, which is a `torch.Size` object with a single dimension corresponding to the length of the input sequence.

### Method **`output_shape`** Overview
The `output_shape` method in Python returns the output shape of a neural network model. It is defined within a class and takes no parameters other than the `self` reference.

The purpose of this method is to determine the shape of the output produced by the neural network model. It does this by checking if the `reduce_output` attribute is not `None`. If it is not `None`, it returns the output shape of the fully connected (fc) stack, which is obtained by calling the `output_shape` method of the `fc_stack` object.

If the `reduce_output` attribute is `None`, it means that the model does not have a fully connected stack. In this case, the method returns the output shape of the transformer stack, which is obtained by calling the `output_shape` method of the `transformer_stack` object.

The `output_shape` method returns the output shape as a `torch.Size` object.

Mathematical operations or procedures are not performed within this method. It simply retrieves the output shape from either the fc stack or the transformer stack based on the value of the `reduce_output` attribute.

LaTeX code for displaying the equations in a markdown document:


$$
\text{{output\_shape}}(self) \rightarrow \text{{torch.Size}}
$$


$$
\text{{if }} \text{{self.reduce\_output}} \text{{ is not None:}}
$$


$$
\quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \

### Method **`forward`** Overview
The `forward` method in Python is a method that is used in PyTorch to define the forward pass of a neural network model. It takes two parameters: `inputs` and `mask`. 

The `inputs` parameter represents the input sequence that is fed into the encoder. It has a shape of [batch x sequence length] and a data type of torch.int32. 

The `mask` parameter is an optional input mask that is currently unused and not yet implemented. 

The purpose of the `forward` method is to perform the forward pass of the neural network model. It consists of several steps:

1. Embeddings: If the `should_embed` flag is set to True, the `embed_sequence` method is called to embed the input sequence. Otherwise, the input sequence is used as is. If the shape of the embedded sequence is not 3-dimensional, it is unsqueezed to make it 3-dimensional.

2. Transformer Layers: The embedded sequence is passed through the `transformer_stack` method, which applies a stack of transformer layers to the sequence. This helps to capture complex patterns and dependencies in the input sequence.

3. Sequence Reduction: If the `reduce_output` flag is not None, the `reduce_sequence` method is called to reduce the sequence to a single vector representation. This can be useful for tasks such as sequence classification or sequence generation.

4. FC Layers: The reduced sequence is passed through a stack of fully connected (FC) layers, which perform linear transformations on the input data. This can help to further extract features and make predictions based on the reduced sequence representation.

Finally, the method returns a dictionary with the key "encoder_output" and the value of the hidden representation after the forward pass.

Here is the LaTex code for the equations performed in the `forward` method:

1. Embeddings:
   - If `should_embed` is True: $embedded\_sequence = embed\_sequence(inputs, mask)$
   - If `should_embed` is False: $embedded\_sequence = inputs$

2. Transformer Layers:
   - $hidden = transformer\_stack(hidden, mask)$

3. Sequence Reduction:
   - If `reduce_output` is not None: $hidden = reduce\_sequence(hidden)$

4. FC Layers:
   - $hidden = fc\_stack(hidden, mask)$

The final output is a dictionary: $output = \{"encoder\_output": hidden\}$

