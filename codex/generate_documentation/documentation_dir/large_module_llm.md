# Module:`llm.py` Overview

This Python code defines a class called `LLM` which is a subclass of `BaseModel`. The `LLM` class represents a large language model and is used for training and generating text. 

The code imports various libraries and modules, including `contextlib`, `logging`, `os`, `tempfile`, `typing`, `numpy`, `torch`, and `transformers`. It also imports classes and functions from the `ludwig` package, which is a deep learning toolbox built on top of TensorFlow and PyTorch.

The `LLM` class has several methods and attributes. Some of the important methods include:
- `__init__()`: Initializes the `LLM` object and loads the large language model.
- `forward()`: Performs a forward pass through the model and produces logits tensor for fine-tuning.
- `generate()`: Generates text using the model.
- `train_loss()`: Computes the training loss for the model.
- `eval_loss()`: Computes the evaluation loss for the model.
- `outputs_to_predictions()`: Converts the model's outputs to predictions for each output feature.
- `save()`: Saves the model to a given path.
- `load()`: Loads the model from a given path.

The code also includes helper functions and classes, such as `DictWrapper`, `realign_target_and_prediction_tensors()`, and `TextOutputFeature`.

Overall, the `LLM` class provides a high-level interface for training and using a large language model for text generation tasks.

## Class **`DictWrapper`** Overview
The `DictWrapper` class is a wrapper for a `LudwigFeatureDict` module that allows for iteration over keys. It is designed to avoid exposing input and output features as modules of the Ludwig Language Model (LLM) in order to simplify training and avoid confusion with systems like DeepSpeed.

The class has the following methods:

- `__init__(self, obj: LudwigFeatureDict)`: Initializes the `DictWrapper` object with a `LudwigFeatureDict` object.
- `get(self, key) -> torch.nn.Module`: Returns the module associated with the given key.
- `set(self, key: str, module: torch.nn.Module) -> None`: Sets the module associated with the given key.
- `__len__(self) -> int`: Returns the number of keys in the `DictWrapper` object.
- `__next__(self) -> None`: Returns the next key in the iteration.
- `__iter__(self) -> None`: Returns an iterator over the keys.
- `keys(self) -> List[str]`: Returns a list of all the keys.
- `values(self) -> List[torch.nn.Module]`: Returns a list of all the modules.
- `items(self) -> List[Tuple[str, torch.nn.Module]]`: Returns a list of key-module pairs.
- `update(self, modules: Dict[str, torch.nn.Module]) -> None`: Updates the `DictWrapper` object with the given modules.

### Method **`__init__`** Overview
The `__init__` method in Python is a special method that is automatically called when an object is created from a class. It is used to initialize the attributes of the object.

In the given code snippet, the `__init__` method takes in a parameter `obj` of type `LudwigFeatureDict`. This parameter is used to initialize the `self.obj` attribute of the object.

The purpose of the `__init__` method is to set up the initial state of the object. It is commonly used to assign values to the object's attributes or perform any necessary setup operations.

As for the mathematical operations or procedures performed in the `__init__` method, there are none specified in the given code snippet. The method simply assigns the value of the `obj` parameter to the `self.obj` attribute. Therefore, there are no mathematical equations or procedures to document in this case.

Here is the code snippet with the `__init__` method:

```python
def __init__(self, obj: LudwigFeatureDict):
    self.obj = obj
```

In LaTeX, the code snippet can be displayed as:

```latex
\begin{verbatim}
def __init__(self, obj: LudwigFeatureDict):
    self.obj = obj
\end{verbatim}
```

Note: The LaTeX code above assumes that you have the necessary packages and configurations in your LaTeX document to display code snippets.

### Method **`get`** Overview
The `get` method in Python is used to retrieve a value associated with a specified key from a dictionary-like object. In this case, the method is defined within a class and takes two parameters: `self` and `key`. 

- `self`: It is a reference to the current instance of the class. It is used to access the attributes and methods of the class.
- `key`: It is the key whose associated value needs to be retrieved from the dictionary-like object.

The method returns the value associated with the specified key by calling the `get` method of the `obj` attribute of the current instance.

Regarding the mathematical operations or procedures performed by this method, there are none explicitly mentioned in the provided code snippet. The method simply retrieves a value from a dictionary-like object and returns it. Therefore, there is no need to generate LaTeX code for mathematical equations in this case.

### Method **`set`** Overview
The `set` method in Python is used to assign a value to a specified key in a dictionary. In the given code snippet, the `set` method is defined as a member function of a class. It takes two parameters: `key` and `module`.

- `key` (type: str): This parameter represents the key to be assigned in the dictionary.
- `module` (type: torch.nn.Module): This parameter represents the value to be assigned to the specified key.

The purpose of this method is to set a key-value pair in the dictionary `self.obj`. The `self.obj` is an instance variable of the class that represents a dictionary.

The mathematical operations or procedures performed by this method are not explicitly mentioned in the code snippet. It seems to be a generic method for setting a key-value pair in a dictionary, and it does not involve any specific mathematical operations. Therefore, there is no need to generate LaTeX code for mathematical equations in this case.

### Method **`__len__`** Overview
The `__len__` method in Python is a special method that is used to define the behavior of the built-in `len()` function when called on an object of a class. This method should be defined within a class and it should return the length of the object.

Here is the syntax for defining the `__len__` method:

```python
def __len__(self) -> int:
    # code to calculate and return the length of the object
```

Parameters:
- `self`: It is a reference to the current instance of the class. It is used to access the attributes and methods of the class.

Return value:
- `int`: The `__len__` method should return an integer value representing the length of the object.

Mathematical operations or procedures:
The `__len__` method does not perform any mathematical operations or procedures. It simply returns the length of the object by calling the built-in `len()` function on the `self.obj` attribute.

LaTeX code for displaying the equation in a markdown document:
The `__len__` method does not involve any mathematical equations, so there is no need for LaTeX code in this case.

### Method **`__next__`** Overview
The `__next__` method is a special method in Python that is used to define the behavior of an iterator object. It is called when the `next()` function is called on the iterator object. 

The `__next__` method takes only one parameter, which is `self`. The `self` parameter refers to the instance of the iterator object itself.

In the given code snippet, the `__next__` method returns the next element in the iterator object `self.obj`. The `iter()` function is used to create an iterator object from `self.obj`, and the `next()` function is used to retrieve the next element from the iterator.

The purpose of the `__next__` method is to provide a way to iterate over the elements of an object in a sequential manner. It allows you to define the logic for retrieving the next element from the iterator object.

As for the mathematical operations or procedures performed in the `__next__` method, there are none in the given code snippet. The method simply returns the next element from the iterator object without performing any mathematical operations. Therefore, there is no need to generate LaTeX code for mathematical equations in this case.

### Method **`__iter__`** Overview
The `__iter__` method in Python is a special method that allows an object to be iterable. It is typically used in conjunction with the `iter()` function or a `for` loop to iterate over the elements of an object.

The `__iter__` method takes only one parameter, which is `self`. The `self` parameter refers to the instance of the object that the method is being called on. It is used to access the attributes and methods of the object within the method.

In the given code snippet, the `__iter__` method returns an iterator object created from the keys of the `self.obj` dictionary. The `iter()` function is used to create the iterator object, which allows the keys of the dictionary to be iterated over.

The purpose of this method is to provide a way to iterate over the keys of the `self.obj` dictionary. By returning an iterator object, it allows the keys to be accessed one by one in a loop or using the `next()` function.

As for the mathematical operations or procedures performed in this method, there are none. The method simply returns an iterator object for the keys of the dictionary. Therefore, no LaTeX code is required to display any equations in a markdown document.

### Method **`keys`** Overview
The `keys` method in Python is used to retrieve all the keys from a dictionary object. It does not take any parameters other than the `self` parameter, which refers to the instance of the object on which the method is called.

The method returns a list of strings, where each string represents a key from the dictionary object.

Here is the LaTex code to display the equations in a markdown document:


$$
\text{{keys}}(\text{{self}}) \rightarrow \text{{List[str]}}
$$

The method does not perform any mathematical operations or procedures. It simply returns the keys of the dictionary object.

### Method **`values`** Overview
The `values` method in Python is used to retrieve all the values from a dictionary-like object. In this case, it is used to retrieve the values from the `self.obj` object, which is expected to be a dictionary-like object.

The method does not take any parameters, as indicated by the absence of any parameters within the parentheses.

The return type of the method is specified as `List[torch.nn.Module]`, which means it returns a list of objects that are instances of the `torch.nn.Module` class.

The method simply calls the `values()` function on the `self.obj` object and returns the result. The `values()` function returns a view object that contains all the values from the dictionary-like object.

As for the mathematical operations or procedures performed by this method, there are none. The method is solely focused on retrieving the values from the dictionary-like object and returning them as a list. Therefore, there is no need for any LaTex code to display equations in a markdown document.

### Method **`items`** Overview
The `items` method in Python is used to return a list of tuples containing the key-value pairs of a dictionary. In the provided code snippet, the `items` method is defined within a class and returns a list of tuples where each tuple consists of a string and a `torch.nn.Module` object.

Parameters:
- `self`: It represents the instance of the class.

Return:
- `List[Tuple[str, torch.nn.Module]]`: It returns a list of tuples where each tuple contains a string and a `torch.nn.Module` object.

Mathematical Operations or Procedures:
The `items` method does not perform any mathematical operations or procedures. It simply returns the key-value pairs of a dictionary-like object. Therefore, there is no need for generating LaTeX code for mathematical equations in this case.

### Method **`update`** Overview
The `update` method in Python is used to update the modules in a dictionary. It takes in two parameters:

1. `self`: It is a reference to the current instance of the class. It is used to access the attributes and methods of the class.

2. `modules`: It is a dictionary that contains the modules to be updated. The keys of the dictionary are strings representing the names of the modules, and the values are instances of the `torch.nn.Module` class.

The purpose of the `update` method is to update the modules in the dictionary `self.obj` with the modules provided in the `modules` parameter. The method performs the following mathematical operations or procedures:

1. It updates the modules in the dictionary `self.obj` by replacing the existing modules with the new modules provided in the `modules` parameter.

The `update` method does not perform any specific mathematical operations or procedures that can be represented using LaTeX code.

## Class **`LLM`** Overview
The Python class `LLM` is a high-level language model class that extends the `BaseModel` class. It is used for training and generating text using large language models. 

The class has the following methods and attributes:

- `type()`: A static method that returns the type of the model as a string.
- `__init__()`: The constructor method that initializes the `LLM` object. It takes a `config_obj` parameter of type `LLMModelConfig`, a `random_seed` parameter of type `int`, a `device` parameter of type `str`, and additional keyword arguments. It calls the constructor of the `BaseModel` class and initializes various attributes of the `LLM` object.
- `create_feature_dict()`: A method that creates and returns a `LudwigFeatureDict` object.
- `output_feature_decoder`: A property that returns the output feature decoder of the model.
- `initialize_adapter()`: A method that initializes the adapter for fine-tuning the model.
- `to_device()`: A method that moves the model to the specified device.
- `build_outputs()`: A class method that builds and returns the output feature of the model.
- `forward()`: A method that performs a forward pass of the model and returns the output tensors.
- `generate()`: A method that generates text using the model.
- `update_metrics()`: A method that updates the model's metrics given targets and predictions.
- `train_loss()`: A method that computes the training loss for the model.
- `eval_loss()`: A method that computes the evaluation loss for the model.
- `outputs_to_predictions()`: A method that converts the model's outputs to predictions for each output feature.
- `save()`: A method that saves the model to a specified path.
- `load()`: A method that loads the model from a specified path.
- `get_args()`: A method that returns the initialization arguments for constructing the model.
- `_generate_merged_ids()`: A private method that merges the input_ids and target_ids together to create a unified tensor to pass into the model.
- `_add_left_padding()`: A private method that adds left padding to the input_ids tensor.
- `_create_attention_mask()`: A private method that creates an attention mask for the input_ids tensor.
- `get_augmentation_pipelines()`: A method that returns the augmentation pipeline for the model.

The `LLM` class is used for training and generating text using large language models. It provides methods for forward pass, generation, loss computation, metric updates, saving and loading the model, and more.

### Method **`type`** Overview
The Python method `type()` is a built-in function that returns the type of an object. It takes one parameter, which is the object whose type needs to be determined. The purpose of this method is to provide information about the type of an object.

The mathematical operations or procedures performed by the `type()` method are not applicable as it is not specifically designed for mathematical calculations. Instead, it is used to determine the type of an object, such as whether it is a string, integer, list, etc.

Here is an example of how to use the `type()` method:

```python
x = 5
print(type(x))  # Output: <class 'int'>

y = "Hello"
print(type(y))  # Output: <class 'str'>

z = [1, 2, 3]
print(type(z))  # Output: <class 'list'>
```

In the above example, the `type()` method is used to determine the type of the variables `x`, `y`, and `z`. The output shows the type of each variable, such as `int`, `str`, and `list`.

Please note that the `type()` method should not be confused with the `type` keyword, which is used to define custom classes in Python.

### Method **`__init__`** Overview
The `__init__` method is the constructor method of a Python class. It is called when an object of the class is created. In this case, the `__init__` method is defined with the following parameters:

- `self`: It is a reference to the current instance of the class.
- `config_obj`: It is an object of the `LLMModelConfig` class, which represents the configuration of the large language model.
- `random_seed`: It is an optional parameter that represents the random seed used for reproducibility.
- `device`: It is an optional parameter that represents the device on which the model will be loaded.
- `**_kwargs`: It is used to accept any additional keyword arguments.

The purpose of the `__init__` method is to initialize the attributes of the class and perform some mathematical operations or procedures. Here are the mathematical operations or procedures performed in the `__init__` method:

1. Call the `__init__` method of the parent class (`super().__init__(random_seed=random_seed)`).
2. Assign the `config_obj` parameter to the `config_obj` attribute of the class (`self.config_obj = config_obj`).
3. Assign the `random_seed` parameter to the `_random_seed` attribute of the class (`self._random_seed = random_seed`).
4. Assign the `model_name` attribute of the `config_obj` to the `model_name` attribute of the class (`self.model_name = self.config_obj.model_name`).
5. Load the large language model using the `AutoModelForCausalLM.from_pretrained` method and assign it to the `model` attribute of the class (`self.model = AutoModelForCausalLM.from_pretrained(self.config_obj.model_name)`).
6. Set the `curr_device` attribute of the class to `torch.device("cpu")` as the model is initially loaded onto the CPU (`self.curr_device = torch.device("cpu")`).
7. Determine the maximum length of the context (input + output tokens) based on the model's configuration. If the model has a `max_sequence_length` attribute, assign its value to the `context_len` attribute. If not, check if the model has a `max_position_embeddings` attribute and assign its value to the `context_len` attribute. If neither attribute is present, assign a default value of 2048 to the `context_len` attribute.
8. Assign the `max_new_tokens` attribute of the `config_obj.generation` to the `max_new_tokens` attribute of the class (`self.max_new_tokens = self.config_obj.generation.max_new_tokens`).
9. Calculate the `max_input_length` attribute by subtracting the `max_new_tokens` attribute, 8, and the `context_len` attribute from each other (`self.max_input_length = self.context_len - self.max_new_tokens - 8`).
10. Initialize the tokenizer using the `AutoTokenizer.from_pretrained` method and assign it to the `tokenizer` attribute of the class. The `use_fast` parameter is set to `True` by default, but if the model's configuration is an instance of `LlamaConfig`, the `use_fast` parameter is set to `False` to disable the Llama fast tokenizer (`self.tokenizer = AutoTokenizer.from_pretrained(self.config_obj.model_name, use_fast=use_fast)`).
11. Set the pad token for the tokenizer using the `set_pad_token` function (`set_pad_token(self.tokenizer)`).
12. Create a `GenerationConfig` object from the `config_obj.generation` and assign it to the `generation` attribute of the class (`self.generation = GenerationConfig(**self.config_obj.generation.to_dict())`).
13. Update the `input_features` attribute of the class by calling the `build_inputs` method with the `input_feature_configs` parameter set to `self.config_obj.input_features`. If a `KeyError` is raised during this process, it is caught and re-raised with a custom error message (`self.input_features.update(self.build_inputs(input_feature_configs=self.config_obj.input_features))`).
14. Assign the `type` attribute of the first `output_feature` in `config_obj.output_features` to the `output_feature_type` attribute of the class (`self.output_feature_type = self.config_obj.output_features[0].type`).
15. Update the `output_features` attribute of the class by calling the `build_outputs` method with the `output_feature_configs` parameter set to `self.config_obj.output_features` and the `input_size` parameter set to the model's vocabulary size if the `output_feature_type` is `TEXT`, otherwise set it to the tokenizer's vocabulary size (`self.output_features.update(self.build_outputs(output_feature_configs=self.config_obj.output_features, input_size=self.input_shape[-1] if self.output_feature_type == TEXT else self.model.config.vocab_size))`).
16. Extract the decoder object for the forward pass by creating a `ModuleWrapper` object with the first item of the `output_features` attribute and assign it to the `_output_feature_decoder` attribute of the class (`self._output_feature_decoder = ModuleWrapper(self.output_features.items()[0][1])`).
17. Initialize the PEFT adapter if one is provided by calling the `initialize_adapter` method (`self.initialize_adapter()`).
18. Clear the data cache using the `clear_data_cache` function (`clear_data_cache()`).

Note: The mathematical operations or procedures described above do not involve any explicit mathematical equations.

### Method **`create_feature_dict`** Overview
The `create_feature_dict` method is a Python method that returns a `LudwigFeatureDict` object wrapped in a `DictWrapper`. The purpose of this method is to create and initialize a feature dictionary.

Parameters:
- `self`: This parameter refers to the instance of the class that the method belongs to.

Mathematical operations or procedures:
This method does not perform any mathematical operations or procedures. It simply creates an empty `LudwigFeatureDict` object and wraps it in a `DictWrapper` before returning it.

### Method **`output_feature_decoder`** Overview
The `output_feature_decoder` method in Python returns the `OutputFeature` module associated with the current instance of the class. The purpose of this method is to provide access to the output feature decoder module, which is used for decoding the output features of a model.

Parameters:
- `self`: The current instance of the class.

Mathematical operations or procedures:
This method does not perform any mathematical operations or procedures. It simply returns the `OutputFeature` module associated with the current instance.

LaTeX code for equations:
There are no equations involved in this method.

### Method **`initialize_adapter`** Overview
The `initialize_adapter` method is used to initialize an adapter for fine-tuning a model. It takes no parameters other than `self`, which refers to the current instance of the class.

The purpose of this method is to check if an adapter configuration is provided and, if so, wrap the model with a PEFT (Plug-and-Play Encoder-Finetuner) model for fine-tuning. The PEFT model is obtained using the `get_peft_model` function from the `peft` module.

If an adapter configuration is provided, the method creates a PEFT configuration object (`peft_config`) based on the adapter configuration and the model's tokenizer. The model is then replaced with the PEFT model obtained by passing the original model and the PEFT configuration to the `get_peft_model` function.

After initializing the adapter, the method logs some information about the trainable parameters for fine-tuning. It prints the type of adapter being used (`self.config_obj.adapter.type`) and calls the `print_trainable_parameters` method of the model to display a summary of the trainable parameters.

No mathematical operations or procedures are performed in this method, so there is no need for LaTex code to display equations.

### Method **`to_device`** Overview
The `to_device` method in Python is used to move the model to a specified device. It takes in a `device` parameter, which specifies the target device.

Here is a breakdown of the purpose of each parameter and the mathematical operations or procedures performed:

- `self`: This parameter refers to the instance of the class that the method is being called on.

- `device`: This parameter specifies the target device to which the model should be moved.

The method performs the following mathematical operations or procedures:

1. It converts the `device` parameter to a `torch.device` object.

2. It checks if the specified `device` is the same as the current device of the model. If they are the same, it returns the current instance of the class. Otherwise, it logs a message indicating that the model is being moved from the current device to the specified device.

3. It creates an empty dictionary `model_kwargs` to store additional model keyword arguments.

4. It checks if the specified `device` is a CUDA device (`torch.device("cuda")`) and if there are multiple GPUs available (`num_gpus > 1`).

5. If the above conditions are met, it updates the `model_kwargs` dictionary with specific parameters for multi-GPU training. These parameters include `low_cpu_mem_usage`, `torch_dtype`, `device_map`, and `max_memory`.

6. It saves the current model's weights to a temporary directory.

7. If the model has an adapter, it loads the `PeftModel` from the `peft` library and initializes it with the specified model name and `model_kwargs`.

8. If the model does not have an adapter, it loads the `AutoModelForCausalLM` from the temporary directory and initializes it with the specified `model_kwargs`.

9. If the specified `device` is not a CUDA device or there is only one GPU available, it moves the model to the specified `device` using the `to` method.

10. It updates the `curr_device` attribute of the class instance with the specified `device`.

11. It returns the updated instance of the class.

The mathematical operations or procedures in this method do not involve any explicit mathematical calculations.

### Method **`build_outputs`** Overview
The `build_outputs` method is a class method that builds and returns an output feature. It takes three parameters:

1. `cls`: The class itself.
2. `output_feature_configs`: A collection of output feature configurations. It is of type `FeatureCollection[BaseOutputFeatureConfig]`.
3. `input_size`: An integer representing the size of the input.

The purpose of the `build_outputs` method is to build and return an output feature based on the given configuration. It performs the following mathematical operations or procedures:

1. Check if the length of `output_feature_configs` is greater than 1. If it is, raise a `ValueError` with the message "Only single task currently supported".
2. Get the first element of `output_feature_configs` and assign it to `output_feature_config`.
3. Set the `input_size` of `output_feature_config` to the given `input_size`.
4. Create an empty dictionary called `output_features`.
5. Call the class method `build_single_output` with `output_feature_config` and `output_features` as parameters, and assign the returned output feature to `output_feature`.
6. Add `output_feature` to `output_features` with the key `output_feature_config.name`.
7. Return `output_features`.

The mathematical operations or procedures performed by the `build_outputs` method do not involve any specific mathematical calculations.

### Method **`forward`** Overview
The `forward` method is a function defined within a Python class. It takes in three parameters: `self`, `inputs`, and `mask`. 

The `self` parameter refers to the instance of the class that the method is being called on. It is used to access the attributes and methods of the class.

The `inputs` parameter is of type `Union`, which means it can accept different types of inputs. It can be a dictionary of input names to input tensors, a dictionary of input names to numpy arrays, or a tuple of dictionaries where the first dictionary contains input names to input tensors and the second dictionary contains target names to target tensors. This parameter represents the inputs to the model.

The `mask` parameter is an optional argument that represents a mask for the inputs. It is used to mask certain parts of the inputs during computation.

The purpose of the `forward` method is to perform a forward pass through the model and produce logits tensor for fine-tuning the model. It takes the inputs, unpacks them, generates merged input_id and target_id pairs for the model, and creates corresponding attention masks. It then performs a forward pass using the model and obtains the model outputs.

If the output feature type is not text, the method passes the generated tokens through the decoder after averaging the token probabilities. This step is required for the classification head for the classifier decoder.

If the output feature type is text, the decoder outputs are set to be the model outputs. Otherwise, the decoder outputs are obtained from the output feature decoder.

The output feature tensor is then set to the decoder outputs (logits) using the `set_output_feature_tensor` function.

Next, the method gets predictions, probabilities, and logits tensor from the output feature's predictions function.

Finally, the method casts the prediction tensors to float32 for metric computation in case reduced precision is being used.

The method returns the outputs, which is a dictionary of output {feature name}::{tensor_name} to output tensor.

### Method **`generate`** Overview
The `generate` method is a function defined within a Python class. It takes in several parameters and returns a dictionary of torch tensors. Here is a breakdown of the method and its parameters:

Parameters:
- `self`: The first parameter `self` refers to the instance of the class that the method belongs to. It is used to access the attributes and methods of the class.
- `inputs`: This parameter is of type `Union`, which means it can accept different types of inputs. It can be a dictionary of string keys and torch tensors, a dictionary of string keys and numpy arrays, or a tuple of two dictionaries of string keys and torch tensors. This parameter represents the input data for the model.
- `mask`: This parameter is optional and has a default value of `None`. It represents a mask that can be applied to the input data.

Mathematical Operations/Procedures:
1. The method first unpacks the `inputs` parameter using the `_unpack_inputs` method, which is not shown in the code snippet provided. The purpose of this step is to extract the input_ids from the `inputs` parameter.
2. The method then iterates over each `input_ids_sample` in the `input_ids` list.
3. For each `input_ids_sample`, the method removes the left padding using the `remove_left_padding` function and the `self.tokenizer`.
4. If the shape of the `input_ids_sample_no_padding` tensor is greater than `self.max_input_length`, a warning message is logged and the tensor is truncated to the maximum input length.
5. The length of the `input_ids_sample_no_padding` tensor is appended to the `input_lengths` list.
6. The method then checks if the CUDA device is available and if the model is on the CUDA device. If both conditions are met, it wraps the generation process with the `torch.backends.cuda.sdp_kernel` context manager, enabling flash attention backend for faster generation. Otherwise, it uses a `nullcontext` to do nothing.
7. Within the context manager, the method generates text using the model by calling the `generate` method of the model. It passes the `input_ids_sample_no_padding`, `mask`, `generation_config`, `return_dict_in_generate`, and `output_scores` parameters to the `generate` method.
8. The generated sequences are appended to the `sequences_list`.
9. After iterating over all `input_ids_sample`, the method calls the `forward` method of the `decoder_obj` attribute of the `output_feature_decoder` object. It passes the `sequences_list` and `input_lengths` as parameters to the `forward` method.
10. The output of the `forward` method is stored in the `outputs` variable.
11. Finally, the `outputs` variable is returned.

Note: The code snippet provided does not include the definitions of the `_unpack_inputs`, `remove_left_padding`, and other referenced functions or classes. Therefore, the complete understanding of the method may require additional context.

### Method **`_unpack_inputs`** Overview
The `_unpack_inputs` method in Python is used to convert input tensors to input ids. It takes in a parameter called `inputs`, which can be a dictionary of tensors, a dictionary of numpy arrays, or a tuple of dictionaries of tensors. The purpose of each parameter is as follows:

- `inputs`: This parameter represents the input tensors or arrays. It can be a dictionary of tensors, a dictionary of numpy arrays, or a tuple of dictionaries of tensors.

The method performs the following mathematical operations or procedures:

1. If the `inputs` parameter is a tuple, it means that both inputs and targets are provided. In this case, the method separates the inputs and targets from the tuple.

2. If the targets are provided, the method converts them to tensors. It iterates over each target feature name and target value in the targets dictionary. If the target value is not already a tensor, it converts it to a tensor using `torch.from_numpy()`. Otherwise, it keeps the target value as it is.

3. The method checks if the keys of the inputs dictionary match the keys of the input features dictionary. It uses the `assert` statement to raise an error if they don't match.

4. The method calls the `get_input_ids` method to convert the input tensors to input ids.

5. If targets are provided, the method calls the `get_target_ids` method to convert the target tensors to target ids. Otherwise, it sets the target ids to None.

6. The method returns the input ids and target ids as a tuple.

Here is the LaTex code for the equations in a markdown document:


$$
\text{{input\_ids}}, \text{{target\_ids}} = \text{{\_unpack\_inputs}}(\text{{inputs}})
$$

### Method **`get_input_ids`** Overview
The `get_input_ids` method in Python is used to retrieve the input ids for the text feature input. It takes in a single parameter `inputs`, which is a dictionary or tuple containing the input data. 

The `inputs` parameter can be of three types:
- A dictionary with string keys and values as either torch tensors or numpy arrays.
- A tuple containing two dictionaries, where the first dictionary contains string keys and values as torch tensors, and the second dictionary contains string keys and values as torch tensors.
 
The method returns a torch tensor that represents the input ids for the text feature input. 

The mathematical operations or procedures performed by this method are as follows:
- The method retrieves the input feature name from the `config_obj` object, which is assumed to be a list of input features.
- It then retrieves the corresponding input data from the `inputs` parameter using the input feature name.
- Finally, it converts the retrieved input data to a torch tensor of type `torch.int32` and returns it.

Here is the LaTex code to display the equations in a markdown document:


$$
\text{{input\_ids}} = \text{{inputs}}[\text{{config\_obj.input\_features[0].name}}].\text{{type}}(\text{{torch.int32}})
$$

### Method **`get_target_ids`** Overview
The `get_target_ids` method in Python is used to retrieve the output ids for the text feature output. It takes in two parameters:

1. `self`: It represents the instance of the class that the method belongs to. It is used to access the attributes and methods of the class.

2. `outputs`: It is a dictionary that contains the output tensors. The keys of the dictionary represent the names of the output features, and the values are the corresponding output tensors.

The method returns a tensor that represents the output ids for the text feature output. The specific output tensor is accessed using the `output_features` attribute of the `config_obj` attribute of the class instance. The `output_features` attribute is a list of output features, and the first feature is accessed using indexing (`[0]`). The name of the first output feature is used as the key to retrieve the corresponding output tensor from the `outputs` dictionary.

The returned tensor is then cast to the `torch.int32` data type using the `type` method of the tensor.

There are no mathematical operations or procedures performed in this method. It simply retrieves the output ids for the text feature output based on the provided parameters.

### Method **`update_metrics`** Overview
The `update_metrics` method in Python is used to update the model's metrics based on the given targets and predictions. Here is a breakdown of the method and its parameters:

```python
def update_metrics(self, targets, predictions):
```

- `self`: The first parameter `self` refers to the instance of the class that the method belongs to.
- `targets`: This parameter represents the target values used for evaluation.
- `predictions`: This parameter represents the predicted values generated by the model.

The method performs the following operations:

1. It iterates over each output feature in the `output_features` dictionary of the model.
2. If the output feature is of type `TextOutputFeature`, it aligns the target length with the predictions length using the `realign_target_and_prediction_tensors` function. This is done to enable text metric evaluation.
3. It then calls the `update_metrics` method of the output feature object, passing the aligned targets and predictions.
4. If the output feature is not of type `TextOutputFeature`, it directly calls the `update_metrics` method of the output feature object, passing the targets and predictions.
5. It calculates the evaluation loss and additional losses by calling the `eval_loss` method of the model, passing the targets and predictions.
6. It updates the `eval_loss_metric` with the evaluation loss.
7. It updates the `eval_additional_losses_metrics` with the additional losses.

Here is the LaTeX code for the equations mentioned in the method:

1. `eval_loss`: The evaluation loss is calculated using the `eval_loss` method.
2. `additional_losses`: The additional losses are calculated using the `eval_loss` method.

```latex
\text{eval\_loss}, \text{additional\_losses} = \text{eval\_loss}(\text{targets}, \text{predictions})
```

Please note that the actual mathematical operations or procedures performed within the `update_metrics` method are not explicitly mentioned in the provided code snippet.

### Method **`train_loss`** Overview
The `train_loss` method is used to compute the training loss for a model. It takes the following parameters:

- `targets`: A dictionary of target names to target tensors.
- `predictions`: A dictionary of output names to output tensors.
- `regularization_type`: One of 'l1', 'l2', 'l1_l2', or None. (optional)
- `regularization_lambda`: The regularization lambda. (optional)

The method returns a tuple containing the loss tensor and a dictionary of loss for every output feature.

Here is a breakdown of the mathematical operations or procedures performed by the `train_loss` method:

1. Initialize `train_loss` and `of_train_losses` variables to 0 and an empty dictionary, respectively.
2. Iterate over each output feature in `self.output_features`.
3. If the output feature is of type `TextOutputFeature`, perform the following operations:
   - Align the target length with the predictions length to enable text metric evaluation.
   - Remove left padding from target tensors.
   - Pad the target tensors with -100 to ensure equal length for loss computation.
   - Re-align target tensors without padding to have equal length before aligning with the prediction tensors.
4. Create a new dictionary `predictions` and copy the values from `_predictions[of_name]` to it.
5. Compute the output feature train loss using the `of_obj.train_loss` method.
6. Multiply the output feature train loss by the weight of the output feature and add it to `train_loss`.
7. Store the output feature train loss in the `of_train_losses` dictionary.
8. Compute any additional losses using the `self.losses` method and add them to `train_loss`.
9. If regularization is enabled, compute the regularization loss using the `reg_loss` function and add it to `train_loss`.
10. Return the `train_loss` and `of_train_losses`.

The mathematical operations or procedures performed by the `train_loss` method can be summarized using LaTeX code as follows:


$$
\begin{align*}
\text{train\_loss} &= 0 \\
\text{of\_train\_losses} &= \{\} \\
\text{for each } \text{of\_name}, \text{of\_obj} \text{ in } \text{self.output\_features}: \\
&\quad \text{if } \text{of\_obj} \text{ is of type TextOutputFeature:} \\
&\quad \quad \text{Align target length with predictions length} \\
&\quad \quad \text{Remove left padding from target tensors} \\
&\quad \quad \text{Pad target tensors with -100} \\
&\quad \quad \text{Re-align target tensors without padding} \\
&\quad \text{Create a new dictionary predictions and copy values from } \_predictions[\text{of\_name}] \\
&\quad \text{Compute output feature train loss} \\
&\quad \text{Multiply output feature train loss by weight and add to train\_loss} \\
&\quad \text{Store output feature train loss in of\_train\_losses} \\
\text{Compute additional losses and add to train\_loss} \\
\text{if regularization is enabled:} \\
&\quad \text{Compute regularization loss and add to train\_loss} \\
\text{Return train\_loss and of\_train\_losses}
\end{align*}
$$

### Method **`eval_loss`** Overview
The `eval_loss` method is used to compute all evaluation losses for the model given targets and predictions. It takes two parameters:

1. `targets`: A dictionary of target names to target tensors. This parameter represents the ground truth values for the model's outputs.
2. `predictions`: A dictionary of output names to output tensors. This parameter represents the predicted values by the model.

The method returns a tuple of loss values for evaluation losses and additional losses.

The method starts by initializing the `eval_loss` variable to 0. Then, it iterates over each output feature in the `output_features` dictionary of the model. For each output feature, it checks if it is an instance of `TextOutputFeature`. If it is, it aligns the target length with the predictions length using the `realign_target_and_prediction_tensors` function. This alignment is necessary for text metric evaluation. Then, it calls the `eval_loss` method of the output feature object to compute the evaluation loss for that feature.

If the output feature is not a `TextOutputFeature`, the method currently has a TODO comment indicating that loss updates need to be figured out. It also mentions that the `SequenceSoftmaxCrossEntropyLoss` function requires logits instead of predictions. Currently, the method fills the `of_eval_loss` variable with zeros as a placeholder.

After computing the evaluation loss for each output feature, the method calculates the weighted sum of the evaluation losses by multiplying each evaluation loss with its corresponding output feature's weight. The weighted evaluation losses are accumulated in the `eval_loss` variable.

Next, the method initializes the `additional_loss` variable to 0. It then calls the `losses` method of the model to get a list of additional losses. If there are additional losses, the method computes their sum using `torch.sum` and accumulates it in the `additional_loss` variable.

Finally, the method returns a tuple containing the evaluation loss (`eval_loss`) and the additional loss (`additional_loss`).

The mathematical operations performed by the `eval_loss` method can be summarized as follows:

1. Initialize `eval_loss` to 0.
2. Iterate over each output feature:
   - If the output feature is a `TextOutputFeature`:
     - Align the target length with the predictions length.
     - Compute the evaluation loss for the feature using the `eval_loss` method of the output feature object.
   - If the output feature is not a `TextOutputFeature`:
     - TODO: Figure out loss updates.
     - Set `of_eval_loss` to 0 as a placeholder.
   - Accumulate the weighted evaluation loss in `eval_loss`.
3. Initialize `additional_loss` to 0.
4. Compute the sum of additional losses and store it in `additional_loss`.
5. Return a tuple containing `eval_loss` and `additional_loss`.

### Method **`outputs_to_predictions`** Overview
The `outputs_to_predictions` method takes in a dictionary `outputs` as a parameter, where the keys are strings representing the names of output features, and the values are tensors representing the model's outputs for each feature.

The purpose of this method is to convert the model's outputs into predictions for each output feature. It returns a dictionary `predictions`, where the keys are the same as the keys in `outputs`, and the values are dictionaries containing the predictions for each output feature.

The method first initializes an empty dictionary `predictions`. Then, for each output feature name `of_name` in the list `self.output_features`, it assigns the value of `outputs` to the corresponding key in `predictions`. This assumes that the model's outputs for each feature are stored in the `outputs` dictionary under the same keys as the output feature names.

The method then returns the `predictions` dictionary.

There are no mathematical operations or procedures performed in this method.

### Method **`save`** Overview
The `save` method is a Python method that is used to save a model to a given path. It takes in two parameters: `self` and `save_path`.

- `self`: This parameter refers to the instance of the class that the method belongs to. It is used to access the attributes and methods of the class.

- `save_path`: This parameter is a string that specifies the path where the model should be saved.

The purpose of the `save` method is to save the model to the specified path. It first checks if the trainer type in the `config_obj` attribute of the class instance is not equal to "none". If it is not "none", it proceeds to save the model weights to the `weights_save_path`, which is obtained by joining the `save_path` and the constant `MODEL_WEIGHTS_FILE_NAME`. The `save_pretrained` method of the model is used to save the weights.

If the trainer type is "none", it logs a message stating that the saving of the model without weight adjustments is skipped.

The `save` method does not perform any mathematical operations or procedures. It is primarily responsible for saving the model weights to the specified path.

### Method **`load`** Overview
The `load` method is a Python method that is used to load a model from a given path. It takes one parameter, `save_path`, which is the path where the model is saved.

The method first constructs the path to the weights file by joining the `save_path` with the constant `MODEL_WEIGHTS_FILE_NAME`. 

Next, it checks if the `adapter` attribute of the `config_obj` is set. If it is, it imports the `PeftConfig` and `PeftModel` classes from the `peft` module. It then creates a new `PeftConfig` object by loading the configuration from the weights file. The `inference_mode` attribute of the config object is set to `False`. 

The method then creates a new `AutoModelForCausalLM` object by loading the base model from the `config.base_model_name_or_path`. Finally, it creates a new `PeftModel` object by loading the weights from the `weights_save_path` and assigns it to the `model` attribute.

If the `adapter` attribute is not set, the method checks if the `trainer.type` attribute of the `config_obj` is not equal to "none". If it is not, it creates a new `AutoModelForCausalLM` object by loading the weights from the `weights_save_path` and assigns it to the `model` attribute.

If neither the `adapter` attribute is set nor the `trainer.type` attribute is not equal to "none", the method logs a message stating that loading the LLM without weight adjustments is skipped.

In terms of mathematical operations or procedures, the `load` method does not perform any specific mathematical operations. It mainly involves loading and initializing the model based on the given parameters and configurations.

### Method **`get_args`** Overview
The `get_args` method in Python is used to retrieve the initialization arguments for constructing a model. It returns a tuple containing the following parameters:

1. `self.config_obj.input_features.to_list()`: This parameter represents the input features of the model. It is obtained from the `input_features` attribute of the `config_obj` object and is converted to a list.

2. `self.config_obj.output_features.to_list()`: This parameter represents the output features of the model. It is obtained from the `output_features` attribute of the `config_obj` object and is converted to a list.

3. `self._random_seed`: This parameter represents the random seed used for initializing the model. It is obtained from the `_random_seed` attribute of the class.

The method returns these parameters as a tuple.

Regarding the mathematical operations or procedures performed by this method, there are none. The `get_args` method simply retrieves and returns the initialization arguments for the model, without performing any mathematical calculations. Therefore, there is no need for LaTex code to display equations in a markdown document.

### Method **`_generate_merged_ids`** Overview
The `_generate_merged_ids` method is a function in a Python class. It takes two parameters: `input_ids` and `target_ids`. 

The purpose of this method is to merge the `input_ids` and `target_ids` tensors together to create a unified tensor that can be passed into the model. It is specifically designed for PEFT (Pre-training with Encoder-decoder Fine-tuning) based fine-tuning. 

If the `target_ids` tensor is `None`, it means that the method is being called during the evaluation of the validation/test sets in the training loop. In this case, the method creates attention masks for the `input_ids` and returns them along with the `input_ids` tensor.

If the `target_ids` tensor is not `None`, the method performs the following steps:

1. It initializes an empty list called `merged_input_and_targets` and another empty list called `lengths`.
2. It creates a tensor called `pad_tensor` which contains the padding token ID from the tokenizer.
3. It iterates over each pair of `input_id_sample` and `target_id_sample` in the `input_ids` and `target_ids` tensors respectively.
4. For each pair, it removes the left padding from both `input_id_sample` and `target_id_sample` using the `remove_left_padding` function.
5. It concatenates the modified `input_id_sample` and `target_id_sample` tensors together, adding the `pad_tensor` at the end of the `target_id_sample` tensor.
6. The resulting tensor is appended to the `merged_input_and_targets` list, and the length of the tensor is appended to the `lengths` list.
7. After iterating over all the pairs, the method determines the maximum length among all the merged tensors.
8. It initializes an empty list called `attention_masks`.
9. It iterates over each merged tensor in the `merged_input_and_targets` list.
10. For each merged tensor, it adds left padding using the `_add_left_padding` method to align it with the maximum length.
11. It generates an attention mask for the merged tensor using the `_create_attention_mask` method.
12. The padded merged tensor and its attention mask are added to the `merged_input_and_targets` and `attention_masks` lists respectively.
13. Finally, the method returns the stacked tensor of merged input and target tensors, and the stacked tensor of attention masks.

The mathematical operations performed in this method involve concatenating tensors, removing left padding, adding left padding, and generating attention masks. These operations are not explicitly mathematical in nature, but rather involve manipulating tensors to prepare them for model input. Therefore, there is no specific LaTex code to display equations in this case.

### Method **`_add_left_padding`** Overview
The `_add_left_padding` method in Python is used to add left padding to a tensor of input IDs. It takes three parameters:

1. `self`: This parameter refers to the instance of the class that the method belongs to. It is used to access the attributes and methods of the class.

2. `input_ids`: This parameter is a tensor containing the input IDs that need to be padded.

3. `max_length`: This parameter specifies the maximum length of the tensor after padding.

4. `pad_value`: This parameter is an optional parameter that specifies the value to be used for padding. The default value is 0.

The method performs the following mathematical operations or procedures:

1. It calculates the number of padding elements required by subtracting the length of `input_ids` from `max_length`.

2. It creates a tensor called `padding` using the `torch.tensor` function. This tensor contains `max_length - input_ids.shape[0]` elements, all initialized with the `pad_value`. The `dtype` of the tensor is set to `torch.int32`, and the `device` is set to the same device as `input_ids`.

3. It concatenates the `padding` tensor with the `input_ids` tensor along the last dimension using the `torch.cat` function. This effectively adds the padding elements to the left of the `input_ids` tensor.

4. Finally, it returns the padded tensor.

Here is the LaTex code to display the equations in a markdown document:


$$
\text{{padding}} = \text{{torch.tensor}}([pad\_value] \times (\text{{max\_length}} - \text{{input\_ids.shape[0]}}), \text{{dtype=torch.int32}}, \text{{device=input\_ids.device}})
$$


$$
\text{{result}} = \text{{torch.cat}}((\text{{padding}}, \text{{input\_ids}}), \text{{dim=-1}})
$$

### Method **`_create_attention_mask`** Overview
The `_create_attention_mask` method in Python is used to create an attention mask for the `input_ids` tensor. The purpose of this method is to identify the padding tokens in the input sequence and create a mask that indicates which tokens should be attended to and which should be ignored during the model's training or inference.

Parameters:
- `self`: The instance of the class that the method belongs to.
- `input_ids`: The input tensor containing the tokenized sequence.

Mathematical operations or procedures:
1. The method compares each element of the `input_ids` tensor with the padding token ID using the inequality operator `!=`.
2. The result of the comparison is a boolean tensor where `True` indicates that the token is not a padding token and `False` indicates that the token is a padding token.
3. The boolean tensor is then converted to a float tensor using the `float()` method.
4. The resulting attention mask tensor is returned.

LaTeX code for the mathematical operations:
Let \( \text{{input\_ids}} \) be the input tensor containing the tokenized sequence and \( \text{{pad\_token\_id}} \) be the ID of the padding token. The attention mask \( \text{{attention\_mask}} \) is created as follows:


$$
\text{{attention\_mask}} = \text{{float}}(\text{{input\_ids}} \neq \text{{pad\_token\_id}})
$$

### Method **`get_augmentation_pipelines`** Overview
The `get_augmentation_pipelines` method is a Python method that returns the augmentation pipeline for a specific model. It takes no parameters and returns an instance of the `AugmentationPipelines` class.

The purpose of the `get_augmentation_pipelines` method is to provide a way to access the augmentation pipeline associated with a model. An augmentation pipeline is a sequence of data transformations that are applied to input data to create augmented versions of the data. These augmented versions can be used to increase the diversity of the training data and improve the performance of the model.

The method does not perform any mathematical operations or procedures. It simply returns an empty instance of the `AugmentationPipelines` class, which represents the augmentation pipeline for the model. The empty dictionary `{}` passed as an argument to the `AugmentationPipelines` constructor indicates that there are no augmentation transformations defined for the model.

## Function **`realign_target_and_prediction_tensors`** Overview
The `realign_target_and_prediction_tensors` function takes in several parameters:

- `targets` (Dict[str, torch.Tensor]): A dictionary containing the target tensor.
- `predictions` (Dict[str, torch.Tensor]): A dictionary containing the prediction tensor.
- `of_name` (str): The name of the output feature.
- `tokenizer` (PreTrainedTokenizer): The tokenizer used for padding.
- `pad_direction` (str, optional): The direction to pad the tensors. Can be 'left' or 'right'. Defaults to 'right'.
- `pad_value` (int, optional): The value to use for padding. If not provided, it defaults to the tokenizer's pad token ID or EOS token ID.

The purpose of this function is to realign the target tensor with the predictions. This is necessary for text metrics that require the target and prediction to be of the same length.

The function first checks if the target length is already equal to the prediction length. If they are equal, it simply returns the targets and predictions as they are.

If the target length is greater than the prediction length, it pads the predictions to match the target length. The number of zeros to add is calculated as the difference between the target length and the prediction length. The padding is done based on the `pad_direction` parameter. If `pad_direction` is "right", the predictions tensor is padded on the right side. If `pad_direction` is "left", the predictions tensor is padded on the left side.

If the target length is smaller than the prediction length, it pads the targets to match the prediction length. Again, the padding is done based on the `pad_direction` parameter.

After padding, the function converts the tensors to float32 type, as metric computation requires float32 tensors.

Finally, the function returns the realigned target tensor and the predictions tensor.

