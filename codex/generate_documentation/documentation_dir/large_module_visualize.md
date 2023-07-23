# Module:`visualize.py` Overview

## **Error in generating module level documentation**

## Function **`_convert_ground_truth`** Overview
The function `_convert_ground_truth` takes in four parameters: `ground_truth`, `feature_metadata`, `ground_truth_apply_idx`, and `positive_label`. 

The purpose of this function is to convert the ground truth data into a numpy array representation. 

The function first checks if the `feature_metadata` dictionary contains a key called "str2idx". If it does, it means that the output feature is categorical and needs to be converted to a binary representation. The function then calls the `_vectorize_ground_truth` function to convert the ground truth data using the mapping provided in `feature_metadata["str2idx"]`. After that, it converts the category index to a binary representation by comparing it with the `positive_label`.

If the `feature_metadata` dictionary does not contain "str2idx", it means that the output feature is binary. In this case, the function checks if the dictionary contains "str2bool". If it does, it means that the boolean representation is non-standard, and the function calls `_vectorize_ground_truth` to convert the ground truth data using the mapping provided in `feature_metadata["str2bool"]`. If "str2bool" is not present, it assumes a standard boolean representation and converts the ground truth data to a numpy array using the `.values` attribute.

Finally, the function ensures that the `positive_label` is set to 1 for binary features. It then converts the ground truth data to a 0/1 representation by using the `astype(int)` method and returns both the converted ground truth and the positive label.

### **Function Details**
The given code defines a function `_convert_ground_truth` that converts the ground truth values to a binary representation.

The function takes four parameters:
- `ground_truth`: The ground truth values to be converted.
- `feature_metadata`: A dictionary containing metadata information about the features.
- `ground_truth_apply_idx`: The indices of the ground truth values to be converted.
- `positive_label`: The label to be considered as positive in the binary representation.

The function first checks if the feature metadata contains a key "str2idx". If it does, it means that the output feature is categorical and needs to be converted to a binary representation. The `_vectorize_ground_truth` function is called to convert the ground truth values using the mapping provided in the feature metadata. Then, the ground truth values are compared with the positive label to create a binary representation.

If the feature metadata does not contain the key "str2idx", it means that the output feature is binary. The function checks if the feature metadata contains a key "str2bool". If it does, it means that the boolean representation of the output feature is non-standard and needs to be converted using the mapping provided in the feature metadata. Otherwise, the ground truth values are assumed to be in the standard boolean representation.

Finally, the ground truth values are converted to 0/1 representation by casting them to integers, and the positive label is set to 1. The converted ground truth values and the positive label are returned.

## Function **`_vectorize_ground_truth`** Overview
The function `_vectorize_ground_truth` takes three parameters: `ground_truth`, `str2idx`, and `ground_truth_apply_idx`. It returns a numpy array.

The function first checks if `ground_truth_apply_idx` is False. If it is, it means that the `ground_truth` values are already in the desired format and don't need to be converted using `str2idx`. In this case, the function uses `np.vectorize` to apply a lambda function to each element of `ground_truth` and returns the result.

If `ground_truth_apply_idx` is True, it means that the `ground_truth` values need to be converted using `str2idx`. The function tries to vectorize the `_encode_categorical_feature` function using `np.vectorize` and applies it to `ground_truth` and `str2idx`. If this operation raises a `KeyError`, it means that the conversion using `str2idx` failed for some reason. In this case, the function logs an error message and falls back to using a lambda function with `np.vectorize` to return the original `ground_truth` values without any conversion.

### **Function Details**
The given code is a function named `_vectorize_ground_truth` that takes three parameters: `ground_truth`, `str2idx`, and `ground_truth_apply_idx`. It returns a numpy array.

The function first checks if `ground_truth_apply_idx` is False. If it is, it directly applies a lambda function to `ground_truth` and `str2idx` using `np.vectorize` and returns the result.

If `ground_truth_apply_idx` is True, the function tries to vectorize `_encode_categorical_feature` function with `ground_truth` and `str2idx` using `np.vectorize`. If a KeyError occurs during this process, it logs an info message and falls back to applying a lambda function to `ground_truth` and `str2idx` using `np.vectorize` and returns the result.

Note: The code assumes that `pd` and `np` are imported and `logger` is defined elsewhere.

## Function **`validate_conf_thresholds_and_probabilities_2d_3d`** Overview
The function `validate_conf_thresholds_and_probabilities_2d_3d` is used to validate the input probabilities and threshold output feature names for a model. It ensures that both the probabilities and threshold output feature names arrays have exactly two members each.

The function takes two parameters:
- `probabilities`: A list of probabilities per model.
- `threshold_output_feature_names`: A list of threshold output feature names per model.

If either the probabilities or threshold output feature names arrays do not have exactly two members, a `RuntimeError` is raised with an error message indicating the number of items provided for each array.

The function uses a dictionary `validation_mapping` to map the input parameters to their respective names. It then iterates over the items in the dictionary and checks the length of each value. If the length is not equal to 2, an error message is logged and a `RuntimeError` is raised.

Overall, the function ensures that the input probabilities and threshold output feature names are valid and have the correct number of members.

### **Function Details**
The code defines a function called `validate_conf_thresholds_and_probabilities_2d_3d` that takes two parameters: `probabilities` and `threshold_output_feature_names`. 

The function checks if both `probabilities` and `threshold_output_feature_names` have exactly two elements. If either of them has a different length, a `RuntimeError` is raised with an error message indicating the expected length.

The function uses a dictionary called `validation_mapping` to map the parameter names to their corresponding values. It then iterates over the items in the dictionary and checks the length of each value. If the length is not equal to 2, an error message is logged using a logger and a `RuntimeError` is raised with the error message.

Note: The code snippet provided is incomplete as it references a logger object that is not defined in the code.

## Function **`load_data_for_viz`** Overview
The function `load_data_for_viz` is used to load JSON files containing model experiment statistics for a list of models. 

The function takes the following parameters:
- `load_type`: The type of data loader to be used. It can be either "load_json" or "load_from_file".
- `model_file_statistics`: A JSON file or a list of JSON files containing the model experiment statistics.
- `dtype`: The data type to be used when loading the files. It is set to `int` by default.
- `ground_truth_split`: The ground truth split to be used when loading the files. It is set to 2 by default.

The function returns a list of training statistics loaded as JSON objects.

Internally, the function uses a dictionary `supported_load_types` to map the `load_type` parameter to the corresponding data loader function. It then calls the appropriate data loader function to load the statistics from the JSON file(s). If there is an error opening the file(s), an exception is raised. Finally, the function returns the loaded statistics as a list.

### **Function Details**
The given code defines a function called `load_data_for_viz` that loads JSON files containing model experiment statistics. The function takes several parameters:

- `load_type`: The type of data loader to be used. It can be either "load_json" or "load_from_file".
- `model_file_statistics`: The JSON file or list of JSON files containing the model experiment stats.
- `dtype`: The data type to be used when loading the files. The default value is `int`.
- `ground_truth_split`: The ground truth split to be used when loading the files. The default value is `2`.

The function returns a list of training statistics loaded as JSON objects.

The function first defines a dictionary called `supported_load_types` that maps the supported load types to their corresponding loader functions. The loader functions are `load_json` and `load_from_file`, which are not defined in the given code.

The function then selects the appropriate loader function based on the `load_type` parameter.

Next, the function tries to load the training statistics from the JSON file(s) using the selected loader function. It iterates over the `model_file_statistics` list and calls the loader function for each file. The loaded statistics are stored in a list called `stats_per_model`.

If there is an exception during the loading process, the function logs an error message and raises the exception.

Finally, the function returns the `stats_per_model` list.

## Function **`load_training_stats_for_viz`** Overview
The function `load_training_stats_for_viz` is used to load model training statistics for visualization. It takes in several parameters: `load_type` (the type of data loader to be used), `model_file_statistics` (a JSON file or list of JSON files containing the model experiment stats), `dtype` (the data type to be used, with the default value as `int`), and `ground_truth_split` (the split value for ground truth, with the default value as 2).

The function first calls another function `load_data_for_viz` to load the data for visualization, passing the parameters `load_type`, `model_file_statistics`, `dtype`, and `ground_truth_split`. The returned data is stored in the variable `stats_per_model`.

Then, the function attempts to load the model statistics as `TrainingStats` objects using the `TrainingStats.Schema().load(j)` method for each item in `stats_per_model`. If an exception occurs during the loading process, an error message is logged and the exception is raised.

Finally, the function returns the loaded model statistics as a list of `TrainingStats` objects.

### **Function Details**
The given code defines a function called `load_training_stats_for_viz` that loads model file data (specifically training statistics) for a list of models. 

The function takes the following parameters:
- `load_type`: The type of data loader to be used.
- `model_file_statistics`: A JSON file or a list of JSON files containing the model experiment statistics.
- `dtype`: The data type to be used for loading the statistics (default is `int`).
- `ground_truth_split`: The ground truth split value (default is `2`).

The function returns a list of model statistics loaded as `TrainingStats` objects.

The function first calls another function called `load_data_for_viz` to load the data for visualization. Then, it tries to load the statistics for each model using the `TrainingStats.Schema().load` method. If an exception occurs during the loading process, it logs an error message and raises the exception.

Overall, this function is used to load and process training statistics for visualization purposes.

## Function **`convert_to_list`** Overview
The function `convert_to_list` takes an input `item` and checks if it is an instance of the list class or if it is None. If `item` is already a list or None, it returns the original `item`. Otherwise, it creates a new list containing `item` and returns that list. 

In summary, the function `convert_to_list` ensures that the input `item` is always returned as a list, either by wrapping it in a list if it is not already a list or by returning the original list if it is already a list or None.

### **Function Details**
The given code defines a function called `convert_to_list` that takes an argument called `item`. The purpose of this function is to check if `item` is an instance of the `list` class or `None`. If it is not, the function will put `item` inside a list and return it. If `item` is already a list or `None`, the function will return `item` as it is.

Here is an example usage of the function:

```python
result = convert_to_list(5)
print(result)  # Output: [5]

result = convert_to_list([1, 2, 3])
print(result)  # Output: [1, 2, 3]

result = convert_to_list(None)
print(result)  # Output: None
```

In the first example, `5` is not a list, so the function puts it inside a list and returns `[5]`. In the second example, `[1, 2, 3]` is already a list, so the function returns it as it is. In the third example, `None` is a special case where the function returns `None` without putting it inside a list.

## Function **`_validate_output_feature_name_from_train_stats`** Overview
The function `_validate_output_feature_name_from_train_stats` takes two parameters: `output_feature_name` and `train_stats_per_model`. 

It validates the `output_feature_name` by checking if it exists in the `train_stats_per_model` and returns it as a list. 

First, it creates an empty set called `output_feature_names_set`. Then, it iterates over each `train_stats` in the `train_stats_per_model`. For each `train_stats`, it iterates over the keys of the `training`, `validation`, and `test` dictionaries and adds them to the `output_feature_names_set`.

Next, it tries to check if the `output_feature_name` exists in the `output_feature_names_set`. If it does, it returns a list containing the `output_feature_name`. If it doesn't, it returns the `output_feature_names_set` as is.

If the `output_feature_name` is an empty iterable (e.g. `[]` in a set), a `TypeError` is raised. In this case, the function also returns the `output_feature_names_set`.

### **Function Details**
This code defines a function called `_validate_output_feature_name_from_train_stats`. 

The function takes two parameters: `output_feature_name` and `train_stats_per_model`. 

The purpose of the function is to validate the `output_feature_name` by checking if it exists in the `train_stats_per_model` and return it as a list. 

The function first creates an empty set called `output_feature_names_set`. 

Then, it iterates over each `train_stats` in the `train_stats_per_model` list. 

For each `train_stats`, it iterates over the keys of the `training`, `validation`, and `test` dictionaries. 

It adds each key to the `output_feature_names_set`. 

After iterating over all the `train_stats`, the function checks if the `output_feature_name` exists in the `output_feature_names_set`. 

If it does, it returns a list containing the `output_feature_name`. 

If it doesn't, it returns the `output_feature_names_set` as is. 

If the `output_feature_name` is an empty iterable (e.g. `[]` in a set), a `TypeError` is raised and the function returns the `output_feature_names_set`.

## Function **`_validate_output_feature_name_from_test_stats`** Overview
The function `_validate_output_feature_name_from_test_stats` takes two parameters: `output_feature_name` and `test_stats_per_model`. 

It first creates an empty set called `output_feature_names_set`. Then, it iterates over each element `ls` in the `test_stats_per_model` list. For each element, it iterates over each key in `ls` and adds it to the `output_feature_names_set`.

Next, it tries to check if the `output_feature_name` is present in the `output_feature_names_set`. If it is, it returns a list containing only the `output_feature_name`. Otherwise, it returns the entire `output_feature_names_set`.

If the `output_feature_name` is an empty iterable (e.g. an empty list), a `TypeError` is raised. In this case, the function also returns the entire `output_feature_names_set`.

In summary, the function validates the `output_feature_name` by checking if it is present in the `test_stats_per_model` and returns it as a list. If the `output_feature_name` is not present or is an empty iterable, it returns all the output feature names found in the `test_stats_per_model`.

### **Function Details**
The code is a function that validates a prediction output feature name from model test statistics. It takes two parameters: `output_feature_name` (the output feature name containing the ground truth) and `test_stats_per_model` (a list of per model test statistics).

The function first creates an empty set called `output_feature_names_set`. It then iterates over each element `ls` in `test_stats_per_model` and for each key in `ls`, it adds the key to the `output_feature_names_set`.

Next, the function checks if the `output_feature_name` is in the `output_feature_names_set`. If it is, it returns a list containing the `output_feature_name`. Otherwise, it returns the `output_feature_names_set`.

If the `output_feature_name` is an empty iterable (e.g. `[]` in a set), a `TypeError` is raised. In this case, the function also returns the `output_feature_names_set`.

Overall, the function ensures that the `output_feature_name` is valid by checking if it exists in the `test_stats_per_model` and returns it as a list. If it doesn't exist, it returns all the output feature names found in the `test_stats_per_model`.

## Function **`_encode_categorical_feature`** Overview
The function `_encode_categorical_feature` takes in a numpy array `raw` which represents categorical string values and a dictionary `str2idx` that maps string representations to encoded numeric values. 

The function encodes each string value in `raw` to its corresponding encoded numeric value using the `str2idx` dictionary. It then returns a numpy array containing the encoded values.

### **Function Details**
The given code defines a function `_encode_categorical_feature` that takes in two parameters: `raw` and `str2idx`. 

The function is used to encode a raw categorical string value to an encoded numeric value using a dictionary `str2idx` that maps string representations to encoded values.

The function returns the encoded value corresponding to the input `raw` string.

## Function **`_get_ground_truth_df`** Overview
The function `_get_ground_truth_df` takes a string parameter `ground_truth` and returns a DataFrame. 

The function first determines the data format of the ground truth data by calling the `figure_data_format_dataset` function. It then checks if the data format is supported by checking if it is in the `CACHEABLE_FORMATS` list. If the data format is not supported, a `ValueError` is raised.

Next, the function retrieves the appropriate reader for the data format by calling the `get_from_registry` function with the data format and `data_reader_registry` as arguments.

If the data format is either "csv" or "tsv", the function calls the reader with the `ground_truth` parameter, `dtype=None`, and `df_lib=pd` to allow type inference. Otherwise, the function calls the reader with just the `ground_truth` parameter and `df_lib=pd`.

In summary, the function `_get_ground_truth_df` determines the data format of the ground truth data, checks if it is supported, retrieves the appropriate reader, and then uses the reader to read the ground truth data into a DataFrame.

### **Function Details**
The code defines a function `_get_ground_truth_df` that takes a string parameter `ground_truth` and returns a DataFrame. 

The function first determines the data format of the ground truth data by calling the `figure_data_format_dataset` function. It then checks if the data format is in the `CACHEABLE_FORMATS` list. If not, it raises a `ValueError` with a message indicating that the data format is not supported.

Next, the function retrieves the appropriate reader for the data format by calling the `get_from_registry` function with the data format and the `data_reader_registry` as arguments.

If the data format is either "csv" or "tsv", the function calls the reader with the `ground_truth` parameter, `dtype=None`, and `df_lib=pd` to allow type inference. Otherwise, it calls the reader with just the `ground_truth` parameter and `df_lib=pd`.

Finally, the function returns the result of calling the reader function.

## Function **`_extract_ground_truth_values`** Overview
The function `_extract_ground_truth_values` is a helper function that is used to extract ground truth values from a source dataset. It takes several arguments:

- `ground_truth`: This can be either a string representing the path to the source data containing ground truth or a DataFrame object representing the ground truth data itself.
- `output_feature_name`: This is a string representing the name of the output feature for the ground truth values.
- `ground_truth_split`: This is an integer representing the dataset split to use for the ground truth. It defaults to 2.
- `split_file`: This is an optional argument that can be a string representing the file path to split values or None.

The function first checks if the `ground_truth` argument is a string or a DataFrame. If it is a string, it calls the `_get_ground_truth_df` function to retrieve the ground truth DataFrame. Otherwise, it uses the `ground_truth` DataFrame directly.

Next, the function checks if the ground truth DataFrame has a column named "SPLIT". If it does, it retrieves the split values from that column and filters the ground truth values based on the `ground_truth_split` argument.

If the ground truth DataFrame does not have a "SPLIT" column, the function checks if the `split_file` argument is provided. If it is, the function reads the split values from the file. If the file has a ".csv" extension, it uses the `load_array` function to load the split values. Otherwise, it assumes the file is in Parquet format and uses `pd.read_parquet` to read the split values.

Finally, if neither the "SPLIT" column nor the `split_file` argument is available, the function returns all the data in the `ground_truth` DataFrame for the specified `output_feature_name`.

The function returns a pandas Series object containing the extracted ground truth values.

### **Function Details**
The given code is a function named `_extract_ground_truth_values` that is used to extract ground truth values from a source data set. Here is a breakdown of the code:

1. The function takes the following parameters:
   - `ground_truth`: Either a string representing the path to the source data containing ground truth or a DataFrame containing the ground truth data.
   - `output_feature_name`: A string representing the output feature name for the ground truth values.
   - `ground_truth_split`: An integer representing the dataset split to use for the ground truth. It defaults to 2.
   - `split_file`: An optional string representing the file path to the split values. It defaults to None.

2. The function first checks if the `ground_truth` parameter is a string or a DataFrame. If it is a string, it calls a helper function `_get_ground_truth_df` to load the ground truth DataFrame from the specified path. If it is already a DataFrame, it assigns it to the variable `ground_truth_df`.

3. The function then checks if the ground truth DataFrame contains a column named "SPLIT". If it does, it extracts the ground truth values for the specified `ground_truth_split` by filtering the DataFrame using the condition `split == ground_truth_split`.

4. If the ground truth DataFrame does not contain a column named "SPLIT", the function checks if a `split_file` is provided. If it is, the function reads the split values from the file. If the file is in CSV format, it raises a DeprecationWarning and loads the split values using the `load_array` function. If the file is in Parquet format, it reads the split values using `pd.read_parquet`.

5. After obtaining the split values, the function creates a boolean mask `mask` by comparing the split values with the `ground_truth_split` value. The mask is then aligned with the index of the ground truth DataFrame to account for any dropped rows during preprocessing.

6. Finally, the function returns the ground truth values by indexing the `output_feature_name` column of the ground truth DataFrame using the mask.

Note: The code assumes the existence of a helper function `_get_ground_truth_df`, which is not provided in the given code snippet.

## Function **`_get_cols_from_predictions`** Overview
The function `_get_cols_from_predictions` takes three parameters: `predictions_paths`, `cols`, and `metadata`. 

It iterates over each `predictions_path` in the `predictions_paths` list. It reads a parquet file at the given `predictions_path` and assigns it to the variable `pred_df`. 

It then checks if a file with the extension "shapes.json" exists by replacing the file extension of `predictions_path` with "shapes.json" using the `replace_file_extension` function. If the file exists, it loads the JSON data from the file into the `column_shapes` variable. It then calls the `unflatten_df` function to unflatten the `pred_df` DataFrame using the `column_shapes` and a specified backend engine.

Next, it iterates over each `col` in the `cols` list. If the `col` ends with the `_PREDICTIONS_SUFFIX` string, it extracts the `feature_name` by removing the `_PREDICTIONS_SUFFIX` from the end of the `col` string. It retrieves the corresponding `feature_metadata` from the `metadata` dictionary using the `feature_name`. If the `feature_metadata` contains a key "str2idx", it maps the values in the `col` column of `pred_df` to their corresponding indices using a lambda function.

After converting the `pred_df` DataFrame to a numpy dataset using the `to_numpy_dataset` function and the `LOCAL_BACKEND`, it appends the selected columns (`col`) from the `pred_df` to the `results_per_model` list.

Finally, it returns the `results_per_model` list, which contains the selected columns from each `pred_df` DataFrame for each `predictions_path` in the `predictions_paths` list.

### **Function Details**
This code defines a function called `_get_cols_from_predictions` that takes three arguments: `predictions_paths`, `cols`, and `metadata`. 

The function reads parquet files specified by `predictions_paths` and stores the resulting dataframes in `pred_df`. It then checks if a file with the same name as `predictions_path` but with the extension "shapes.json" exists. If it does, it loads the JSON data from that file into `column_shapes` and uses it to unflatten `pred_df` using the `unflatten_df` function. 

Next, the function iterates over each column specified in `cols`. If the column name ends with `_PREDICTIONS_SUFFIX`, it extracts the feature name by removing `_PREDICTIONS_SUFFIX` from the end of the column name. It then retrieves the corresponding feature metadata from `metadata` and checks if it contains a "str2idx" key. If it does, it maps the values in the `col` column of `pred_df` to their corresponding indices using a lambda function.

After converting `pred_df` to a numpy dataset using the `to_numpy_dataset` function, the function appends the specified columns to the `results_per_model` list.

Finally, the function returns the `results_per_model` list, which contains the specified columns from all the prediction dataframes.

## Function **`generate_filename_template_path`** Overview
The function `generate_filename_template_path` takes two parameters: `output_dir` and `filename_template`. 

The purpose of this function is to ensure that a path to a template file can be constructed given an output directory. It first checks if the `output_dir` parameter is not None. If it is not None, it creates the output directory if it does not already exist using the `os.makedirs` function. Then, it returns the path to the filename template inside the output directory by joining the `output_dir` and `filename_template` using `os.path.join`. 

If the `output_dir` parameter is None, the function returns None.

### **Function Details**
The given code is a function named `generate_filename_template_path` that takes two parameters: `output_dir` and `filename_template`. 

The function first checks if the `output_dir` is not None. If it is not None, it creates the output directory using `os.makedirs` with the `exist_ok=True` parameter, which ensures that the directory is created if it does not exist. 

Then, the function returns the path to the filename template by joining the `output_dir` and `filename_template` using `os.path.join`. 

If the `output_dir` is None, the function returns None.

## Function **`compare_performance_cli`** Overview
The function `compare_performance_cli` is a command-line interface (CLI) function that is used to compare the performance of different models. 

The function takes two parameters:
- `test_statistics`: A path to a file or a list of paths to files containing experiment test statistics.
- `kwargs`: Additional parameters for the requested visualizations.

The function first loads the model data from the specified files using the `load_data_for_viz` function with the "load_json" option. The loaded data is then passed to the `compare_performance` function along with the additional parameters.

The function does not return any value, as indicated by the `None` type in the function signature.

### **Function Details**
The given code defines a function `compare_performance_cli` that takes in two parameters: `test_statistics` and `kwargs`. 

The `test_statistics` parameter can be either a string or a list of strings representing the path(s) to experiment test statistics file(s). 

The `kwargs` parameter is a dictionary that contains parameters for the requested visualizations.

Inside the function, the `load_data_for_viz` function is called to load the model data from the test statistics file(s) using the "load_json" method. The loaded data is then passed to the `compare_performance` function along with the `kwargs` parameters.

The function does not return anything (`None` is returned).

## Function **`learning_curves_cli`** Overview
The function `learning_curves_cli` is a command-line interface for loading model data from files and displaying learning curves. 

The function takes two parameters:
- `training_statistics`: A path to the file(s) containing training statistics for the experiment. It can be a single file path or a list of file paths.
- `kwargs`: Additional parameters for the requested visualizations.

The function first loads the training statistics from the file(s) using the `load_training_stats_for_viz` function. This function uses the "load_json" method to load the data from the file(s).

Then, the function calls the `learning_curves` function, passing the loaded training statistics and the additional parameters. The `learning_curves` function is responsible for generating and displaying the learning curves based on the provided data.

The function does not return any value, as indicated by the `None` return type.

### **Function Details**
The given code defines a function `learning_curves_cli` that takes in two parameters: `training_statistics` and `kwargs`. 

The `training_statistics` parameter can be either a string or a list of strings representing the path(s) to the experiment training statistics file(s).

The `kwargs` parameter is a dictionary that contains additional parameters for the requested visualizations.

Inside the function, the `load_training_stats_for_viz` function is called to load the training statistics data from the file(s) specified by `training_statistics`. The loaded data is then passed to the `learning_curves` function along with the additional parameters from `kwargs`.

The function does not return anything (`None`).

Note: The `Union` and `List` types are imported from the `typing` module.

## Function **`compare_classifiers_performance_from_prob_cli`** Overview
The function `compare_classifiers_performance_from_prob_cli` is a command-line interface (CLI) function that compares the performance of different classifiers based on their prediction probabilities. 

The function takes several input parameters:
- `probabilities`: A list of file names containing prediction results. These files are used to extract the probabilities.
- `ground_truth`: The path to the ground truth file.
- `ground_truth_split`: An integer indicating the type of ground truth split. 0 represents the training split, 1 represents the validation split, and 2 represents the test split.
- `split_file`: The path to a CSV file containing split values. This parameter is optional and can be set to `None`.
- `ground_truth_metadata`: The path to a JSON file containing feature metadata that was created during training.
- `output_feature_name`: The name of the output feature to visualize.
- `output_directory`: The name of the output directory where the training results will be stored.
- `kwargs`: Additional parameters for the requested visualizations.

The function first loads the feature metadata from the `ground_truth_metadata` file. It then extracts the ground truth values based on the provided parameters. Next, it retrieves the prediction probabilities for each model from the `probabilities` files and converts them to encoded values using the feature metadata.

Finally, the function calls the `compare_classifiers_performance_from_prob` function with the extracted probabilities, ground truth values, metadata, output feature name, output directory, and any additional parameters.

### **Function Details**
The given code defines a function called `compare_classifiers_performance_from_prob_cli`. This function takes several input parameters including `probabilities`, `ground_truth`, `ground_truth_split`, `split_file`, `ground_truth_metadata`, `output_feature_name`, `output_directory`, and `kwargs`. 

The function first loads the feature metadata from a JSON file using the `load_json` function. It then extracts the ground truth values based on the provided parameters using the `_extract_ground_truth_values` function. 

Next, it retrieves the probabilities for each model from the prediction files using the `_get_cols_from_predictions` function. The probabilities are stored in the `probabilities_per_model` variable. 

Finally, it calls the `compare_classifiers_performance_from_prob` function with the extracted ground truth, probabilities per model, metadata, output feature name, output directory, and any additional keyword arguments provided.

## Function **`compare_classifiers_performance_from_pred_cli`** Overview
The function `compare_classifiers_performance_from_pred_cli` is a command-line interface (CLI) function that compares the performance of different classifiers based on their prediction results.

The function takes the following inputs:
- `predictions`: A list of prediction results file names to extract predictions from.
- `ground_truth`: The path to the ground truth file.
- `ground_truth_metadata`: The path to the ground truth metadata file.
- `ground_truth_split`: The type of ground truth split, where `0` represents the training split, `1` represents the validation split, and `2` represents the test split.
- `split_file`: The file path to a CSV file containing split values.
- `output_feature_name`: The name of the output feature to visualize.
- `output_directory`: The name of the output directory containing training results.
- `kwargs`: Additional parameters for the requested visualizations.

The function first loads the feature metadata from the ground truth metadata file. Then, it extracts the ground truth values based on the specified output feature name, ground truth split, and split file.

Next, it retrieves the predictions for each model from the prediction files and converts them to encoded values using the feature metadata.

Finally, it calls the `compare_classifiers_performance_from_pred` function to compare the performance of the classifiers using the predictions, ground truth, metadata, output feature name, and other optional parameters.

### **Function Details**
The given code defines a function `compare_classifiers_performance_from_pred_cli` that compares the performance of different classifiers based on their predictions.

The function takes the following inputs:
- `predictions`: A list of prediction results file names to extract predictions from.
- `ground_truth`: The path to the ground truth file.
- `ground_truth_metadata`: The path to the ground truth metadata file.
- `ground_truth_split`: The type of ground truth split - `0` for training split, `1` for validation split, or `2` for test split.
- `split_file`: The file path to a CSV file containing split values.
- `output_feature_name`: The name of the output feature to visualize.
- `output_directory`: The name of the output directory containing training results.
- `**kwargs`: Additional parameters for the requested visualizations.

The function first loads the feature metadata from the ground truth metadata file using the `load_json` function.

Then, it extracts the ground truth values based on the provided parameters using the `_extract_ground_truth_values` function.

Next, it retrieves the predictions for each model from the prediction files using the `_get_cols_from_predictions` function.

Finally, it calls the `compare_classifiers_performance_from_pred` function to compare the performance of the classifiers based on the predictions, ground truth, and metadata.

The function does not return any value.

## Function **`compare_classifiers_performance_subset_cli`** Overview
The function `compare_classifiers_performance_subset_cli` is a command-line interface (CLI) function that compares the performance of different classifiers on a subset of data. 

The function takes several input parameters:
- `probabilities`: A list of file names containing prediction results. These files are used to extract probabilities.
- `ground_truth`: The path to the ground truth file.
- `ground_truth_split`: An integer indicating the type of ground truth split. 0 represents the training split, 1 represents the validation split, and 2 represents the test split.
- `split_file`: The path to a CSV file containing split values. This parameter is optional and can be set to `None`.
- `ground_truth_metadata`: The path to a JSON file containing feature metadata created during training.
- `output_feature_name`: The name of the output feature to visualize.
- `output_directory`: The name of the output directory where the training results will be stored.
- `kwargs`: Additional parameters for the requested visualizations.

The function first loads the feature metadata from the `ground_truth_metadata` file using the `load_json` function.

Then, it extracts the ground truth values from the `ground_truth` file using the `_extract_ground_truth_values` function, passing the `output_feature_name`, `ground_truth_split`, and `split_file` as parameters.

Next, it retrieves the probabilities for each model from the `probabilities` files using the `_get_cols_from_predictions` function, passing the `col` (constructed from `output_feature_name` and a suffix) and the metadata.

Finally, it calls the `compare_classifiers_performance_subset` function, passing the probabilities per model, ground truth, metadata, output feature name, output directory, and any additional parameters specified in `kwargs`.

The function does not return any value (`None`).

### **Function Details**
The given code defines a function `compare_classifiers_performance_subset_cli` that loads model data from files and calls the `compare_classifiers_performance_subset` function to compare the performance of different classifiers.

The function takes the following parameters:
- `probabilities`: A string or a list of strings representing the file names of prediction results to extract probabilities from.
- `ground_truth`: A string representing the path to the ground truth file.
- `ground_truth_split`: An integer representing the type of ground truth split. `0` for training split, `1` for validation split, or `2` for test split.
- `split_file`: A string representing the file path to a CSV file containing split values.
- `ground_truth_metadata`: A string representing the file path to a feature metadata JSON file created during training.
- `output_feature_name`: A string representing the name of the output feature to visualize.
- `output_directory`: A string representing the name of the output directory containing training results.
- `**kwargs`: Additional keyword arguments for the requested visualizations.

The function first loads the feature metadata from the `ground_truth_metadata` file using the `load_json` function.

Then, it extracts the ground truth values from the `ground_truth` file using the `_extract_ground_truth_values` function, passing the `output_feature_name`, `ground_truth_split`, and `split_file` as arguments.

Next, it retrieves the probabilities per model from the prediction files using the `_get_cols_from_predictions` function, passing the `probabilities` list, the `col` (constructed from `output_feature_name` and `_PROBABILITIES_SUFFIX`), and the metadata.

Finally, it calls the `compare_classifiers_performance_subset` function, passing the probabilities per model, ground truth, metadata, output_feature_name, output_directory, and any additional keyword arguments.

The function does not return any value (`None`).

Note: The code references some functions (`load_json`, `_extract_ground_truth_values`, `_get_cols_from_predictions`, `compare_classifiers_performance_subset`) that are not defined in the given code snippet.

## Function **`compare_classifiers_performance_changing_k_cli`** Overview
The function `compare_classifiers_performance_changing_k_cli` is a command-line interface (CLI) function that is used to compare the performance of different classifiers by changing the value of k. 

The function takes several input parameters:
- `probabilities`: A list of file names containing prediction results. These files are used to extract probabilities.
- `ground_truth`: The path to the ground truth file.
- `ground_truth_split`: The type of ground truth split, which can be 0 for the training split, 1 for the validation split, or 2 for the test split.
- `split_file`: The path to a CSV file containing split values.
- `ground_truth_metadata`: The path to a JSON file containing feature metadata created during training.
- `output_feature_name`: The name of the output feature to visualize.
- `output_directory`: The name of the output directory containing training results.
- `kwargs`: Additional parameters for the requested visualizations.

The function first loads the feature metadata from the `ground_truth_metadata` file. Then, it extracts the ground truth values using the `_extract_ground_truth_values` function, passing the `ground_truth`, `output_feature_name`, `ground_truth_split`, and `split_file` parameters.

Next, the function retrieves the probabilities per model using the `_get_cols_from_predictions` function, passing the `probabilities`, `[col]` (where `col` is the output feature name with a suffix), and `metadata` parameters.

Finally, the function calls the `compare_classifiers_performance_changing_k` function, passing the `probabilities_per_model`, `ground_truth`, `metadata`, `output_feature_name`, `output_directory`, and `kwargs` parameters. This function is responsible for actually comparing the performance of the classifiers and generating the visualizations.

The function does not return any value (`None`).

### **Function Details**
The given code defines a function `compare_classifiers_performance_changing_k_cli` that loads model data from files and calls the `compare_classifiers_performance_changing_k` function to compare the performance of different classifiers.

The function takes the following parameters:
- `probabilities`: A string or a list of strings representing the file names of prediction results to extract probabilities from.
- `ground_truth`: A string representing the path to the ground truth file.
- `ground_truth_split`: A string representing the type of ground truth split. It can be `'0'` for the training split, `'1'` for the validation split, or `'2'` for the test split.
- `split_file`: A string representing the file path to a CSV file containing split values.
- `ground_truth_metadata`: A string representing the file path to a feature metadata JSON file created during training.
- `output_feature_name`: A string representing the name of the output feature to visualize.
- `output_directory`: A string representing the name of the output directory containing training results.
- `**kwargs`: Additional keyword arguments for the requested visualizations.

The function first loads the feature metadata from the `ground_truth_metadata` file using the `load_json` function. Then, it extracts the ground truth values from the source data set using the `_extract_ground_truth_values` function. Next, it retrieves the probabilities per model using the `_get_cols_from_predictions` function. Finally, it calls the `compare_classifiers_performance_changing_k` function with the extracted data and additional parameters.

The function does not return any value (`None`).

Note: The code snippet provided is incomplete and contains some duplicate lines (`split_file` parameter is mentioned twice).

## Function **`compare_classifiers_multiclass_multimetric_cli`** Overview
The function `compare_classifiers_multiclass_multimetric_cli` is a command-line interface (CLI) function that loads model data from files and calls the `compare_classifiers_multiclass_multimetric` function to perform a comparison of multiple classifiers using multiple metrics.

The function takes three parameters:
- `test_statistics`: A path to the experiment test statistics file. It can be a string or a list of strings.
- `ground_truth_metadata`: A path to the ground truth metadata file.
- `kwargs`: Additional parameters for the requested visualizations.

The function first loads the test statistics data using the `load_data_for_viz` function, passing the "load_json" method as an argument. It then loads the ground truth metadata using the `load_json` function.

Finally, the function calls the `compare_classifiers_multiclass_multimetric` function, passing the loaded test statistics data, the metadata, and any additional parameters specified in `kwargs`.

The function does not return any value (`None`).

### **Function Details**
The given code defines a function called `compare_classifiers_multiclass_multimetric_cli`. This function takes three parameters: `test_statistics`, `ground_truth_metadata`, and `kwargs`. 

The `test_statistics` parameter can be either a string or a list of strings representing the path(s) to the experiment test statistics file(s). 

The `ground_truth_metadata` parameter is a string representing the path to the ground truth metadata file.

The `kwargs` parameter is a dictionary that contains additional parameters for the requested visualizations.

Inside the function, the `load_data_for_viz` function is called to load the test statistics data from the file(s) specified by `test_statistics`. The `load_json` function is also called to load the ground truth metadata from the file specified by `ground_truth_metadata`.

Finally, the `compare_classifiers_multiclass_multimetric` function is called with the loaded test statistics data (`test_stats_per_model`), the loaded metadata (`metadata`), and the additional parameters (`kwargs`) passed as arguments.

The function does not return any value (`None`).

## Function **`compare_classifiers_predictions_cli`** Overview
The function `compare_classifiers_predictions_cli` is a command-line interface (CLI) function that is used to compare the predictions of multiple classifiers. 

The function takes several input parameters:
- `predictions`: A list of prediction results file names to extract predictions from.
- `ground_truth`: The path to the ground truth file.
- `ground_truth_split`: The type of ground truth split, which can be `0` for the training split, `1` for the validation split, or `2` for the test split.
- `split_file`: The file path to a CSV file containing split values.
- `ground_truth_metadata`: The file path to a feature metadata JSON file created during training.
- `output_feature_name`: The name of the output feature to visualize.
- `output_directory`: The name of the output directory containing training results.
- `kwargs`: Additional parameters for the requested visualizations.

The function first loads the feature metadata from the `ground_truth_metadata` file. Then, it extracts the ground truth values using the `_extract_ground_truth_values` function, passing the `ground_truth`, `output_feature_name`, `ground_truth_split`, and `split_file` as arguments.

Next, the function retrieves the predictions for each model using the `_get_cols_from_predictions` function, passing the `predictions`, `col` (constructed from `output_feature_name` and a suffix), and `metadata` as arguments.

Finally, the function calls the `compare_classifiers_predictions` function, passing the `predictions_per_model`, `ground_truth`, `metadata`, `output_feature_name`, `output_directory`, and any additional parameters specified in `kwargs`.

### **Function Details**
The given code defines a function `compare_classifiers_predictions_cli` that loads model data from files and calls another function `compare_classifiers_predictions` to compare the predictions made by different classifiers.

The function takes the following parameters:
- `predictions`: A list of prediction results file names to extract predictions from.
- `ground_truth`: The path to the ground truth file.
- `ground_truth_split`: The type of ground truth split - `0` for training split, `1` for validation split, or `2` for `'test'` split.
- `split_file`: The file path to a CSV file containing split values.
- `ground_truth_metadata`: The file path to a feature metadata JSON file created during training.
- `output_feature_name`: The name of the output feature to visualize.
- `output_directory`: The name of the output directory containing training results.
- `**kwargs`: Additional parameters for the requested visualizations.

The function first loads the feature metadata from the `ground_truth_metadata` file using the `load_json` function. Then, it extracts the ground truth values from the source dataset using the `_extract_ground_truth_values` function, passing the `ground_truth`, `output_feature_name`, `ground_truth_split`, and `split_file` as arguments.

Next, it retrieves the predictions for each model from the prediction files using the `_get_cols_from_predictions` function, passing the `predictions`, `[col]` (where `col` is the output feature name with a suffix), and `metadata` as arguments.

Finally, it calls the `compare_classifiers_predictions` function, passing the `predictions_per_model`, `ground_truth`, `metadata`, `output_feature_name`, and `output_directory` as arguments, along with any additional parameters specified in `kwargs`.

The function does not return any value (`None`).

## Function **`compare_classifiers_predictions_distribution_cli`** Overview
The function `compare_classifiers_predictions_distribution_cli` is a command-line interface (CLI) function that compares the predictions of multiple classifiers and visualizes their distribution. 

The function takes the following inputs:
- `predictions`: A list of prediction results file names to extract predictions from.
- `ground_truth`: The path to the ground truth file.
- `ground_truth_split`: The type of ground truth split, where `0` represents the training split, `1` represents the validation split, and `2` represents the test split.
- `split_file`: The file path to a CSV file containing split values.
- `ground_truth_metadata`: The file path to a feature metadata JSON file created during training.
- `output_feature_name`: The name of the output feature to visualize.
- `output_directory`: The name of the output directory containing training results.
- `kwargs`: Additional parameters for the requested visualizations.

The function first loads the feature metadata from the `ground_truth_metadata` file. Then, it extracts the ground truth values based on the `ground_truth`, `output_feature_name`, `ground_truth_split`, and `split_file` parameters.

Next, the function retrieves the predictions for each model specified in the `predictions` list. It uses the `metadata` to convert the raw predictions to encoded values.

Finally, the function calls the `compare_classifiers_predictions_distribution` function to compare the predictions per model, along with the ground truth values. It also passes the `metadata`, `output_feature_name`, `output_directory`, and any additional parameters specified in `kwargs`.

The function does not return any value (`None`).

### **Function Details**
The given code defines a function `compare_classifiers_predictions_distribution_cli` that compares the predictions of different classifiers based on their distribution. 

The function takes the following parameters:
- `predictions`: A list of prediction results file names to extract predictions from.
- `ground_truth`: The path to the ground truth file.
- `ground_truth_split`: The type of ground truth split - `0` for training split, `1` for validation split, or `2` for `'test'` split.
- `split_file`: The file path to a CSV file containing split values.
- `ground_truth_metadata`: The file path to a feature metadata JSON file created during training.
- `output_feature_name`: The name of the output feature to visualize.
- `output_directory`: The name of the output directory containing training results.
- `**kwargs`: Additional parameters for the requested visualizations.

The function first loads the feature metadata from the `ground_truth_metadata` file using the `load_json` function. It then extracts the ground truth values based on the provided parameters using the `_extract_ground_truth_values` function.

Next, it retrieves the predictions for each model from the prediction files using the `_get_cols_from_predictions` function. The predictions are retrieved for the specified `output_feature_name` and stored in the `predictions_per_model` variable.

Finally, the function calls the `compare_classifiers_predictions_distribution` function with the extracted predictions, ground truth, metadata, and other parameters to perform the comparison and visualize the results.

The function does not return any value, as indicated by the `None` return type.

## Function **`confidence_thresholding_cli`** Overview
The function `confidence_thresholding_cli` is a command-line interface function that loads model data from files and performs confidence thresholding on the predictions.

The function takes the following inputs:
- `probabilities`: A list of prediction results file names or a single file name.
- `ground_truth`: The path to the ground truth file.
- `ground_truth_split`: The type of ground truth split, which can be `0` for the training split, `1` for the validation split, or `2` for the test split.
- `split_file`: The file path to a CSV file containing split values.
- `ground_truth_metadata`: The file path to a feature metadata JSON file created during training.
- `output_feature_name`: The name of the output feature to visualize.
- `output_directory`: The name of the output directory containing training results.
- `kwargs`: Additional parameters for the requested visualizations.

The function first loads the feature metadata from the `ground_truth_metadata` file. Then, it extracts the ground truth values from the source dataset using the `_extract_ground_truth_values` function. 

Next, the function retrieves the probabilities for each model from the prediction files using the `_get_cols_from_predictions` function. The probabilities are retrieved for the specified `output_feature_name` and stored in the `probabilities_per_model` variable.

Finally, the function calls the `confidence_thresholding` function, passing the `probabilities_per_model`, `ground_truth`, `metadata`, `output_feature_name`, `output_directory`, and any additional parameters specified in `kwargs`. The `confidence_thresholding` function performs the confidence thresholding on the predictions and generates the visualizations.

The function does not return any value (`None`).

### **Function Details**
The given code defines a function `confidence_thresholding_cli` that loads model data from files and performs confidence thresholding on the predictions.

The function takes the following inputs:
- `probabilities`: A list of prediction results file names or a single file name.
- `ground_truth`: The path to the ground truth file.
- `ground_truth_split`: The type of ground truth split, which can be `0` for training split, `1` for validation split, or `2` for test split.
- `split_file`: The file path to a CSV file containing split values.
- `ground_truth_metadata`: The file path to a feature metadata JSON file created during training.
- `output_feature_name`: The name of the output feature to visualize.
- `output_directory`: The name of the output directory containing training results.
- `kwargs`: Additional parameters for the requested visualizations.

The function first loads the feature metadata from the `ground_truth_metadata` file using the `load_json` function.

Then, it extracts the ground truth values from the source dataset using the `_extract_ground_truth_values` function, passing the `ground_truth`, `output_feature_name`, `ground_truth_split`, and `split_file` as arguments.

Next, it retrieves the probabilities for each model from the prediction files using the `_get_cols_from_predictions` function, passing the `probabilities`, `[col]` (where `col` is the output feature name with a suffix), and `metadata` as arguments.

Finally, it calls the `confidence_thresholding` function, passing the `probabilities_per_model`, `ground_truth`, `metadata`, `output_feature_name`, `output_directory`, and `kwargs` as arguments.

The function does not return any value (`None`).

Note: The code references some functions (`load_json`, `_extract_ground_truth_values`, `_get_cols_from_predictions`, `confidence_thresholding`) that are not defined in the given code snippet.

## Function **`confidence_thresholding_data_vs_acc_cli`** Overview
The function `confidence_thresholding_data_vs_acc_cli` is used to load model data from files and display it using the `confidence_thresholding_data_vs_acc` function. 

The function takes the following inputs:
- `probabilities`: A list of prediction results file names or a single file name.
- `ground_truth`: The path to the ground truth file.
- `ground_truth_split`: The type of ground truth split, which can be `0` for training split, `1` for validation split, or `2` for test split.
- `split_file`: The file path to a CSV file containing split values.
- `ground_truth_metadata`: The file path to a feature metadata JSON file created during training.
- `output_feature_name`: The name of the output feature to visualize.
- `output_directory`: The name of the output directory containing training results.
- `kwargs`: Additional parameters for the requested visualizations.

The function first loads the feature metadata from the `ground_truth_metadata` file. Then, it extracts the ground truth values using the `_extract_ground_truth_values` function. Next, it retrieves the probabilities per model using the `_get_cols_from_predictions` function. Finally, it calls the `confidence_thresholding_data_vs_acc` function with the retrieved data and additional parameters.

The function does not return any value.

### **Function Details**
The given code defines a function `confidence_thresholding_data_vs_acc_cli` that loads model data from files and visualizes the data using the `confidence_thresholding_data_vs_acc` function.

The function takes the following parameters:
- `probabilities`: A string or a list of strings representing the file names of prediction results files to extract probabilities from.
- `ground_truth`: A string representing the path to the ground truth file.
- `ground_truth_split`: A string representing the type of ground truth split. It can be `'0'` for the training split, `'1'` for the validation split, or `'2'` for the test split.
- `split_file`: A string representing the file path to a CSV file containing split values.
- `ground_truth_metadata`: A string representing the file path to a feature metadata JSON file created during training.
- `output_feature_name`: A string representing the name of the output feature to visualize.
- `output_directory`: A string representing the name of the output directory containing training results.
- `kwargs`: Additional parameters for the requested visualizations.

The function first loads the feature metadata from the `ground_truth_metadata` file using the `load_json` function. Then, it extracts the ground truth values from the `ground_truth` file using the `_extract_ground_truth_values` function. Next, it retrieves the probabilities per model from the prediction files using the `_get_cols_from_predictions` function. Finally, it calls the `confidence_thresholding_data_vs_acc` function to visualize the data, passing the probabilities per model, ground truth, metadata, output feature name, and other parameters.

The function does not return any value.

## Function **`confidence_thresholding_data_vs_acc_subset_cli`** Overview
The function `confidence_thresholding_data_vs_acc_subset_cli` is a command-line interface function that loads model data from files and passes it to the `confidence_thresholding_data_vs_acc_subset` function for visualization.

The function takes the following inputs:
- `probabilities`: A string or a list of strings representing the file names of prediction results files from which probabilities will be extracted.
- `ground_truth`: A string representing the path to the ground truth file.
- `ground_truth_split`: A string representing the type of ground truth split. It can be '0' for the training split, '1' for the validation split, or '2' for the 'test' split.
- `split_file`: A string representing the file path to a CSV file containing split values.
- `ground_truth_metadata`: A string representing the file path to a feature metadata JSON file created during training.
- `output_feature_name`: A string representing the name of the output feature to visualize.
- `output_directory`: A string representing the name of the output directory containing training results.
- `kwargs`: Additional parameters for the requested visualizations.

The function first loads the feature metadata from the `ground_truth_metadata` file using the `load_json` function.

Then, it extracts the ground truth values from the source dataset using the `_extract_ground_truth_values` function, passing the `ground_truth`, `output_feature_name`, `ground_truth_split`, and `split_file` as arguments.

Next, it retrieves the probabilities per model using the `_get_cols_from_predictions` function, passing the `probabilities`, `[col]` (where `col` is the output feature name appended with the `_PROBABILITIES_SUFFIX`), and `metadata` as arguments.

Finally, it calls the `confidence_thresholding_data_vs_acc_subset` function, passing the `probabilities_per_model`, `ground_truth`, `metadata`, `output_feature_name`, `output_directory`, and `kwargs` as arguments for visualization.

The function does not return any value.

### **Function Details**
The given code defines a function `confidence_thresholding_data_vs_acc_subset_cli` that loads model data from files and visualizes the data using the `confidence_thresholding_data_vs_acc_subset` function.

The function takes the following inputs:
- `probabilities`: A string or a list of strings representing the file names of prediction results to extract probabilities from.
- `ground_truth`: A string representing the path to the ground truth file.
- `ground_truth_split`: A string representing the type of ground truth split. It can be `'0'` for the training split, `'1'` for the validation split, or `'2'` for the test split.
- `split_file`: A string representing the file path to a CSV file containing split values.
- `ground_truth_metadata`: A string representing the file path to a feature metadata JSON file created during training.
- `output_feature_name`: A string representing the name of the output feature to visualize.
- `output_directory`: A string representing the name of the output directory containing training results.
- `kwargs`: Additional parameters for the requested visualizations.

The function first loads the feature metadata from the `ground_truth_metadata` file using the `load_json` function. Then, it extracts the ground truth values from the `ground_truth` file using the `_extract_ground_truth_values` function. Next, it retrieves the probabilities per model from the `probabilities` files using the `_get_cols_from_predictions` function. Finally, it calls the `confidence_thresholding_data_vs_acc_subset` function to visualize the data, passing the probabilities per model, ground truth, metadata, output feature name, output directory, and any additional parameters.

The function does not return any value.

## Function **`confidence_thresholding_data_vs_acc_subset_per_class_cli`** Overview
The function `confidence_thresholding_data_vs_acc_subset_per_class_cli` is a command-line interface (CLI) function that is used to load model data from files and visualize the results using the `confidence_thresholding_data_vs_acc_subset_per_class` function. 

The function takes the following inputs:
- `probabilities`: A list of prediction results file names or a single file name.
- `ground_truth`: The path to the ground truth file.
- `ground_truth_metadata`: The path to the ground truth metadata file.
- `ground_truth_split`: The type of ground truth split, which can be 0 for the training split, 1 for the validation split, or 2 for the test split.
- `split_file`: The file path to a CSV file containing split values.
- `output_feature_name`: The name of the output feature to visualize.
- `output_directory`: The name of the output directory containing training results.
- `kwargs`: Additional parameters for the requested visualizations.

The function first loads the feature metadata from the ground truth metadata file. Then, it extracts the ground truth values based on the output feature name, ground truth split, and split file. 

Next, it retrieves the probabilities per model using the `_get_cols_from_predictions` function, passing the probabilities, the output feature name with a suffix, and the metadata. 

Finally, it calls the `confidence_thresholding_data_vs_acc_subset_per_class` function, passing the probabilities per model, ground truth, metadata, output feature name, output directory, and any additional parameters specified in `kwargs`.

### **Function Details**
The given code defines a function `confidence_thresholding_data_vs_acc_subset_per_class_cli` that takes several input parameters and returns `None`. 

The function is used to load model data from files and visualize the results using the `confidence_thresholding_data_vs_acc_subset_per_class` function. 

Here is a breakdown of the function's parameters:

- `probabilities`: A string or a list of strings representing the file names of prediction results to extract probabilities from.
- `ground_truth`: A string representing the path to the ground truth file.
- `ground_truth_metadata`: A string representing the path to the ground truth metadata file.
- `ground_truth_split`: An integer representing the type of ground truth split. It can be `0` for the training split, `1` for the validation split, or `2` for the test split.
- `split_file`: A string representing the file path to a CSV file containing split values.
- `output_feature_name`: A string representing the name of the output feature to visualize.
- `output_directory`: A string representing the name of the output directory containing training results.
- `**kwargs`: Additional keyword arguments for the requested visualizations.

The function first loads the metadata from the ground truth metadata file using the `load_json` function. Then, it extracts the ground truth values based on the provided parameters using the `_extract_ground_truth_values` function. 

Next, it retrieves the probabilities per model by calling the `_get_cols_from_predictions` function with the probabilities, the column name, and the metadata. 

Finally, it calls the `confidence_thresholding_data_vs_acc_subset_per_class` function with the extracted probabilities, ground truth, metadata, output feature name, output directory, and any additional keyword arguments.

## Function **`confidence_thresholding_2thresholds_2d_cli`** Overview
The function `confidence_thresholding_2thresholds_2d_cli` is a command-line interface function that loads model data from files and performs a visualization using the `confidence_thresholding_2thresholds_2d` function.

The function takes the following inputs:
- `probabilities`: A list of prediction results file names or a single file name as a string.
- `ground_truth`: The path to the ground truth file.
- `ground_truth_split`: An integer representing the type of ground truth split (0 for training split, 1 for validation split, or 2 for test split).
- `split_file`: The file path to a CSV file containing split values.
- `ground_truth_metadata`: The file path to a feature metadata JSON file created during training.
- `threshold_output_feature_names`: A list of names of the output features to visualize.
- `output_directory`: The name of the output directory containing training results.
- `kwargs`: Additional parameters for the requested visualizations.

The function first loads the feature metadata from the `ground_truth_metadata` file. Then, it extracts the ground truth values for the two output features specified in `threshold_output_feature_names` using the `_extract_ground_truth_values` function. Next, it retrieves the probabilities for each model from the `probabilities` files using the `_get_cols_from_predictions` function.

Finally, the function calls the `confidence_thresholding_2thresholds_2d` function with the retrieved data and parameters, and saves the visualization results in the specified `output_directory`.

The function does not return any value (`None`).

### **Function Details**
The given code defines a function `confidence_thresholding_2thresholds_2d_cli` that loads model data from files and performs a visualization using the `confidence_thresholding_2thresholds_2d` function.

The function takes the following inputs:
- `probabilities`: A string or a list of strings representing the file names of prediction results to extract probabilities from.
- `ground_truth`: A string representing the path to the ground truth file.
- `ground_truth_split`: A string representing the type of ground truth split. It can be `'0'` for the training split, `'1'` for the validation split, or `'2'` for the test split.
- `split_file`: A string representing the file path to a CSV file containing split values.
- `ground_truth_metadata`: A string representing the file path to a feature metadata JSON file created during training.
- `threshold_output_feature_names`: A list of strings representing the names of the output features to visualize.
- `output_directory`: A string representing the name of the output directory containing training results.
- `kwargs`: Additional parameters for the requested visualizations.

The function first loads the feature metadata from the `ground_truth_metadata` file using the `load_json` function.

Then, it extracts the ground truth values for the two output features specified in `threshold_output_feature_names` using the `_extract_ground_truth_values` function.

Next, it retrieves the probabilities for each model from the prediction files using the `_get_cols_from_predictions` function.

Finally, it calls the `confidence_thresholding_2thresholds_2d` function with the probabilities, ground truth values, metadata, and other parameters to perform the visualization.

The function does not return any value (`None`).

Note: The code references some helper functions (`load_json`, `_extract_ground_truth_values`, `_get_cols_from_predictions`) that are not provided in the given code snippet.

## Function **`confidence_thresholding_2thresholds_3d_cli`** Overview
The function `confidence_thresholding_2thresholds_3d_cli` is a command-line interface function that loads model data from files and performs visualization using the `confidence_thresholding_2thresholds_3d` function.

The function takes the following inputs:
- `probabilities`: A list of prediction results file names or a single file name.
- `ground_truth`: The path to the ground truth file.
- `ground_truth_split`: The type of ground truth split, which can be `0` for training split, `1` for validation split, or `2` for test split.
- `split_file`: The file path to a CSV file containing split values.
- `ground_truth_metadata`: The file path to a feature metadata JSON file created during training.
- `threshold_output_feature_names`: A list of names of the output features to visualize.
- `output_directory`: The name of the output directory containing training results.
- `kwargs`: Additional parameters for the requested visualizations.

The function first loads the feature metadata from the `ground_truth_metadata` file. Then, it extracts the ground truth values for the two specified output features using the `_extract_ground_truth_values` function. Next, it retrieves the probabilities per model using the `_get_cols_from_predictions` function, passing the probabilities, column names, and metadata as arguments. Finally, it calls the `confidence_thresholding_2thresholds_3d` function, passing the probabilities per model, ground truth values, metadata, output feature names, output directory, and additional parameters.

The function does not return any value (`None`).

### **Function Details**
The given code defines a function `confidence_thresholding_2thresholds_3d_cli` that loads model data from files and performs a visualization task using the loaded data.

The function takes the following inputs:
- `probabilities`: A string or a list of strings representing the file names of prediction results to extract probabilities from.
- `ground_truth`: A string representing the path to the ground truth file.
- `ground_truth_split`: A string representing the type of ground truth split. It can be `'0'` for the training split, `'1'` for the validation split, or `'2'` for the test split.
- `split_file`: A string representing the file path to a CSV file containing split values.
- `ground_truth_metadata`: A string representing the file path to a JSON file containing feature metadata created during training.
- `threshold_output_feature_names`: A list of strings representing the names of the output features to visualize.
- `output_directory`: A string representing the name of the output directory containing training results.
- `kwargs`: Additional parameters for the requested visualizations.

The function first loads the feature metadata from the `ground_truth_metadata` file using the `load_json` function.

Then, it extracts the ground truth values for the two output features specified in `threshold_output_feature_names` using the `_extract_ground_truth_values` function.

Next, it retrieves the probabilities for each model from the `probabilities` files using the `_get_cols_from_predictions` function.

Finally, it calls the `confidence_thresholding_2thresholds_3d` function to perform the visualization task using the loaded data and the specified parameters.

The function does not return any value.

## Function **`binary_threshold_vs_metric_cli`** Overview
The function `binary_threshold_vs_metric_cli` is a command-line interface function that loads model data from files and visualizes the binary threshold versus metric. 

It takes the following inputs:
- `probabilities`: A list of prediction results file names or a single file name to extract probabilities from.
- `ground_truth`: The path to the ground truth file.
- `ground_truth_split`: The type of ground truth split, where `0` represents the training split, `1` represents the validation split, and `2` represents the test split.
- `split_file`: The file path to a CSV file containing split values.
- `ground_truth_metadata`: The file path to a feature metadata JSON file created during training.
- `output_feature_name`: The name of the output feature to visualize.
- `output_directory`: The name of the output directory containing training results.
- `kwargs`: Additional parameters for the requested visualizations.

The function first loads the feature metadata from the `ground_truth_metadata` file. Then, it extracts the ground truth values using the `_extract_ground_truth_values` function, passing the `ground_truth`, `output_feature_name`, `ground_truth_split`, and `split_file` as arguments.

Next, it retrieves the probabilities per model using the `_get_cols_from_predictions` function, passing the `probabilities`, `[col]` (where `col` is the output feature name appended with `_PROBABILITIES_SUFFIX`), and `metadata` as arguments.

Finally, it calls the `binary_threshold_vs_metric` function, passing the `probabilities_per_model`, `ground_truth`, `metadata`, `output_feature_name`, `output_directory`, and `kwargs` as arguments to visualize the binary threshold versus metric.

The function does not return any value.

### **Function Details**
The given code defines a function `binary_threshold_vs_metric_cli` that loads model data from files and visualizes the binary threshold vs metric using the `binary_threshold_vs_metric` function.

The function takes the following parameters:
- `probabilities`: A string or a list of strings representing the file names of prediction results to extract probabilities from.
- `ground_truth`: A string representing the path to the ground truth file.
- `ground_truth_split`: A string representing the type of ground truth split. It can be `'0'` for the training split, `'1'` for the validation split, or `'2'` for the test split.
- `split_file`: A string representing the file path to a CSV file containing split values.
- `ground_truth_metadata`: A string representing the file path to a feature metadata JSON file created during training.
- `output_feature_name`: A string representing the name of the output feature to visualize.
- `output_directory`: A string representing the name of the output directory containing training results.
- `**kwargs`: Additional keyword arguments for the requested visualizations.

The function first loads the feature metadata from the `ground_truth_metadata` file using the `load_json` function. Then, it extracts the ground truth values from the `ground_truth` file using the `_extract_ground_truth_values` function. Next, it retrieves the probabilities per model from the prediction files using the `_get_cols_from_predictions` function.

Finally, it calls the `binary_threshold_vs_metric` function with the retrieved probabilities, ground truth, metadata, output feature name, output directory, and any additional keyword arguments.

The function does not return any value (`None`).

## Function **`precision_recall_curves_cli`** Overview
The function `precision_recall_curves_cli` is a command-line interface function that loads model data from files and generates precision-recall curves for evaluation.

The function takes the following arguments:
- `probabilities`: A list of file names or a single file name containing prediction results. These files are used to extract probabilities.
- `ground_truth`: The path to the ground truth file.
- `ground_truth_split`: The type of ground truth split, which can be `0` for the training split, `1` for the validation split, or `2` for the test split.
- `split_file`: The file path to a CSV file containing split values.
- `ground_truth_metadata`: The file path to a JSON file containing feature metadata created during training.
- `output_feature_name`: The name of the output feature to visualize.
- `output_directory`: The name of the output directory containing training results.
- `kwargs`: Additional parameters for the requested visualizations.

The function first loads the feature metadata from the `ground_truth_metadata` file. Then, it extracts the ground truth values using the `_extract_ground_truth_values` function. Next, it retrieves the probabilities per model using the `_get_cols_from_predictions` function. Finally, it calls the `precision_recall_curves` function to generate the precision-recall curves using the probabilities, ground truth, metadata, output feature name, and output directory.

The function does not return any value.

### **Function Details**
The given code defines a function `precision_recall_curves_cli` that loads model data from files and visualizes precision-recall curves.

The function takes the following arguments:
- `probabilities`: A string or a list of strings representing the file names of prediction results files to extract probabilities from.
- `ground_truth`: A string representing the path to the ground truth file.
- `ground_truth_split`: A string representing the type of ground truth split. It can be `'0'` for the training split, `'1'` for the validation split, or `'2'` for the test split.
- `split_file`: A string representing the file path to a CSV file containing split values.
- `ground_truth_metadata`: A string representing the file path to a feature metadata JSON file created during training.
- `output_feature_name`: A string representing the name of the output feature to visualize.
- `output_directory`: A string representing the name of the output directory containing training results.
- `**kwargs`: Additional keyword arguments for the requested visualizations.

The function first loads the feature metadata from the `ground_truth_metadata` file using the `load_json` function. Then, it extracts the ground truth values from the `ground_truth` file using the `_extract_ground_truth_values` function. Next, it retrieves the probabilities per model from the prediction files using the `_get_cols_from_predictions` function.

Finally, it calls the `precision_recall_curves` function with the probabilities per model, ground truth values, metadata, output feature name, output directory, and any additional keyword arguments.

The function does not return any value (`None`).

## Function **`roc_curves_cli`** Overview
The function `roc_curves_cli` is a command-line interface function that is used to load model data from files and generate ROC curves for visualization.

The function takes the following inputs:
- `probabilities`: A list of prediction results file names or a single file name as a string.
- `ground_truth`: The path to the ground truth file.
- `ground_truth_split`: The type of ground truth split, which can be `0` for the training split, `1` for the validation split, or `2` for the test split.
- `split_file`: The file path to a CSV file containing split values.
- `ground_truth_metadata`: The file path to a feature metadata JSON file created during training.
- `output_feature_name`: The name of the output feature to visualize.
- `output_directory`: The name of the output directory containing training results.
- `kwargs`: Additional parameters for the requested visualizations.

The function first loads the feature metadata from the `ground_truth_metadata` file. Then, it extracts the ground truth values using the `_extract_ground_truth_values` function, passing the `ground_truth`, `output_feature_name`, `ground_truth_split`, and `split_file` as arguments.

Next, the function retrieves the probabilities per model using the `_get_cols_from_predictions` function, passing the `probabilities`, `[col]` (where `col` is the output feature name appended with `_PROBABILITIES_SUFFIX`), and `metadata` as arguments.

Finally, the function calls the `roc_curves` function, passing the `probabilities_per_model`, `ground_truth`, `metadata`, `output_feature_name`, `output_directory`, and any additional `kwargs` as arguments to generate the ROC curves for visualization.

The function does not return any value (`None`).

### **Function Details**
The given code defines a function `roc_curves_cli` that loads model data from files and visualizes ROC curves.

The function takes the following inputs:
- `probabilities`: A string or a list of strings representing the file names of prediction results files to extract probabilities from.
- `ground_truth`: A string representing the path to the ground truth file.
- `ground_truth_split`: A string representing the type of ground truth split. It can be `'0'` for the training split, `'1'` for the validation split, or `'2'` for the test split.
- `split_file`: A string representing the file path to a CSV file containing split values.
- `ground_truth_metadata`: A string representing the file path to a feature metadata JSON file created during training.
- `output_feature_name`: A string representing the name of the output feature to visualize.
- `output_directory`: A string representing the name of the output directory containing training results.
- `**kwargs`: Additional keyword arguments for the requested visualizations.

The function first loads the feature metadata from the `ground_truth_metadata` file using the `load_json` function. Then, it extracts the ground truth values from the `ground_truth` file using the `_extract_ground_truth_values` function. Next, it retrieves the probabilities per model from the prediction files using the `_get_cols_from_predictions` function. Finally, it calls the `roc_curves` function to visualize the ROC curves using the probabilities, ground truth, metadata, output feature name, output directory, and any additional keyword arguments.

The function does not return any value.

## Function **`roc_curves_from_test_statistics_cli`** Overview
The function `roc_curves_from_test_statistics_cli` is a command-line interface (CLI) function that takes in test statistics and additional parameters as input. It loads model data from files and then calls the `roc_curves_from_test_statistics` function to generate ROC curves based on the test statistics.

The function takes the following parameters:
- `test_statistics`: A path to the experiment test statistics file or a list of paths to multiple test statistics files.
- `kwargs`: Additional parameters for the requested visualizations.

The function returns `None`.

### **Function Details**
The given code defines a function `roc_curves_from_test_statistics_cli` that takes in two parameters: `test_statistics` and `kwargs`. 

The `test_statistics` parameter can be either a string or a list of strings representing the path(s) to experiment test statistics file(s). 

The `kwargs` parameter is a dictionary that contains additional parameters for the requested visualizations.

Inside the function, the `load_data_for_viz` function is called to load the data from the test statistics file(s) using the "load_json" method. The loaded data is then passed to the `roc_curves_from_test_statistics` function along with the additional parameters from `kwargs`.

The function does not return anything (`None`).

Note: The code assumes that the necessary imports and definitions for the `load_data_for_viz` and `roc_curves_from_test_statistics` functions are present.

## Function **`precision_recall_curves_from_test_statistics_cli`** Overview
The function `precision_recall_curves_from_test_statistics_cli` is a command-line interface (CLI) function that takes in test statistics data and additional parameters as input. It loads the model data from files and then calls the `precision_recall_curves_from_test_statistics` function to generate precision-recall curves based on the test statistics.

The function takes two parameters:
- `test_statistics`: A string or a list of strings representing the path(s) to the experiment test statistics file(s).
- `kwargs`: A dictionary containing additional parameters for the visualization.

The function returns `None` and does not have any side effects.

### **Function Details**
The given code defines a function `precision_recall_curves_from_test_statistics_cli` that takes in a parameter `test_statistics` (which can be either a string or a list of strings) and additional keyword arguments `kwargs`. 

The function first calls the `load_data_for_viz` function to load the test statistics data from the specified file(s) using the "load_json" method. The loaded data is then stored in the `test_stats_per_model` variable.

Finally, the function calls the `precision_recall_curves_from_test_statistics` function, passing in the `test_stats_per_model` variable and the additional keyword arguments.

The function does not return any value, as indicated by the `None` return type.

## Function **`calibration_1_vs_all_cli`** Overview
The function `calibration_1_vs_all_cli` is used to load model data from files and visualize the calibration of a binary classification model using the 1-vs-all approach. 

Here is a general description of what the function does:

1. It takes several input parameters including the file names of prediction results, the path to the ground truth file, the type of ground truth split, the file path to a csv file containing split values, the file path to feature metadata json file, the name of the output feature to visualize, the name of the output directory containing training results, and optional parameters for visualization.

2. It loads the feature metadata from the ground truth metadata file.

3. It extracts the ground truth values based on the specified split type and split file.

4. It vectorizes the ground truth values using the feature metadata.

5. It retrieves the probabilities per model from the prediction results files.

6. It calls the `calibration_1_vs_all` function to visualize the calibration using the probabilities per model, ground truth values, metadata, output feature name, and output directory.

7. The function does not return any value.

### **Function Details**
The given code defines a function called `calibration_1_vs_all_cli` which is used to load model data from files and visualize the calibration of a binary classification model.

The function takes the following parameters:
- `probabilities`: A string or a list of strings representing the file names of prediction results files to extract probabilities from.
- `ground_truth`: A string representing the path to the ground truth file.
- `ground_truth_split`: A string representing the type of ground truth split. It can be `'0'` for the training split, `'1'` for the validation split, or `'2'` for the test split.
- `split_file`: A string representing the file path to a CSV file containing split values.
- `ground_truth_metadata`: A string representing the file path to a feature metadata JSON file created during training.
- `output_feature_name`: A string representing the name of the output feature to visualize.
- `output_directory`: A string representing the name of the output directory containing training results.
- `output_feature_proc_name`: An optional string representing the name of the output feature column in the ground truth. If the ground truth is a preprocessed Parquet or HDF5 file, the column name will be `<output_feature>_<hash>`.
- `ground_truth_apply_idx`: A boolean indicating whether to use metadata['str2idx'] in np.vectorize.
- `**kwargs`: Additional parameters for the requested visualizations.

The function first loads the feature metadata from the ground truth metadata file. Then, it extracts the ground truth values based on the specified split and feature name. The ground truth values are vectorized using the metadata['str2idx'] mapping. 

Next, it retrieves the probabilities per model from the prediction files using the specified column name and metadata. Finally, it calls the `calibration_1_vs_all` function to visualize the calibration using the probabilities, ground truth, metadata, output feature name, and output directory.

The function does not return any value.

## Function **`calibration_multiclass_cli`** Overview
The function `calibration_multiclass_cli` is a command-line interface function that loads model data from files and performs multiclass calibration. 

It takes the following inputs:
- `probabilities`: A list of prediction results file names or a single file name as a string.
- `ground_truth`: The path to the ground truth file.
- `ground_truth_split`: The type of ground truth split, which can be `0` for the training split, `1` for the validation split, or `2` for the test split.
- `split_file`: The file path to a CSV file containing split values. This parameter is optional and can be set to `None`.
- `ground_truth_metadata`: The file path to a feature metadata JSON file created during training.
- `output_feature_name`: The name of the output feature to visualize.
- `output_directory`: The name of the output directory containing training results.
- `kwargs`: Additional parameters for the requested visualizations.

The function first loads the feature metadata from the `ground_truth_metadata` file using the `load_json` function.

Then, it extracts the ground truth values from the source dataset using the `_extract_ground_truth_values` function, passing the `ground_truth`, `output_feature_name`, `ground_truth_split`, and `split_file` parameters.

Next, it retrieves the probabilities per model by calling the `_get_cols_from_predictions` function, passing the `probabilities`, `[col]` (where `col` is the output feature name appended with a suffix), and the metadata.

Finally, it calls the `calibration_multiclass` function, passing the `probabilities_per_model`, `ground_truth`, `metadata`, `output_feature_name`, `output_directory`, and any additional parameters specified in `kwargs`.

The function does not return any value (`None`).

### **Function Details**
The given code defines a function `calibration_multiclass_cli` that loads model data from files and performs multiclass calibration. 

The function takes the following inputs:
- `probabilities`: A string or list of strings representing the file names of prediction results files to extract probabilities from.
- `ground_truth`: A string representing the path to the ground truth file.
- `ground_truth_split`: An integer representing the type of ground truth split. `0` for training split, `1` for validation split, or `2` for test split.
- `split_file`: A string representing the file path to a CSV file containing split values.
- `ground_truth_metadata`: A string representing the file path to a feature metadata JSON file created during training.
- `output_feature_name`: A string representing the name of the output feature to visualize.
- `output_directory`: A string representing the name of the output directory containing training results.
- `**kwargs`: Additional parameters for the requested visualizations.

The function first loads the feature metadata from the `ground_truth_metadata` file using the `load_json` function. Then, it extracts the ground truth values from the `ground_truth` file using the `_extract_ground_truth_values` function. 

Next, it retrieves the probabilities per model from the prediction files using the `_get_cols_from_predictions` function. The probabilities are extracted for the specified `output_feature_name` and stored in the `probabilities_per_model` variable.

Finally, the `calibration_multiclass` function is called with the extracted probabilities, ground truth, metadata, output feature name, output directory, and any additional parameters specified in `kwargs`.

The function does not return any value (`None`).

## Function **`confusion_matrix_cli`** Overview
The function `confusion_matrix_cli` is a command-line interface (CLI) function that is used to load model data from files and display a confusion matrix. 

The function takes three parameters:
- `test_statistics`: A path to the experiment test statistics file. It can be either a string representing a single file path or a list of strings representing multiple file paths.
- `ground_truth_metadata`: A path to the ground truth metadata file.
- `kwargs`: Additional parameters for the requested visualizations. These parameters are passed as keyword arguments.

The function first loads the model data from the test statistics file(s) using the `load_data_for_viz` function with the "load_json" method. It then loads the ground truth metadata using the `load_json` function.

Finally, the function calls the `confusion_matrix` function, passing the loaded test statistics data, metadata, and any additional parameters specified in `kwargs`. The confusion matrix is then displayed.

The function does not return any value (`None`).

### **Function Details**
The given code defines a function `confusion_matrix_cli` that takes three parameters: `test_statistics`, `ground_truth_metadata`, and `kwargs`. 

The `test_statistics` parameter can be either a string or a list of strings representing the path(s) to the experiment test statistics file(s). 

The `ground_truth_metadata` parameter is a string representing the path to the ground truth metadata file.

The `kwargs` parameter is a dictionary that can contain additional parameters for the requested visualizations.

Inside the function, the `load_data_for_viz` function is called with the "load_json" argument and the `test_statistics` parameter to load the test statistics data for visualization. The result is stored in the `test_stats_per_model` variable.

The `load_json` function is called with the `ground_truth_metadata` parameter to load the ground truth metadata. The result is stored in the `metadata` variable.

Finally, the `confusion_matrix` function is called with the `test_stats_per_model`, `metadata`, and `kwargs` parameters to display the confusion matrix visualization.

The function does not return any value (`None`).

## Function **`frequency_vs_f1_cli`** Overview
The function `frequency_vs_f1_cli` is a command-line interface (CLI) function that loads model data from files and calls the `frequency_vs_f1` function to generate visualizations.

The function takes three parameters:
- `test_statistics`: A path to the experiment test statistics file. It can be either a string representing a single file path or a list of strings representing multiple file paths.
- `ground_truth_metadata`: A path to the ground truth metadata file.
- `kwargs`: Additional parameters for the requested visualizations. These parameters are passed as keyword arguments.

The function first loads the data from the test statistics file(s) using the `load_data_for_viz` function with the "load_json" method. It then loads the ground truth metadata using the `load_json` function.

Finally, the function calls the `frequency_vs_f1` function, passing the loaded test statistics data, metadata, and any additional parameters specified in `kwargs`. The `frequency_vs_f1` function is responsible for generating the visualizations based on the provided data.

The function does not return any value (`None`).

### **Function Details**
The given code defines a function `frequency_vs_f1_cli` that takes in three parameters: `test_statistics`, `ground_truth_metadata`, and `kwargs`. 

The `test_statistics` parameter can be either a string or a list of strings representing the path(s) to the experiment test statistics file(s). 

The `ground_truth_metadata` parameter is a string representing the path to the ground truth metadata file.

The `kwargs` parameter is a dictionary that can contain additional parameters for the requested visualizations.

Inside the function, the `load_data_for_viz` function is called with the "load_json" argument and the `test_statistics` parameter to load the test statistics data for visualization. The result is stored in the `test_stats_per_model` variable.

The `load_json` function is called with the `ground_truth_metadata` parameter to load the ground truth metadata. The result is stored in the `metadata` variable.

Finally, the `frequency_vs_f1` function is called with the `test_stats_per_model`, `metadata`, and `kwargs` parameters to perform the visualization.

The function does not return any value (`None`).

## Function **`learning_curves`** Overview
The `learning_curves` function takes in several parameters including `train_stats_per_model`, `output_feature_name`, `model_names`, `output_directory`, `file_format`, and `callbacks`. 

The function generates line plots to show how model metrics change over the course of training and validation data epochs. It does this for each model and for each output feature and metric of the model. 

The function first validates the input parameters and sets up the filename template for saving the plots. It then iterates over the output feature names and metrics to generate the learning curves plot for each combination. 

For each combination, the function extracts the training and validation statistics from the `train_stats_per_model` list. It also determines the filename for saving the plot if an output directory is specified. 

Finally, the function calls the `learning_curves_plot` function from the `visualization_utils` module to generate the line plot. The plot includes the training and validation statistics, the metric being plotted, the x-axis label and step size, the model names, and a title. The plot can be saved to a file if an output directory is specified.

### **Function Details**
The code provided is a function called `learning_curves` that generates line plots showing how model metrics change over training and validation data epochs. 

The function takes the following inputs:
- `train_stats_per_model`: a list containing dictionaries of training statistics per model.
- `output_feature_name`: the name of the output feature to use for the visualization. If `None`, all output features are used.
- `model_names`: the model name or a list of model names to use as labels.
- `output_directory`: the directory where to save the plots. If not specified, the plots will be displayed in a window.
- `file_format`: the file format of the output plots, either `'pdf'` or `'png'`.
- `callbacks`: a list of `ludwig.callbacks.Callback` objects that provide hooks into the Ludwig pipeline.

The function does not return anything (`None`).

The function first defines a filename template for the output plots based on the output feature name and metric. It then converts the `train_stats_per_model` and `model_names` inputs into lists if they are not already. It also validates the output feature name from the training statistics.

The function then defines a list of metrics to include in the plots. For each output feature name and metric, the function checks if the metric is present in the training statistics of the first model. If it is, the function retrieves the training and validation statistics for that metric from each model in the `train_stats_per_model` list.

The function then calls the `visualization_utils.learning_curves_plot` function to generate the line plot, passing in the training and validation statistics, metric, x-axis label, x-axis step, model names, title, filename, and callbacks.

Overall, the function provides a convenient way to visualize the learning curves of different models and metrics over training and validation epochs.

## Function **`compare_performance`** Overview
The function `compare_performance` takes in several parameters including `test_stats_per_model`, `output_feature_name`, `model_names`, `output_directory`, and `file_format`. 

It produces a bar plot visualization for each overall metric in the `test_stats_per_model` dictionary. The bar plot compares the performance of each model specified in the `model_names` list. The `output_feature_name` parameter specifies the output feature to use for the visualization. If `output_feature_name` is `None`, all output features are used.

The resulting bar plot is either displayed in a window or saved in the specified `output_directory` as a file in the specified `file_format` (default is "pdf").

The function first checks for any metrics to compare. It then creates a dictionary `metrics_dict` to store the metric values for each model and metric name. It removes any ignored metric names specified in the `ignore_names` list.

The function then iterates over the `test_stats_per_model` list and adds the metric values to the `metrics_dict` dictionary.

If there are metrics to compare, the function determines the minimum and maximum values among the metric values. It then generates a filename for the plot based on the `output_feature_name` and saves the plot using the `visualization_utils.compare_classifiers_plot` function.

The function does not return any value.

### **Function Details**
The code defines a function `compare_performance` that takes in several parameters and produces a bar plot visualization for model comparison based on evaluation performance statistics.

The function takes the following parameters:
- `test_stats_per_model`: A list of dictionaries containing evaluation performance statistics for each model.
- `output_feature_name`: The name of the output feature to use for the visualization. If `None`, all output features are used.
- `model_names`: The name or list of names of the models to use as labels.
- `output_directory`: The directory where the plots will be saved. If not specified, the plots will be displayed in a window.
- `file_format`: The file format of the output plots, either `'pdf'` or `'png'`.
- `**kwargs`: Additional keyword arguments.

The function returns `None`.

The function first defines a list of names to ignore in the evaluation statistics. Then, it generates a filename template for saving the plots based on the output directory and file format.

Next, it converts the input `test_stats_per_model` and `model_names` to lists if they are not already. It also validates the output feature name from the test statistics.

The function then iterates over each output feature name and performs the following steps:
- It collects the metric names available in the evaluation statistics for the current output feature name.
- It removes the ignored metric names from the collected metric names.
- It creates an empty dictionary `metrics_dict` to store the metric values for each model.
- It iterates over each model's evaluation statistics and collects the metric values for each metric name.
- It checks if there are any metrics to compare.
- If there are metrics, it creates a list `metrics` to store the metric values and a list `metrics_names` to store the metric names.
- It finds the minimum and maximum metric values across all models.
- It generates a filename for saving the plot if the output directory is specified.
- Finally, it calls the `compare_classifiers_plot` function from the `visualization_utils` module to create the bar plot visualization, passing in the collected metrics, metric names, model names, and other parameters.

The function can be used by providing the evaluation statistics for each model and the corresponding model names. The resulting bar plot will show the comparison of metrics for each model.

## Function **`compare_classifiers_performance_from_prob`** Overview
The function `compare_classifiers_performance_from_prob` takes in probabilities per model, ground truth values, metadata, output feature name, and other optional parameters. It produces a bar plot visualization comparing the performance of different models based on overall metrics computed from the probabilities of predictions.

The function first checks if the ground truth values are not a numpy array and if so, it translates the raw values to encoded values using the metadata. It then converts the `top_n_classes` and `model_names` parameters into lists if they are not already. If `labels_limit` is greater than 0, it limits the ground truth values to be less than or equal to `labels_limit`.

The function then initializes empty lists for accuracies, hits_at_ks, and mrrs. It iterates over the probabilities for each model and performs calculations to compute the accuracy, hits_at_k, and mrr for each model. These metrics are appended to the respective lists.

If an `output_directory` is specified, the function creates the directory if it doesn't exist and sets the `filename` variable to the path of the output file. 

Finally, the function calls `visualization_utils.compare_classifiers_plot` to generate the bar plot visualization using the computed metrics, model names, and the output file path.

### **Function Details**
The given code defines a function called `compare_classifiers_performance_from_prob` that compares the performance of different classifiers using probabilities. 

The function takes the following inputs:
- `probabilities_per_model`: A list of numpy arrays containing the probabilities predicted by each model.
- `ground_truth`: The ground truth values.
- `metadata`: A dictionary containing feature metadata.
- `output_feature_name`: The name of the output feature.
- `top_n_classes`: A list or integer specifying the number of classes to plot.
- `labels_limit`: An integer specifying the upper limit on the numeric encoded label value.
- `model_names`: A string or list of strings specifying the names of the models to use as labels.
- `output_directory`: A string specifying the directory where to save the plots.
- `file_format`: A string specifying the file format of the output plots.
- `ground_truth_apply_idx`: A boolean indicating whether to use metadata['str2idx'] in np.vectorize.

The function produces a bar plot visualization for each model, with bars representing different overall metrics computed from the probabilities. The metrics include accuracy, hits at k, and mean reciprocal rank (MRR).

The function returns None.

Note: The code references a function `visualization_utils.compare_classifiers_plot`, which is not provided in the given code.

## Function **`compare_classifiers_performance_from_pred`** Overview
The function `compare_classifiers_performance_from_pred` takes in predictions from multiple classifiers, the ground truth values, and other parameters, and produces a bar plot visualization comparing the performance of the classifiers based on various metrics.

The function first checks if the ground truth values are not a numpy array, in which case it assumes that the values need to be translated to encoded values based on the feature metadata. It then flattens and converts the predictions from each model to numpy arrays.

If a limit on the numeric encoded label value is specified, any ground truth values higher than the limit are set to the limit.

The function then calculates the accuracy, precision, recall, and F1 score for each model's predictions using the sklearn.metrics module.

If an output directory is specified, the function creates the directory if it doesn't exist and saves the visualization as a file in the specified format. Otherwise, the visualization is displayed in a window.

The resulting bar plot visualization compares the accuracies, precisions, recalls, and F1 scores of the models, with the model names as labels on the x-axis.

### **Function Details**
The given code defines a function `compare_classifiers_performance_from_pred` that compares the performance of different classifiers based on their predictions. 

The function takes the following inputs:
- `predictions_per_model`: A list of numpy arrays containing the predictions made by each model.
- `ground_truth`: The ground truth values.
- `metadata`: A dictionary containing feature metadata.
- `output_feature_name`: The name of the output feature to visualize.
- `labels_limit`: An upper limit on the numeric encoded label value.
- `model_names`: The name or list of names of the models to use as labels.
- `output_directory`: The directory where to save the plots.
- `file_format`: The file format of the output plots.
- `ground_truth_apply_idx`: A boolean indicating whether to use metadata['str2idx'] in np.vectorize.

The function produces a bar plot visualization for each model, with bars representing different overall metrics computed from the predictions. The metrics include accuracy, precision, recall, and F1 score. The bar plot is created using the `compare_classifiers_plot` function from the `visualization_utils` module.

The function also handles some preprocessing steps such as flattening the prediction arrays, mapping label values using metadata, and applying an upper limit to label values.

If the `output_directory` is specified, the plots are saved in that directory with the specified `file_format`. Otherwise, the plots are displayed in a window.

The function does not return any value.

## Function **`compare_classifiers_performance_subset`** Overview
The function `compare_classifiers_performance_subset` takes in several inputs including a list of model probabilities, ground truth values, metadata, output feature name, top N classes, labels limit, subset type, model names, output directory, file format, and a boolean flag. 

The function produces a bar plot visualization that compares the performance of different models. For each model, it computes overall metrics based on the probabilities predictions for the specified `model_names`, considering only a subset of the full training set. The subset is obtained using the `top_n_classes` and `subset` parameters.

The function first checks if the ground truth values are not a numpy array and if so, it translates the raw values to encoded values using the metadata. It then converts the `top_n_classes` and `model_names` inputs into lists and applies a label limit if specified.

Next, it determines the subset indices based on the subset type. If the subset type is "ground_truth", it selects the subset where the ground truth values are less than `k`. If the subset type is "predictions", it selects the subset where the argmax of the probabilities is less than `k`. It also updates the model names to include the percentage of the subset.

The function then processes the probabilities for each model. If a label limit is specified and the number of classes in the probabilities is greater than the label limit, it limits the probabilities to the label limit and sums the remaining probabilities into the last class. It then calculates the top-1 and top-3 predictions for the subset.

The function computes the accuracies and hits@k metrics for each model based on the subset. It then generates a title for the plot based on the subset type and creates a filename if an output directory is specified.

Finally, it calls the `compare_classifiers_plot` function from the `visualization_utils` module to generate the bar plot visualization, using the accuracies, hits@k, model names, title, and filename as inputs.

### **Function Details**
The code defines a function called `compare_classifiers_performance_subset` that takes several input parameters and produces a model comparison barplot visualization. 

The function takes the following parameters:
- `probabilities_per_model`: a list of numpy arrays representing the probabilities predicted by each model.
- `ground_truth`: the ground truth values.
- `metadata`: a dictionary containing feature metadata.
- `output_feature_name`: the name of the output feature.
- `top_n_classes`: a list containing the number of classes to plot.
- `labels_limit`: an upper limit on the numeric encoded label value.
- `subset`: a string specifying the type of subset filtering.
- `model_names`: the name or list of names of the models to use as labels.
- `output_directory`: the directory where to save the plots.
- `file_format`: the file format of the output plots.
- `ground_truth_apply_idx`: a boolean indicating whether to use metadata['str2idx'] in np.vectorize.

The function then performs several operations to compute the model performance metrics and generate the barplot visualization. It calculates the accuracies and hits@k metrics for each model, based on the probabilities and ground truth values. It also applies subset filtering based on the specified parameters.

Finally, the function calls another function called `compare_classifiers_plot` from a module called `visualization_utils` to generate the barplot visualization. The visualization includes the accuracies and hits@k metrics for each model, with the model names as labels. The title and filename for the visualization are also specified.

The function does not return any value, it only produces the visualization.

## Function **`compare_classifiers_performance_changing_k`** Overview
The function `compare_classifiers_performance_changing_k` takes in several inputs including a list of model probabilities, ground truth values, metadata, output feature name, top_k value, labels_limit value, model names, output directory, file format, and ground_truth_apply_idx. 

The function produces a line plot that shows the Hits@K metric while changing k from 1 to `top_k` for each model. The Hits@K metric counts a prediction as correct if the model produces it among the first k predictions. 

The function first checks if the ground truth values are not a numpy array and if so, it translates the raw values to encoded values using the feature metadata. 

Next, it sets the value of k to top_k and applies a limit to the ground truth labels if labels_limit is greater than 0. 

Then, it calculates the hits_at_k metric for each model by iterating over the probabilities and ground truth values. It sorts the probabilities, calculates the hits_at_k values, and appends them to a list. 

Finally, it creates a line plot using the hits_at_k values, model names, and other parameters. The plot can be saved to an output directory if specified, or displayed in a window.

### **Function Details**
The code defines a function called `compare_classifiers_performance_changing_k` that takes several input parameters and produces a line plot to compare the performance of different classifiers.

The function takes the following input parameters:
- `probabilities_per_model`: a list of numpy arrays representing the probabilities predicted by each model.
- `ground_truth`: the ground truth values.
- `metadata`: a dictionary containing feature metadata.
- `output_feature_name`: the name of the output feature.
- `top_k`: the number of elements in the ranklist to consider.
- `labels_limit`: an upper limit on the numeric encoded label value.
- `model_names`: the name or list of names of the models to use as labels.
- `output_directory`: the directory where to save the plots.
- `file_format`: the file format of the output plots.
- `ground_truth_apply_idx`: a boolean indicating whether to use metadata['str2idx'] in np.vectorize.

The function first checks if the ground truth values are not a numpy array and if so, it translates the raw values to encoded values using the feature metadata.

Next, it sets the value of `k` to `top_k` and applies a limit to the ground truth values if `labels_limit` is greater than 0.

Then, it calculates the hits@k metric for each model by sorting the probabilities and counting the number of correct predictions among the top k predictions.

Finally, it calls a visualization utility function to create the line plot comparing the hits@k metric for each model.

The resulting plot can be saved to a file if `output_directory` is specified, otherwise it will be displayed in a window.

The function does not return any value.

## Function **`compare_classifiers_multiclass_multimetric`** Overview
The function `compare_classifiers_multiclass_multimetric` takes in several inputs including a list of evaluation performance statistics for different models, metadata, output feature name, top N classes, model names, output directory, and file format. 

The function generates four plots for each model, showing the precision, recall, and F1 score of the model on several classes for the specified output feature. The plots are generated using the `compare_classifiers_multiclass_multimetric_plot` function from the `visualization_utils` module.

The function iterates over the test statistics for each model and output feature, retrieves the per-class statistics, and extracts the precision, recall, F1 score, and labels for each class. It then generates plots for the top N classes, the best N classes based on F1 score, the worst N classes based on F1 score, and all classes sorted by F1 score.

The plots are either displayed in a window or saved in the specified output directory in the specified file format. The function also logs information about the model, the top and worst classes based on F1 score, and the number of classes with F1 score greater than 0 and equal to 0.

### **Function Details**
This is a function named `compare_classifiers_multiclass_multimetric` that compares the performance of multiple classifiers on a multiclass classification task. 

The function takes the following inputs:
- `test_stats_per_model`: A list of dictionaries containing evaluation performance statistics for each model.
- `metadata`: A dictionary containing intermediate preprocess structure created during training, which includes mappings of the input dataset.
- `output_feature_name`: The name of the output feature to use for the visualization.
- `top_n_classes`: A list of integers specifying the number of classes to plot.
- `model_names`: A string or a list of strings representing the names of the models to use as labels (optional).
- `output_directory`: The directory where to save the plots (optional).
- `file_format`: The file format of the output plots, either "pdf" or "png" (optional).

The function produces four plots for each model, showing the precision, recall, and F1 score of the model on several classes for the specified output feature. The plots are saved in the specified output directory or displayed in a window if no directory is specified.

The function returns `None`.

Note: The code provided is incomplete and requires additional functions and imports to run properly.

## Function **`compare_classifiers_predictions`** Overview
The function `compare_classifiers_predictions` compares the predictions of two models for a specified output feature. It takes in the following inputs:
- `predictions_per_model`: A list containing the predictions of each model.
- `ground_truth`: The ground truth values for the output feature.
- `metadata`: A dictionary containing metadata for the features.
- `output_feature_name`: The name of the output feature.
- `labels_limit`: An upper limit on the numeric encoded label value.
- `model_names`: The names of the models to use as labels.
- `output_directory`: The directory where to save the plots.
- `file_format`: The file format of the output plots.
- `ground_truth_apply_idx`: Whether to use metadata['str2idx'] in np.vectorize.

The function first checks if the ground truth values need to be translated to encoded values. It then extracts the names of the models and the predictions for each model. If a labels limit is specified, it applies it to the ground truth and predictions.

The function then calculates various metrics by comparing the ground truth and predictions. These metrics include the number and percentage of datapoints where both models are right, one model is right, both models are wrong, and the predictions are the same or different.

The function logs the calculated metrics and saves a donut plot visualizing the comparison between the models' predictions. The plot shows the number of datapoints where both models are right, one model is right, and both models are wrong. It also shows the breakdown of both wrong predictions into cases where the predictions are the same or different.

If an output directory is specified, the plot is saved in that directory with a filename based on the names of the models.

The function does not return any value.

### **Function Details**
The given code defines a function `compare_classifiers_predictions` that compares the predictions of two classifiers for a specified output feature. 

The function takes the following inputs:
- `predictions_per_model`: A list containing the model predictions for the specified output feature.
- `ground_truth`: The ground truth values for the output feature.
- `metadata`: A dictionary containing feature metadata.
- `output_feature_name`: The name of the output feature.
- `labels_limit`: An upper limit on the numeric encoded label value. Labels higher than this limit are considered "rare" labels.
- `model_names`: The name or list of names of the models to use as labels.
- `output_directory`: The directory where to save the plots. If not specified, the plots will be displayed in a window.
- `file_format`: The file format of the output plots (default is "pdf").
- `ground_truth_apply_idx`: A boolean indicating whether to use the metadata's "str2idx" mapping in np.vectorize.

The function compares the predictions of the two models and calculates various metrics such as the number of datapoints where both models are right, one model is right, both models are wrong, etc. It also logs these metrics and generates a donut plot to visualize the comparison.

The function returns None.

Note: The code references some external functions and libraries such as `convert_to_list`, `logger`, `os`, and `visualization_utils`. These functions and libraries are not defined in the given code snippet and may be part of a larger codebase.

## Function **`compare_classifiers_predictions_distribution`** Overview
The function `compare_classifiers_predictions_distribution` takes in several inputs including a list of model predictions, ground truth values, metadata, output feature name, labels limit, model names, output directory, file format, and other optional arguments. 

The function produces a radar plot that compares the distributions of predictions from different models for the first 10 classes of the specified output feature. 

The function first checks if the ground truth values are a numpy array, and if not, it assumes that the raw values need to be translated to encoded values using the metadata. 

Next, the function converts the model names to a list and applies a label limit to the ground truth and predictions if specified. 

The function then calculates the maximum ground truth value and maximum prediction value among all models. 

Using the ground truth and predictions, the function calculates the counts and probabilities for each class. 

If an output directory is specified, the function creates the directory if it doesn't exist and saves the radar plot in the specified file format. 

Finally, the function calls a visualization utility function to create the radar chart using the calculated probabilities, model names, and the optional filename.

### **Function Details**
This is a function that compares the distribution of predictions from multiple classifiers for a specified output feature. It produces a radar plot to visualize the distributions.

The function takes the following inputs:
- `predictions_per_model`: A list containing the model predictions for the specified output feature.
- `ground_truth`: The ground truth values.
- `metadata`: A dictionary containing feature metadata.
- `output_feature_name`: The name of the output feature.
- `labels_limit`: An upper limit on the numeric encoded label value. Labels higher than this limit are considered "rare" labels.
- `model_names`: The name or list of names of the models to use as labels.
- `output_directory`: The directory where to save the plots. If not specified, plots will be displayed in a window.
- `file_format`: The file format of the output plots (either "pdf" or "png").
- `ground_truth_apply_idx`: Whether to use metadata['str2idx'] in np.vectorize.

The function returns `None`.

The function first checks if the ground truth values are a numpy array. If not, it assumes that the values need to be translated to encoded values using the metadata. It then converts the model names to a list if necessary.

If `labels_limit` is greater than 0, it applies the limit to the ground truth values and the predictions.

Next, it calculates the maximum values for the ground truth and predictions, and adds 1 to get the maximum value for the radar plot.

It then calculates the counts and probabilities for the ground truth and predictions.

If an `output_directory` is specified, it creates the directory if it doesn't exist and sets the filename for the radar plot.

Finally, it calls a visualization utility function to create the radar chart using the probabilities and model names, and saves the plot if an `output_directory` is specified.

## Function **`confidence_thresholding`** Overview
The function `confidence_thresholding` takes in several inputs including a list of model probabilities, ground truth values, metadata, output feature name, labels limit, model names, output directory, file format, and other optional arguments. 

The function calculates the accuracy and data coverage for each model while increasing a threshold on the probabilities of predictions for the specified output feature. It does this by iterating over a range of thresholds and filtering the predictions based on the threshold. It then calculates the accuracy and data coverage for each filtered subset of predictions.

The function also handles cases where the ground truth values need to be translated to encoded values and applies a limit to the numeric encoded label values. It saves the resulting accuracy and data coverage plots to a specified output directory or displays them in a window if no output directory is specified.

### **Function Details**
The code defines a function called `confidence_thresholding` that takes in several input parameters and returns nothing (`None`). 

The purpose of the function is to show the accuracy and data coverage of multiple models while increasing a threshold on the probabilities of predictions for a specified output feature. 

Here is a breakdown of the input parameters:

- `probabilities_per_model` (List[np.array]): A list of numpy arrays representing the probabilities predicted by each model.
- `ground_truth` (Union[pd.Series, np.ndarray]): The ground truth values.
- `metadata` (dict): A dictionary containing metadata for the features.
- `output_feature_name` (str): The name of the output feature.
- `labels_limit` (int): An upper limit on the numeric encoded label value. Labels higher than this limit are considered "rare" labels.
- `model_names` (Union[str, List[str]], default: `None`): The name or names of the models to use as labels.
- `output_directory` (str, default: `None`): The directory where to save the plots. If not specified, the plots will be displayed in a window.
- `file_format` (str, default: `'pdf'`): The file format of the output plots (either `'pdf'` or `'png'`).
- `ground_truth_apply_idx` (bool, default: `True`): Whether to use the metadata's `'str2idx'` mapping in `np.vectorize`.

The function first checks if the `ground_truth` is not a numpy array and if so, it assumes that the raw values need to be translated to encoded values using the metadata's `'str2idx'` mapping.

Next, the function sets up some variables and lists to store the accuracies and dataset coverage for each model and threshold.

Then, for each model, it calculates the maximum probability and predicted labels based on the probabilities. It then iterates over the thresholds and filters the predictions based on the threshold. It calculates the accuracy and dataset coverage for each threshold and stores them in the respective lists.

Finally, the function calls a visualization utility function called `confidence_filtering_plot` to plot the accuracies and dataset coverage for each model and threshold. The plot is either displayed in a window or saved to a file in the specified output directory.

Overall, the function provides a way to analyze and compare the accuracy and data coverage of multiple models based on different confidence thresholds.

## Function **`confidence_thresholding_data_vs_acc`** Overview
The function `confidence_thresholding_data_vs_acc` takes in several inputs including a list of model probabilities, ground truth values, metadata, output feature name, labels limit, model names, output directory, file format, and other optional arguments. 

The function compares the accuracy of different models based on the confidence threshold of their predictions. It calculates the accuracy and data coverage for each model by increasing the threshold on the probabilities of predictions. The function uses two axes to visualize the data coverage and accuracy, instead of three axes used in the `confidence_thresholding` function. 

The function first checks if the ground truth values are not a numpy array and converts them to encoded values if necessary. It then applies a label limit if specified. 

Next, the function calculates the maximum probability and predictions for each model. It iterates over a range of thresholds and filters the predictions based on the threshold. It calculates the accuracy and data coverage for each threshold and stores them in separate lists. 

Finally, the function generates a plot using the `confidence_filtering_data_vs_acc_plot` function from the `visualization_utils` module. The plot compares the accuracy and data coverage for each model and saves it to a file if an output directory is specified.

### **Function Details**
The code defines a function `confidence_thresholding_data_vs_acc` that compares the accuracy of different models based on the confidence threshold of their predictions. 

The function takes the following inputs:
- `probabilities_per_model`: a list of numpy arrays representing the probabilities predicted by each model.
- `ground_truth`: the ground truth values.
- `metadata`: a dictionary containing metadata about the features.
- `output_feature_name`: the name of the output feature.
- `labels_limit`: an upper limit on the numeric encoded label value.
- `model_names`: the name or list of names of the models.
- `output_directory`: the directory where to save the plots.
- `file_format`: the file format of the output plots.
- `ground_truth_apply_idx`: a boolean indicating whether to use metadata['str2idx'] in np.vectorize.

The function first checks if the ground truth values are a numpy array, and if not, it assumes that the raw values need to be translated to encoded values using the metadata. If `labels_limit` is greater than 0, it limits the encoded label values to be less than or equal to `labels_limit`.

Next, it calculates the maximum probability and predicted labels for each model. It then iterates over a range of thresholds and filters the predictions based on the threshold. It calculates the accuracy of the filtered predictions and the percentage of data kept after filtering.

The accuracies and dataset coverage for each model and threshold are stored in lists. The function then calls a visualization function to plot the accuracies vs dataset coverage for each model.

If an `output_directory` is specified, the plots are saved in that directory with the specified `file_format`. Otherwise, the plots are displayed in a window.

The function does not return any value.

## Function **`confidence_thresholding_data_vs_acc_subset`** Overview
The function `confidence_thresholding_data_vs_acc_subset` compares the accuracy of multiple models based on the confidence threshold of their predictions. It takes in the following inputs:
- `probabilities_per_model`: A list of numpy arrays containing the predicted probabilities for each model.
- `ground_truth`: The ground truth values for the predictions.
- `metadata`: A dictionary containing metadata about the features.
- `output_feature_name`: The name of the output feature.
- `top_n_classes`: A list of integers specifying the number of classes to plot.
- `labels_limit`: An upper limit on the numeric encoded label value.
- `subset`: A string specifying the type of subset filtering to be applied.
- `model_names`: The name or list of names of the models to use as labels.
- `output_directory`: The directory where the plots will be saved.
- `file_format`: The file format of the output plots.
- `ground_truth_apply_idx`: A boolean indicating whether to use metadata['str2idx'] in np.vectorize.

The function calculates the accuracy and data coverage for each model at different confidence thresholds. It then visualizes the results using a line plot, with the x-axis representing the data coverage and the y-axis representing the accuracy.

The function returns None.

### **Function Details**
The code defines a function `confidence_thresholding_data_vs_acc_subset` that compares the accuracy of multiple models based on the confidence threshold of their predictions. The function takes the following inputs:

- `probabilities_per_model`: A list of numpy arrays containing the predicted probabilities for each model.
- `ground_truth`: The ground truth values for the output feature.
- `metadata`: A dictionary containing metadata for the features.
- `output_feature_name`: The name of the output feature.
- `top_n_classes`: A list of integers specifying the number of classes to plot.
- `labels_limit`: An upper limit on the numeric encoded label value.
- `subset`: A string specifying the type of subset filtering. Valid values are "ground_truth" or "predictions".
- `model_names`: A string or list of strings specifying the model names to use as labels.
- `output_directory`: The directory where to save the plots.
- `file_format`: The file format of the output plots.
- `ground_truth_apply_idx`: A boolean indicating whether to use metadata['str2idx'] in np.vectorize.

The function first checks if the ground truth values are a numpy array and if not, it converts the raw values to encoded values using the metadata. It then converts the `top_n_classes` to a list and assigns the first element to `k`. If `labels_limit` is greater than 0, it sets all ground truth values higher than `labels_limit` to `labels_limit`. 

The function then initializes empty lists for `accuracies` and `dataset_kept`. It creates a boolean array `subset_indices` based on the `subset` parameter and the ground truth values. If `subset` is "ground_truth", it selects only the datapoints where the ground truth class is within the top `k` most frequent ones. If `subset` is "predictions", it selects only the datapoints where the model predicts a class within the top `k` most frequent ones. It also updates the `gt_subset` variable accordingly.

Next, the function iterates over the probabilities for each model and performs the following steps:
- If `labels_limit` is greater than 0 and the number of classes in the probabilities is higher than `labels_limit + 1`, it limits the probabilities to `labels_limit + 1` classes and sums the probabilities of the remaining classes into the last class.
- If `subset` is "predictions", it updates the `subset_indices` and `gt_subset` variables based on the argmax of the probabilities.
- It selects the subset of probabilities based on the `subset_indices`.
- It calculates the maximum probability and the predicted class for each datapoint in the subset.
- It initializes empty lists for `accuracies_alg` and `dataset_kept_alg`.
- It iterates over the thresholds and performs the following steps:
  - If the threshold is greater than or equal to 1, it sets it to 0.999.
  - It filters the datapoints based on the maximum probability threshold.
  - It calculates the accuracy by comparing the filtered ground truth values with the filtered predicted classes.
  - It appends the accuracy and the percentage of datapoints kept to `accuracies_alg` and `dataset_kept_alg`, respectively.
- It appends `accuracies_alg` and `dataset_kept_alg` to `accuracies` and `dataset_kept`, respectively.

Finally, the function creates a filename based on the `output_directory` and `file_format` parameters. It creates the output directory if it doesn't exist. It then calls the `confidence_filtering_data_vs_acc_plot` function from the `visualization_utils` module to plot the accuracies and dataset coverage for each model. The plot is saved to the specified filename or displayed in a window if `output_directory` is not specified.

## Function **`confidence_thresholding_data_vs_acc_subset_per_class`** Overview
The function `confidence_thresholding_data_vs_acc_subset_per_class` compares the accuracy of different models based on their confidence threshold on a subset of data per class. 

The function takes the following inputs:
- `probabilities_per_model`: A list of numpy arrays representing the probabilities predicted by each model.
- `ground_truth`: The ground truth values.
- `metadata`: A dictionary containing intermediate preprocess structures created during training.
- `output_feature_name`: The name of the output feature to use for the visualization.
- `top_n_classes`: The number of top classes or a list containing the number of top classes to plot.
- `labels_limit`: An upper limit on the numeric encoded label value.
- `subset`: A string specifying the type of subset filtering. Valid values are "ground_truth" or "predictions".
- `model_names`: The name or list of names of the models to use as labels.
- `output_directory`: The directory where to save the plots.
- `file_format`: The file format of the output plots.
- `ground_truth_apply_idx`: A boolean indicating whether to use metadata['str2idx'] in np.vectorize.

The function produces a line plot for each class within the top_n_classes. The x-axis represents the data coverage (percentage of datapoints kept from the original set), and the y-axis represents the accuracy of the models. The plot shows how the accuracy changes as the confidence threshold on the probabilities of predictions increases.

The function first processes the inputs and prepares the necessary variables and filenames. It then iterates over each class within the top_n_classes and calculates the accuracies and dataset coverage for each model at different confidence thresholds. Finally, it calls a visualization function to plot the results.

The function does not return any value.

### **Function Details**
The code defines a function called `confidence_thresholding_data_vs_acc_subset_per_class` that compares the confidence threshold data vs accuracy on a subset of data per class in the top n classes. 

The function takes the following parameters:
- `probabilities_per_model`: a list of numpy arrays representing the probabilities of each model.
- `ground_truth`: the ground truth values.
- `metadata`: a dictionary containing intermediate preprocess structures created during training.
- `output_feature_name`: the name of the output feature to use for the visualization.
- `top_n_classes`: the number of top classes or a list containing the number of top classes to plot.
- `labels_limit`: an upper limit on the numeric encoded label value.
- `subset`: a string specifying the type of subset filtering. Valid values are "ground_truth" or "predictions".
- `model_names`: the model name or a list of model names to use as labels.
- `output_directory`: the directory where to save the plots.
- `file_format`: the file format of the output plots.
- `ground_truth_apply_idx`: a boolean indicating whether to use metadata['str2idx'] in np.vectorize.

The function produces a line plot for each class within the top_n_classes, showing the accuracy of the model and the data coverage while increasing a threshold on the probabilities of predictions for the specified output_feature_name. The subset of data used for the analysis depends on the value of the `subset` parameter. If `subset` is "ground_truth", only datapoints where the ground truth class is within the top n most frequent ones will be considered as the test set. If `subset` is "predictions", only datapoints where the model predicts a class that is within the top n most frequent ones will be considered as the test set.

The resulting plots can be saved in the specified output_directory or displayed in a window if no output_directory is specified. The file format of the plots can be either "pdf" or "png".

The function returns None.

## Function **`confidence_thresholding_2thresholds_2d`** Overview
The function `confidence_thresholding_2thresholds_2d` takes in several inputs including model probabilities, ground truth data, metadata, threshold output feature names, labels limit, model names, output directory, and file format. 

The function generates plots that show the relationship between confidence thresholds and accuracy for two output feature names. It uses the probabilities from multiple models to calculate accuracy based on different threshold values. The plots visualize the data coverage percentage or accuracy as the z-axis, with the confidence thresholds of the two output feature names as the x and y axes. 

The function first validates the input probabilities and thresholds. Then, it processes the ground truth data and applies any necessary transformations. It calculates the accuracies and data coverage for different combinations of threshold values. 

The function generates multiple plots using the `visualization_utils` module. These plots include a multiline plot that shows the relationship between coverage and accuracy for different threshold combinations, a max line plot that shows the maximum accuracy for each coverage level, and a max line plot with thresholds that shows the maximum accuracy along with the corresponding threshold values. 

The plots can be saved to an output directory in either PDF or PNG format, or displayed in a window if no output directory is specified.

### **Function Details**
The code provided is a function called `confidence_thresholding_2thresholds_2d`. This function takes several inputs including `probabilities_per_model`, `ground_truths`, `metadata`, `threshold_output_feature_names`, `labels_limit`, `model_names`, `output_directory`, `file_format`, and additional keyword arguments.

The function is used to visualize the relationship between confidence thresholds and accuracy for two output feature names. It generates multiple plots to show this relationship.

The function first validates the input probabilities and thresholds. Then, it processes the ground truth data and applies a label limit if specified. It calculates the maximum probabilities and predictions for each model. 

Next, it iterates over a range of thresholds for both output feature names and calculates the coverage and accuracy for each combination of thresholds. It stores these values in lists.

The function then generates three different plots using the `visualization_utils` module. The first plot is a multiline plot that shows the coverage vs accuracy for each combination of thresholds. The second plot is a max line plot that shows the maximum accuracy achieved for each coverage level. The third plot is a max line plot with thresholds, which shows the maximum accuracy achieved for each coverage level along with the corresponding thresholds.

Finally, the function saves the plots to the specified output directory if provided, or displays them in a window if no output directory is specified.

The function returns `None`.

## Function **`confidence_thresholding_2thresholds_3d`** Overview
The function `confidence_thresholding_2thresholds_3d` takes in several inputs including a list of model probabilities, ground truth data, feature metadata, output feature names, a limit on label values, an output directory, and a file format. 

The function first validates the input probabilities and output feature names using the `validate_conf_thresholds_and_probabilities_2d_3d` function. If the validation fails, the function returns.

Next, the function processes the ground truth data. If the ground truth data is not a numpy array, it assumes that the raw values need to be translated to encoded values using the feature metadata. The function applies a vectorized function `_encode_categorical_feature` to each element of the ground truth data to perform the translation.

The function then applies a label limit to the ground truth data if the limit is greater than 0. Any label values higher than the limit are set to the limit.

The function defines a list of thresholds ranging from 0 to 1 with a step size of 0.05.

Next, the function calculates the maximum probabilities and predictions for each model. It also handles the case where the number of label values exceeds the label limit by summing the probabilities of the "rare" labels.

The function then iterates over the thresholds to calculate accuracies and dataset coverage percentages for different combinations of the two thresholds. It filters the ground truth data and predictions based on the two thresholds and calculates the accuracy as the ratio of correctly predicted samples to the total number of filtered samples.

The accuracies and dataset coverage percentages are stored in lists.

Finally, the function creates an output file path if an output directory is specified, and calls the `confidence_filtering_3d_plot` function from the `visualization_utils` module to generate a 3D plot of the accuracies and dataset coverage percentages. The plot is saved to the output file path if specified, or displayed in a window if not.

The function does not return any value.

### **Function Details**
The code defines a function `confidence_thresholding_2thresholds_3d` that takes in several input parameters and plots a 3D surface plot. Here is a breakdown of the code:

1. The function signature specifies the input and output types of the function.
2. The function documentation provides a description of the function and its parameters.
3. The function tries to validate the input probabilities and threshold output feature names using a helper function `validate_conf_thresholds_and_probabilities_2d_3d`. If an error occurs, the function returns without further execution.
4. The probabilities are assigned to the `probs` variable.
5. If the ground truths are not numpy arrays, they are assumed to be raw values and are encoded using a helper function `_encode_categorical_feature`.
6. If a labels limit is specified, any label values higher than the limit are considered "rare" labels and are replaced with the limit value.
7. Thresholds are generated as a list of values ranging from 0 to 1 with a step size of 0.05.
8. Empty lists `accuracies` and `dataset_kept` are initialized.
9. If the labels limit is greater than 0 and the number of classes in the probabilities exceeds the limit, the probabilities are adjusted to include a "rare" label.
10. The maximum probabilities and corresponding predictions are computed for each set of probabilities.
11. Two nested loops iterate over the thresholds to calculate the accuracy and dataset coverage for each combination of thresholds.
12. The accuracy and dataset coverage values are appended to the `accuracies` and `dataset_kept` lists, respectively.
13. If an output directory is specified, the function creates the directory if it doesn't exist and generates a filename for the plot.
14. The `confidence_filtering_3d_plot` function from the `visualization_utils` module is called to generate the 3D plot using the threshold values, accuracies, dataset coverage, and other parameters.
15. The plot is saved to the specified output directory if provided, otherwise it is displayed in a window.

Overall, the function takes in probabilities, ground truths, and other parameters, calculates accuracies and dataset coverage for different combinations of thresholds, and generates a 3D plot to visualize the relationship between the thresholds and the accuracy/dataset coverage.

## Function **`binary_threshold_vs_metric`** Overview
The function `binary_threshold_vs_metric` takes in several inputs including a list of model probabilities, ground truth values, metadata, output feature name, metrics to display, positive label, model names, output directory, file format, and a boolean flag. 

The function visualizes the confidence of the model against a specified metric for the given output feature name. It produces a line chart with a threshold on the model's confidence plotted against the metric. 

If the output feature name is a category feature, the positive label indicates the class to be considered as the positive class, while all other classes are considered negative. 

The function calculates the specified metric for each model and threshold combination, and stores the scores in a list. It then uses a visualization utility function to plot the threshold vs. metric chart, with optional saving of the plot to a file.

### **Function Details**
The code defines a function called `binary_threshold_vs_metric` that visualizes the confidence of a model against a specified metric for a given output feature. 

The function takes the following inputs:
- `probabilities_per_model`: A list of numpy arrays representing the model probabilities.
- `ground_truth`: The ground truth values, either as a pandas Series or a numpy array.
- `metadata`: A dictionary containing feature metadata.
- `output_feature_name`: The name of the output feature.
- `metrics`: A list of metrics to display, including `'f1'`, `'precision'`, `'recall'`, and `'accuracy'`.
- `positive_label`: The numeric encoded value for the positive class (default is 1).
- `model_names`: A list of model names to use as labels (default is None).
- `output_directory`: The directory where to save the plots (default is None).
- `file_format`: The file format of the output plots, either `'pdf'` or `'png'` (default is `'pdf'`).
- `ground_truth_apply_idx`: A boolean indicating whether to use metadata['str2idx'] in np.vectorize (default is True).
- `**kwargs`: Additional keyword arguments.

The function first checks if the ground truth is not a numpy array and converts the raw values to encoded values if necessary. It then assigns the probabilities to the `probs` variable and converts the `model_names` and `metrics` inputs to lists if they are not already. 

Next, the function defines a filename template for saving the plots and generates the filename template path based on the output directory. It also creates a list of thresholds ranging from 0 to 1 with a step size of 0.05.

The function then checks if the specified metrics are supported (f1, precision, recall, accuracy) and proceeds to calculate the scores for each model and threshold. The scores are stored in a nested list.

Finally, the function calls a visualization utility function called `threshold_vs_metric_plot` to plot the threshold vs metric for each model. The plot is either displayed in a window or saved to a file in the specified output directory.

The function does not return any value.

## Function **`precision_recall_curves`** Overview
The function `precision_recall_curves` is used to visualize precision-recall curves for output features in specified models. 

The function takes the following inputs:
- `probabilities_per_model`: A list of numpy arrays representing the model probabilities.
- `ground_truth`: The ground truth values.
- `metadata`: A dictionary containing feature metadata.
- `output_feature_name`: The name of the output feature.
- `positive_label`: The numeric encoded value for the positive class (default: 1).
- `model_names`: The name or list of names of the models to use as labels (default: None).
- `output_directory`: The directory where to save the plots (default: None).
- `file_format`: The file format of the output plots (default: "pdf").
- `ground_truth_apply_idx`: Whether to use metadata['str2idx'] in np.vectorize (default: True).

The function first checks if the ground truth values are not a numpy array and converts them to the encoded value if necessary. Then, it calculates the precision and recall values for each model using the `sklearn.metrics.precision_recall_curve` function. The precision and recall values are stored in a list of dictionaries.

If an output directory is specified, the function creates the directory if it doesn't exist and saves the precision-recall curve plot in the specified file format. Finally, the function calls `visualization_utils.precision_recall_curves_plot` to display the precision-recall curves.

### **Function Details**
The given code defines a function `precision_recall_curves` that visualizes precision-recall curves for output features in specified models. 

The function takes the following inputs:
- `probabilities_per_model`: A list of numpy arrays representing the model probabilities.
- `ground_truth`: The ground truth values, which can be either a pandas Series or a numpy array.
- `metadata`: A dictionary containing feature metadata.
- `output_feature_name`: The name of the output feature for which precision-recall curves are to be plotted.
- `positive_label`: The numeric encoded value for the positive class (default: 1).
- `model_names`: The name or list of names of the models to use as labels (default: None).
- `output_directory`: The directory where the plots will be saved (default: None).
- `file_format`: The file format of the output plots (default: "pdf").
- `ground_truth_apply_idx`: A boolean indicating whether to use metadata['str2idx'] in np.vectorize (default: True).

The function calculates precision and recall values for each model using the `sklearn.metrics.precision_recall_curve` function and stores them in a list of dictionaries. It then calls the `precision_recall_curves_plot` function from the `visualization_utils` module to plot the precision-recall curves.

If `output_directory` is specified, the plots are saved in the specified directory with the given `file_format`. Otherwise, the plots are displayed in a window.

The function does not return any value.

## Function **`precision_recall_curves_from_test_statistics`** Overview
The function `precision_recall_curves_from_test_statistics` takes in several parameters including `test_stats_per_model`, `output_feature_name`, `model_names`, `output_directory`, and `file_format`. 

It is used to visualize precision-recall curves for binary output features of different models. The `test_stats_per_model` parameter is a list of dictionaries containing evaluation performance statistics for each model. The `output_feature_name` parameter specifies the name of the output feature to use for the visualization.

The function generates precision-recall curves for each model using the provided statistics and output feature name. It then uses the `model_names` parameter to label the curves. The resulting visualization is a line chart showing the precision-recall curves for the specified output feature.

The function also allows for saving the plots to a specified `output_directory` in either PDF or PNG format, depending on the `file_format` parameter. If no `output_directory` is specified, the plots will be displayed in a window.

The function returns `None` after generating the precision-recall curves plot.

### **Function Details**
The given code defines a function `precision_recall_curves_from_test_statistics` that visualizes precision-recall curves for binary classification models. 

The function takes the following parameters:
- `test_stats_per_model`: A list of dictionaries containing evaluation performance statistics for each model.
- `output_feature_name`: The name of the output feature to use for the visualization.
- `model_names`: The name or list of names of the models to use as labels.
- `output_directory`: The directory where the plots will be saved. If not specified, the plots will be displayed in a window.
- `file_format`: The file format of the output plots, either "pdf" or "png".
- `**kwargs`: Additional keyword arguments.

The function first converts the `model_names` parameter to a list if it is not already a list. It then generates a filename template for the output plots based on the `file_format` parameter and the specified output directory.

Next, the function extracts the precision and recall values from the `test_stats_per_model` parameter for the specified `output_feature_name`. It creates a list of dictionaries, where each dictionary contains the precision and recall values for a specific model.

Finally, the function calls the `precision_recall_curves_plot` function from the `visualization_utils` module, passing the precision-recall data, model names, title, and filename template path as arguments.

The function does not return any value.

## Function **`roc_curves`** Overview
The function `roc_curves` is used to plot ROC curves for output features in specified models. 

The function takes the following inputs:
- `probabilities_per_model`: A list of numpy arrays representing the model probabilities.
- `ground_truth`: The ground truth values, which can be either a pandas Series or a numpy array.
- `metadata`: A dictionary containing feature metadata.
- `output_feature_name`: The name of the output feature for which ROC curves will be plotted.
- `positive_label`: The numeric encoded value for the positive class (default is 1).
- `model_names`: The name or list of names of the models to be used as labels (default is None).
- `output_directory`: The directory where the plots will be saved (default is None, which means the plots will be displayed in a window).
- `file_format`: The file format of the output plots, either 'pdf' or 'png' (default is 'pdf').
- `ground_truth_apply_idx`: A boolean indicating whether to use metadata['str2idx'] in np.vectorize (default is True).

The function first checks if the ground truth values are not a numpy array, in which case it assumes that the raw values need to be translated to encoded values. It then calls the `_convert_ground_truth` function to perform this translation.

Next, the function iterates over the probabilities for each model and calculates the false positive rate (fpr), true positive rate (tpr), and thresholds using the `sklearn.metrics.roc_curve` function. These values are stored in a list of tuples.

If an output directory is specified, the function creates the directory if it doesn't exist and sets the filename for the output plot.

Finally, the function calls the `visualization_utils.roc_curves` function to plot the ROC curves using the fpr, tpr, and model names. The title of the plot is set to "ROC curves" and the filename is passed if it was specified.

The function does not return any value.

### **Function Details**
The given code defines a function `roc_curves` that plots ROC curves for output features in specified models. The function takes the following inputs:

- `probabilities_per_model`: A list of numpy arrays representing the model probabilities.
- `ground_truth`: The ground truth values, either as a pandas Series or a numpy array.
- `metadata`: A dictionary containing feature metadata.
- `output_feature_name`: The name of the output feature for which ROC curves are to be plotted.
- `positive_label`: The numeric encoded value for the positive class (default: 1).
- `model_names`: The name or list of names of the models to use as labels (default: None).
- `output_directory`: The directory where the plots will be saved (default: None).
- `file_format`: The file format of the output plots (default: "pdf").
- `ground_truth_apply_idx`: A boolean indicating whether to use metadata['str2idx'] in np.vectorize (default: True).

The function first checks if the ground truth values are not a numpy array and converts them to encoded values if necessary. Then, it iterates over the probabilities for each model, calculates the false positive rate (fpr) and true positive rate (tpr) using sklearn.metrics.roc_curve, and appends them to a list. Finally, it calls a visualization utility function to plot the ROC curves.

If `output_directory` is specified, the plots are saved in the specified directory with the given `file_format`. Otherwise, the plots are displayed in a window.

The function does not return any value.

## Function **`roc_curves_from_test_statistics`** Overview
The function `roc_curves_from_test_statistics` takes in several parameters including `test_stats_per_model`, `output_feature_name`, `model_names`, `output_directory`, and `file_format`. 

It is used to visualize the ROC (Receiver Operating Characteristic) curves for the specified models based on the evaluation performance statistics provided in `test_stats_per_model`. The ROC curves are plotted for the binary output feature specified by `output_feature_name`.

The function first converts the `model_names` parameter to a list if it is not already a list. It then generates a filename template based on the `file_format` parameter and the output directory specified in `output_directory`.

Next, the function iterates over each set of test statistics in `test_stats_per_model` and extracts the false positive rate (FPR) and true positive rate (TPR) values from the ROC curve for the specified `output_feature_name`. These FPR and TPR values are stored in a list `fpr_tprs`.

Finally, the function calls the `roc_curves` function from the `visualization_utils` module, passing in the `fpr_tprs`, `model_names_list`, and other parameters such as the title and filename for the plot. The `roc_curves` function is responsible for actually plotting the ROC curves.

The function does not return any value, as indicated by the `-> None` in the function signature.

### **Function Details**
The given code defines a function `roc_curves_from_test_statistics` that visualizes ROC curves for binary classification models. 

The function takes the following parameters:
- `test_stats_per_model`: A list of dictionaries containing evaluation performance statistics for each model.
- `output_feature_name`: The name of the output feature to use for the visualization.
- `model_names`: The name or list of names of the models to use as labels (optional).
- `output_directory`: The directory where to save the plots (optional).
- `file_format`: The file format of the output plots, either "pdf" or "png" (optional).
- `**kwargs`: Additional keyword arguments.

The function first converts the `model_names` parameter to a list if it is not already a list. It then generates a filename template based on the `file_format` parameter and the output directory. 

Next, the function iterates over each dictionary in `test_stats_per_model` and extracts the false positive rate (fpr) and true positive rate (tpr) from the `output_feature_name` key. These fpr and tpr values are appended to a list `fpr_tprs`.

Finally, the function calls a visualization utility function `roc_curves` with the `fpr_tprs`, `model_names_list`, a title for the plot, and the filename template path. The `roc_curves` function is not defined in the given code.

The function does not return any value, it only displays or saves the ROC curves plot.

## Function **`calibration_1_vs_all`** Overview
The function `calibration_1_vs_all` takes in several inputs including a list of model probabilities, ground truth values, metadata, and other parameters. 

The function produces two plots for each class or the top k most frequent classes, where k is specified by the `top_n_classes` parameter. 

The first plot is a calibration curve that shows the calibration of the predictions for each model, considering the current class as the true one and all others as false. Each model is represented by a line on the plot.

The second plot shows the distributions of the predictions for each model, considering the current class as the true one and all others as false. 

The function also calculates Brier scores for each class and produces a plot showing the scores for each model and class.

The plots can be saved to a specified output directory or displayed in a window. The file format of the output plots can be specified as either PDF or PNG.

### **Function Details**
The code provided is a function called `calibration_1_vs_all` that takes in several input parameters and performs calibration analysis for binary classification models. Here is a breakdown of the function:

1. The function takes the following input parameters:
   - `probabilities_per_model`: A list of numpy arrays containing the predicted probabilities for each model.
   - `ground_truth`: The ground truth values for the target variable.
   - `metadata`: A dictionary containing metadata information about the features.
   - `output_feature_name`: The name of the output feature.
   - `top_n_classes`: A list containing the number of classes to plot.
   - `labels_limit`: An upper limit on the numeric encoded label value.
   - `model_names`: A list of strings representing the names of the models.
   - `output_directory`: The directory where the plots will be saved.
   - `file_format`: The file format of the output plots.
   - `ground_truth_apply_idx`: A boolean indicating whether to use metadata['str2idx'] in np.vectorize.

2. The function first checks if the `ground_truth` is not a numpy array and converts it to an encoded value if necessary.

3. It assigns the input probabilities and model names to local variables.

4. It sets up a filename template for saving the plots and creates the output directory if specified.

5. If `labels_limit` is greater than 0, it limits the ground truth values to the specified limit.

6. It then iterates over the probabilities for each model and limits the probabilities to the specified labels limit if necessary.

7. It calculates the number of classes and initializes an empty list for storing Brier scores.

8. It determines the number of classes to plot based on the `top_n_classes` parameter.

9. It retrieves the class names from the feature metadata.

10. It iterates over each class and performs the following steps:
    - Initializes empty lists for storing calibration curve values, mean predicted values, probabilities, and Brier scores for each model.
    - For each model, it calculates the calibration curve and mean predicted values using the `calibration_curve` function.
    - If the length of the calibration curve or mean predicted values is less than 2, it appends a zero value to the beginning of the arrays.
    - It appends the calculated values to the respective lists.
    - It calculates the Brier score using the `brier_score_loss` function and appends it to the Brier scores list.
    - It saves the calibration plot and prediction distribution plot for the current class if an output directory is specified.
    
11. After iterating over all classes, it saves the Brier scores plot if an output directory is specified.

Overall, the function performs calibration analysis for binary classification models and generates calibration plots, prediction distribution plots, and Brier scores plots for each class. The plots can be saved to an output directory or displayed in a window.

## Function **`calibration_multiclass`** Overview
The function `calibration_multiclass` takes in several inputs including a list of model probabilities, ground truth values, feature metadata, output feature name, labels limit, model names, output directory, file format, and other optional arguments. 

The function calculates the calibration curve for each model's probability predictions for each class of the specified output feature. It also calculates the fraction of positives, mean predicted values, and Brier scores for each model. 

The function then generates and saves or displays plots of the calibration curve and the comparison of Brier scores for each model. It also logs the Brier scores for each model.

### **Function Details**
The code provided is a function called `calibration_multiclass` that takes in several input parameters and performs calibration analysis for multiclass classification models. Here is a breakdown of the function:

- `probabilities_per_model` (List[np.array]): A list of numpy arrays containing the predicted probabilities for each class from multiple models.
- `ground_truth` (Union[pd.Series, np.ndarray]): The ground truth values for the target variable.
- `metadata` (dict): A dictionary containing metadata information about the features.
- `output_feature_name` (str): The name of the output feature.
- `labels_limit` (int): An upper limit on the numeric encoded label value. Labels higher than this limit are considered "rare" labels.
- `model_names` (Union[str, List[str]], default: None): A list of names for the models. If not provided, the models will be labeled as Model 1, Model 2, etc.
- `output_directory` (str, default: None): The directory where the plots will be saved. If not specified, the plots will be displayed in a window.
- `file_format` (str, default: 'pdf'): The file format of the output plots, either 'pdf' or 'png'.
- `ground_truth_apply_idx` (bool, default: True): Whether to use the metadata['str2idx'] in np.vectorize.

The function first checks if the ground truth values are not a numpy array and converts them to the encoded values if necessary. Then, it performs some preprocessing on the predicted probabilities and ground truth values to prepare them for calibration analysis.

Next, it calculates the fraction of positives and mean predicted values for each model using the `calibration_curve` function and stores them in `fraction_positives` and `mean_predicted_vals` lists, respectively. It also calculates the Brier scores for each model using the `brier_score_loss` function and stores them in the `brier_scores` list.

The function then generates the filenames for the calibration plot and the comparison plot based on the output directory and file format. It calls the `calibration_plot` function from the `visualization_utils` module to create the calibration plot using the fraction of positives and mean predicted values. It also calls the `compare_classifiers_plot` function from the `visualization_utils` module to create the comparison plot using the Brier scores.

Finally, the function logs the Brier scores for each model using the `logger.info` function.

Note: Some helper functions and modules used in the code, such as `_vectorize_ground_truth`, `convert_to_list`, `generate_filename_template_path`, `calibration_curve`, `brier_score_loss`, and `visualization_utils`, are not provided in the code snippet.

## Function **`confusion_matrix`** Overview
The `confusion_matrix` function takes in several parameters including `test_stats_per_model`, `metadata`, `output_feature_name`, `top_n_classes`, `normalize`, `model_names`, `output_directory`, and `file_format`. 

This function is used to display the confusion matrix in the models' predictions for each `output_feature_name`. It iterates through the `test_stats_per_model` list and checks if a confusion matrix exists for each output feature. If a confusion matrix is found, it creates a heatmap of the confusion matrix and saves it as a plot. It also calculates the entropy of each row in the confusion matrix and creates a bar plot of the classes ranked by entropy.

The function returns `None` and raises an error if no confusion matrix is found in the evaluation data.

### **Function Details**
The code provided is a function called `confusion_matrix` that takes several input parameters and returns `None`. 

The function is used to generate confusion matrix visualizations for the predictions of multiple models. It takes the following parameters:

- `test_stats_per_model`: A list of dictionaries containing evaluation performance statistics for each model.
- `metadata`: A dictionary containing intermediate preprocess structures created during training.
- `output_feature_name`: The name of the output feature to use for the visualization. If `None`, all output features are used.
- `top_n_classes`: A list of integers specifying the number of top classes to plot in the confusion matrix.
- `normalize`: A boolean flag indicating whether to normalize the rows in the confusion matrix.
- `model_names`: The name or list of names of the models to use as labels.
- `output_directory`: The directory where the plots will be saved. If not specified, the plots will be displayed in a window.
- `file_format`: The file format of the output plots (either "pdf" or "png").

The function first converts the `model_names` parameter to a list and defines a filename template for the output plots. It then checks if the confusion matrix is present in the evaluation data for each model and output feature. If a confusion matrix is found, it extracts the matrix, model name, and labels from the metadata. It then generates the confusion matrix plot and saves it to a file if `output_directory` is specified. 

After generating the confusion matrix plot, the function calculates the entropy of each row in the confusion matrix and generates a bar plot of the classes ranked by entropy. This plot is also saved to a file if `output_directory` is specified.

If no confusion matrix is found in the evaluation data, an error is logged and a `FileNotFoundError` is raised.

Overall, the function is used to visualize and analyze the confusion matrices of multiple models for different output features.

## Function **`frequency_vs_f1`** Overview
The function `frequency_vs_f1` takes in several parameters including `test_stats_per_model`, `metadata`, `output_feature_name`, `top_n_classes`, `model_names`, `output_directory`, `file_format`, and `**kwargs`. 

The function generates two plots for each model in the `test_stats_per_model` list. The first plot is a line plot with two vertical axes colored in orange and blue. The orange axis represents the frequency of each class, and the blue axis represents the F1 score for each class. The classes on the x-axis are sorted by F1 score. 

The second plot has the same structure as the first one, but the axes are flipped, and the classes on the x-axis are sorted by frequency. 

The function uses the provided `test_stats_per_model` and `model_names` to iterate over each model and output feature. It retrieves the necessary statistics and metadata from the `test_stats_per_model` and `metadata` dictionaries. It then sorts the classes based on F1 score and frequency, and creates the line plots using the `visualization_utils.double_axis_line_plot` function. The plots can be saved to a specified `output_directory` or displayed in a window. The file format of the output plots can be specified using the `file_format` parameter.

### **Function Details**
The code provided is a function called `frequency_vs_f1` that takes in several parameters and produces two plots for each model in the `test_stats_per_model` list.

The function takes the following parameters:
- `test_stats_per_model`: A list of dictionaries containing evaluation performance statistics for each model.
- `metadata`: A dictionary containing intermediate preprocess structure created during training, including mappings of the input dataset.
- `output_feature_name`: The name of the output feature to use for the visualization. If `None`, all output features are used.
- `top_n_classes`: A list of integers representing the number of top classes or a list containing the number of top classes to plot.
- `model_names`: The name of the model or a list of model names to use as labels. Default is `None`.
- `output_directory`: The directory where to save the plots. If not specified, the plots will be displayed in a window. Default is `None`.
- `file_format`: The file format of the output plots, either `'pdf'` or `'png'`. Default is `'pdf'`.
- `**kwargs`: Additional keyword arguments.

The function produces two plots for each model and output feature combination. The first plot is a line plot with one x-axis representing the different classes. The plot has two vertical axes colored in orange and blue. The orange axis represents the frequency of the class, and an orange line is plotted to show the trend. The blue axis represents the F1 score for that class, and a blue line is plotted to show the trend. The classes on the x-axis are sorted by F1 score.

The second plot has the same structure as the first one, but the axes are flipped, and the classes on the x-axis are sorted by frequency.

The function returns `None`.

## Function **`hyperopt_report_cli`** Overview
The function `hyperopt_report_cli` is a command-line interface (CLI) function that generates a report about hyperparameter optimization. It takes in the path to a JSON file containing the hyperopt results, as well as optional parameters for the output directory and file format. 

The function uses the `hyperopt_report` function to create one graph per hyperparameter, showing the distribution of results. It also generates an additional graph that visualizes the interactions between pairwise hyperparameters. 

The purpose of this function is to provide a convenient way to generate a report summarizing the results of hyperparameter optimization, making it easier to analyze and interpret the performance of different hyperparameter configurations. The output plots can be saved in either PDF or PNG format.

### **Function Details**
The given code defines a function `hyperopt_report_cli` that generates a report about hyperparameter optimization. The function takes the following parameters:

- `hyperopt_stats_path`: a string representing the path to the hyperopt results JSON file.
- `output_directory`: an optional string representing the path where the output plots will be saved. If not provided, the plots will not be saved.
- `file_format`: a string representing the format of the output plots. It can be either "pdf" or "png".
- `**kwargs`: additional keyword arguments that can be passed to the `hyperopt_report` function.

The function calls the `hyperopt_report` function with the provided parameters to generate the report. The `hyperopt_report` function is not shown in the given code, so its implementation is not known.

## Function **`hyperopt_report`** Overview
The function `hyperopt_report` is used to produce a report about hyperparameter optimization. It takes in the path to a JSON file containing the hyperopt results, as well as optional parameters for the output directory and file format. 

The function generates one graph per hyperparameter to show the distribution of results, and an additional graph to visualize pairwise interactions between hyperparameters. The output plots can be saved in the specified output directory or displayed in a window.

The function first loads the hyperopt results from the JSON file. It then calls the `hyperopt_report` function from the `visualization_utils` module, passing in the hyperparameter configuration, the hyperopt results converted to a dataframe, the metric used for optimization, and the filename template for the output plots.

### **Function Details**
The code defines a function called `hyperopt_report` that generates a report about hyperparameter optimization. The function takes the following parameters:

- `hyperopt_stats_path`: a string representing the path to the hyperopt results JSON file.
- `output_directory`: a string representing the directory where the plots will be saved. If not specified, the plots will be displayed in a window.
- `file_format`: a string representing the file format of the output plots, either `'pdf'` or `'png'`.

The function first defines a `filename_template` string that will be used to generate the filenames for the output plots. The `filename_template` string includes a placeholder `{}` that will be replaced with the hyperparameter name.

The function then loads the hyperopt results from the JSON file using the `load_json` function. The loaded results are stored in the `hyperopt_stats` variable.

Finally, the function calls the `visualization_utils.hyperopt_report` function to generate the report. It passes the hyperparameter configuration, the hyperopt results converted to a dataframe, the metric used for optimization, and the filename template path as arguments to the `hyperopt_report` function.

The function does not return any value.

## Function **`hyperopt_hiplot_cli`** Overview
The function `hyperopt_hiplot_cli` is a command-line interface for generating a parallel coordinate plot for hyperparameter optimization. It takes as input the path to a JSON file containing the results of hyperparameter optimization using the Hyperopt library. 

The function generates an HTML file that displays the parallel coordinate plot, which visualizes the relationship between different hyperparameters and their corresponding performance metrics. Additionally, it can also generate a CSV file that can be read by the HiPlot library.

The `output_directory` parameter allows the user to specify the path where the output plots should be saved. If not provided, the plots will be saved in the current working directory.

Overall, the function provides a convenient way to visualize and analyze the results of hyperparameter optimization using Hyperopt.

### **Function Details**
The given code defines a function `hyperopt_hiplot_cli` that takes in the path to a hyperopt results JSON file and an optional output directory. It then calls another function `hyperopt_hiplot` with the same arguments.

The purpose of this code is to generate a parallel coordinate plot using the hiplot library for hyperparameter optimization. The resulting plot is saved as an HTML file, and an optional CSV file can also be generated.

The `hyperopt_hiplot` function is not provided in the given code, so it is assumed to be defined elsewhere.

## Function **`hyperopt_hiplot`** Overview
The function `hyperopt_hiplot` is used to produce a parallel coordinate plot for hyperparameter optimization. It takes as input the path to a JSON file containing the hyperopt results, and optionally an output directory where the plot will be saved. If no output directory is specified, the plot will be displayed in a window.

The function first loads the hyperopt results from the JSON file using the `load_json` function. It then converts the hyperopt results into a pandas DataFrame using the `hyperopt_results_to_dataframe` function, which takes the hyperopt results, hyperparameter configuration, and metric as input.

Finally, the function calls the `hyperopt_hiplot` function from the `visualization_utils` module to generate the parallel coordinate plot. The plot is saved as an HTML file with the specified filename in the output directory.

The function does not return any value.

### **Function Details**
The code provided is a function called `hyperopt_hiplot` that takes in a path to a hyperopt results JSON file and an optional output directory. It produces a parallel coordinate plot about hyperparameter optimization and saves it as an HTML file. The function uses the `load_json` function to load the hyperopt results from the JSON file, and then converts the results into a pandas DataFrame using the `hyperopt_results_to_dataframe` function. Finally, it uses the `hyperopt_hiplot` function from the `visualization_utils` module to generate the plot and save it as an HTML file.

## Function **`_convert_space_to_dtype`** Overview
The function `_convert_space_to_dtype` takes a string `space` as input and returns a string representing the data type of the space. 

If the input `space` is found in the list `RAY_TUNE_FLOAT_SPACES` defined in the `visualization_utils` module, the function returns the string "float". 

If the input `space` is found in the list `RAY_TUNE_INT_SPACES` defined in the `visualization_utils` module, the function returns the string "int". 

If the input `space` is not found in either of the above lists, the function returns the string "object".

### **Function Details**
The given code is a Python function named `_convert_space_to_dtype` that takes a string parameter `space` and returns a string representing the data type of the space.

The function checks if the `space` parameter is present in two lists `visualization_utils.RAY_TUNE_FLOAT_SPACES` and `visualization_utils.RAY_TUNE_INT_SPACES`. If it is present in either of the lists, it returns the string "float" or "int" respectively. Otherwise, it returns the string "object".

Note: The code assumes that the lists `visualization_utils.RAY_TUNE_FLOAT_SPACES` and `visualization_utils.RAY_TUNE_INT_SPACES` are defined elsewhere in the code.

## Function **`hyperopt_results_to_dataframe`** Overview
The function `hyperopt_results_to_dataframe` takes in three parameters: `hyperopt_results`, `hyperopt_parameters`, and `metric`. 

It converts the results from a hyperparameter optimization process, stored in `hyperopt_results`, into a pandas DataFrame. Each row in the DataFrame represents a set of hyperparameters and their corresponding metric score.

The function iterates over each result in `hyperopt_results` and creates a dictionary with the metric score and the hyperparameters. It then creates a DataFrame using this list of dictionaries.

Next, the function converts the data types of the hyperparameters in the DataFrame based on the information provided in `hyperopt_parameters`. It uses the `_convert_space_to_dtype` function to determine the appropriate data type for each hyperparameter.

Finally, the function returns the resulting DataFrame.

### **Function Details**
The given code defines a function `hyperopt_results_to_dataframe` that takes three arguments: `hyperopt_results`, `hyperopt_parameters`, and `metric`. 

The function creates an empty DataFrame `df` and then iterates over each result in `hyperopt_results`. For each result, it extracts the value of the `metric_score` and the `parameters` dictionary. It then creates a new dictionary with the `metric_score` as one key-value pair and the parameters as the remaining key-value pairs. This dictionary is then added as a row to the DataFrame `df`.

After iterating over all results, the function converts the data types of the columns in the DataFrame based on the `hyperopt_parameters` dictionary. It uses the `_convert_space_to_dtype` function to determine the appropriate data type for each column based on the corresponding space defined in `hyperopt_parameters`.

Finally, the function returns the resulting DataFrame `df`.

Note: The code assumes that the `pd` module from the pandas library is imported and that the `_convert_space_to_dtype` function is defined elsewhere.

## Function **`get_visualizations_registry`** Overview
The function `get_visualizations_registry` returns a dictionary that maps string keys to callable functions. Each key represents a specific visualization, and the corresponding value is the function that generates that visualization. The function can be used to retrieve the appropriate visualization function based on a given key.

### **Function Details**
This code defines a function `get_visualizations_registry()` that returns a dictionary. The keys of the dictionary are strings representing different visualization names, and the values are callable functions that correspond to each visualization.

Here is an example of how this function can be used:

```python
visualizations = get_visualizations_registry()
visualization_name = "compare_performance"
visualization_function = visualizations[visualization_name]
visualization_function()
```

In this example, `visualization_name` is set to "compare_performance", and `visualization_function` is retrieved from the dictionary using the key "compare_performance". The function `visualization_function` can then be called to execute the corresponding visualization.

## Function **`cli`** Overview
The function `cli` is a command-line interface function that takes a list of command-line arguments (`sys_argv`) as input. It uses the `argparse` module to define and parse the command-line arguments.

The function creates an argument parser with various options and arguments. These options and arguments include specifying the ground truth file, ground truth metadata file, split file, output directory, file format of output plots, type of visualization to generate, output feature name, ground truth split, threshold output feature names, predictions files, probabilities files, training statistics files, test statistics files, hyperopt stats file, model names, number of classes to plot, number of elements in the ranklist to consider, maximum number of labels, type of subset filtering, whether to normalize rows in confusion matrix, metrics to display in threshold_vs_metric, label of the positive class for the roc curve, and logging level.

The function then adds any additional callback arguments using the `add_contrib_callback_args` function.

After parsing the command-line arguments, the function sets the logging level and retrieves the visualization function based on the specified visualization argument.

Finally, the function calls the visualization function with the parsed arguments using the `vis_func(**vars(args))` syntax.

### **Function Details**
The given code defines a command-line interface (CLI) function called `cli` that takes in a list of command-line arguments (`sys_argv`). The function uses the `argparse` module to define and parse the command-line arguments.

The `argparse.ArgumentParser` is created with a description, program name, and usage information. Then, various command-line arguments are added using the `add_argument` method. Each argument has a short and long option, a help message, and optional default values and choices.

After parsing the command-line arguments, the function performs some additional processing and logging based on the parsed arguments. Finally, it calls a visualization function based on the value of the `visualization` argument.

Note: The code references some functions and variables (`get_visualizations_registry`, `add_contrib_callback_args`, `get_logging_level_registry`, etc.) that are not defined in the given code snippet. These functions and variables are likely defined elsewhere in the codebase.

