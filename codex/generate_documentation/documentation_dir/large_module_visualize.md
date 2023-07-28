# Module:`visualize.py` Overview

## **Error in generating module level documentation**

## Function **`_convert_ground_truth`** Overview
The function `_convert_ground_truth` takes four parameters: `ground_truth`, `feature_metadata`, `ground_truth_apply_idx`, and `positive_label`. 

The purpose of each parameter is as follows:
- `ground_truth`: This parameter represents the ground truth data, which is the actual values of the target variable.
- `feature_metadata`: This parameter contains metadata information about the features, including any necessary conversions or transformations.
- `ground_truth_apply_idx`: This parameter is used to apply the ground truth data to a specific index or subset of the data.
- `positive_label`: This parameter represents the positive label for binary classification.

The function performs the following mathematical operations or procedures:

1. It checks if the feature metadata contains a key called "str2idx". If it does, it means that the output feature is categorical and needs to be converted to a binary representation. 
2. If the feature metadata contains "str2idx", the function calls the `_vectorize_ground_truth` function to convert the ground truth data to a binary representation using the mapping provided by the "str2idx" key in the feature metadata. The `ground_truth_apply_idx` parameter is used to apply this conversion to a specific index or subset of the data.
3. After converting the categorical output feature to a binary representation, the function further converts the category index to a binary representation by comparing it with the positive label. This step assigns a value of `True` to the positive label and `False` to other labels.
4. If the feature metadata does not contain "str2idx", it means that the output feature is binary. In this case, the function checks if the feature metadata contains a key called "str2bool". If it does, it means that the boolean representation of the binary feature is non-standard and needs to be converted.
5. If the feature metadata contains "str2bool", the function calls the `_vectorize_ground_truth` function to convert the ground truth data to a binary representation using the mapping provided by the "str2bool" key in the feature metadata. The `ground_truth_apply_idx` parameter is used to apply this conversion to a specific index or subset of the data.
6. If the feature metadata does not contain "str2bool", it means that the boolean representation of the binary feature is standard. In this case, the function converts the ground truth data to a numpy array using the `.values` attribute.
7. The function then ensures that the positive label is set to 1 for binary features.
8. Finally, the function converts the ground truth data to a 0/1 representation by using the `astype(int)` method and returns the converted ground truth data and the positive label.

The mathematical operations or procedures performed by the function can be summarized as follows:

1. Convert categorical output feature to binary representation if "str2idx" is present in feature metadata.
2. Convert category index to binary representation by comparing it with the positive label.
3. Convert non-standard boolean representation of binary feature to binary representation if "str2bool" is present in feature metadata.
4. Convert standard boolean representation of binary feature to numpy array.
5. Set positive label to 1 for binary features.
6. Convert ground truth data to 0/1 representation using `astype(int)` method.

## Function **`_vectorize_ground_truth`** Overview
The function `_vectorize_ground_truth` takes three parameters: `ground_truth`, `str2idx`, and `ground_truth_apply_idx`. It returns a numpy array.

- `ground_truth` is a pandas Series that represents the ground truth values.
- `str2idx` is a numpy array that maps string values to integer indices.
- `ground_truth_apply_idx` is a boolean parameter that indicates whether the `str2idx` mapping should be applied or not. It is set to `True` by default.

The function first checks if `ground_truth_apply_idx` is `False`. If so, it directly applies a lambda function to the `ground_truth` and `str2idx` arrays using `np.vectorize` and returns the result. This is done to handle cases where the ground truth values are already in the desired format and don't need to be converted using `str2idx`.

If `ground_truth_apply_idx` is `True`, the function tries to vectorize the `_encode_categorical_feature` function using `np.vectorize`. This function takes the `ground_truth` and `str2idx` arrays as inputs and encodes the categorical features using the mapping provided by `str2idx`. If a `KeyError` occurs during this process, it means that some values in `ground_truth` are not present in `str2idx`, and the function falls back to ignoring the `str2idx` mapping and applies the lambda function to `ground_truth` and `str2idx` arrays using `np.vectorize`.

The mathematical operations or procedures performed by this function involve applying a lambda function or `_encode_categorical_feature` function to the `ground_truth` and `str2idx` arrays using `np.vectorize`. The lambda function simply returns the value of `x`, while `_encode_categorical_feature` encodes the categorical features using the mapping provided by `str2idx`.

Here is the LaTex code for the equations involved:

1. Lambda function:

$$
\lambda(x, y) = x
$$

2. `_encode_categorical_feature` function:

$$
_encode_categorical_feature(x, y)
$$

Note: The exact mathematical operations performed by `_encode_categorical_feature` are not provided in the code snippet, so the specific equations cannot be generated.

## Function **`validate_conf_thresholds_and_probabilities_2d_3d`** Overview
The purpose of the function `validate_conf_thresholds_and_probabilities_2d_3d` is to validate that the input arrays `probabilities` and `threshold_output_feature_names` have exactly two members each. If either of the arrays does not have two members, a `RuntimeError` is raised.

The function takes two parameters:
1. `probabilities`: A list of probabilities per model.
2. `threshold_output_feature_names`: A list of threshold output feature names per model.

The function performs the following steps:
1. Creates a dictionary `validation_mapping` with keys "probabilities" and "threshold_output_feature_names" and values `probabilities` and `threshold_output_feature_names` respectively.
2. Iterates over each key-value pair in `validation_mapping`.
3. For each key-value pair, it checks the length of the value.
4. If the length is not equal to 2, it raises a `RuntimeError` with an error message indicating the expected length and the actual length of the value.

The mathematical operations or procedures performed by this function are minimal and do not involve any complex calculations.

## Function **`load_data_for_viz`** Overview
The `load_data_for_viz` function is used to load JSON files containing model experiment statistics for a list of models. It takes several parameters:

- `load_type`: This parameter specifies the type of data loader to be used. It can take two values: "load_json" or "load_from_file".
- `model_file_statistics`: This parameter can be a JSON file or a list of JSON files containing the model experiment statistics.
- `dtype`: This parameter specifies the data type to be used when loading the files. It has a default value of `int`.
- `ground_truth_split`: This parameter specifies the ground truth split to be used when loading the files. It has a default value of `2`.

The function returns a list of training statistics loaded as JSON objects.

The function first creates a dictionary `supported_load_types` that maps the `load_type` parameter to the corresponding data loader function. The two supported load types are "load_json" and "load_from_file". The `load_json` function is a separate function that loads JSON files, while the `load_from_file` function is a partial function that loads files with the specified data type and ground truth split.

The function then selects the appropriate data loader function based on the `load_type` parameter.

Next, the function tries to load the training statistics from the JSON file(s) specified in the `model_file_statistics` parameter. It iterates over each file and uses the selected data loader function to load the statistics. The loaded statistics are stored in a list called `stats_per_model`.

If there is an error while loading the statistics (e.g., the file cannot be opened), an exception is raised with an error message.

Finally, the function returns the `stats_per_model` list containing the loaded training statistics.

There are no mathematical operations or procedures performed in this function.

## Function **`load_training_stats_for_viz`** Overview
The `load_training_stats_for_viz` function is used to load model file data, specifically training statistics, for a list of models. It takes the following parameters:

- `load_type`: The type of data loader to be used.
- `model_file_statistics`: A JSON file or a list of JSON files containing the model experiment statistics.
- `dtype` (optional, default=int): The data type to be used for loading the statistics.
- `ground_truth_split` (optional, default=2): The ground truth split value.

The function returns a list of model statistics loaded as `TrainingStats` objects.

The function first calls the `load_data_for_viz` function to load the data for visualization. This function is not provided in the code snippet, so its purpose and mathematical operations cannot be determined.

Next, the function attempts to load the statistics for each model using the `TrainingStats.Schema().load(j)` method. This method is likely part of a data loading library or framework and is responsible for parsing and loading the JSON data into `TrainingStats` objects. The specific mathematical operations or procedures performed during this loading process cannot be determined without further information about the `TrainingStats` class and its associated schema.

If an exception occurs during the loading process, the function logs an error message and raises the exception.

Finally, the function returns the list of loaded model statistics.

## Function **`convert_to_list`** Overview
The `convert_to_list` function takes an `item` as a parameter and checks if it is an instance of the `list` class or if it is `None`. If the `item` is already a list or `None`, it returns the original `item`. Otherwise, it creates a new list containing the `item` and returns it.

The purpose of this function is to ensure that the input `item` is always returned as a list, even if it is initially not a list or `None`. This can be useful in cases where a function expects a list as input, but the user may provide a single item instead.

The mathematical operations or procedures performed by this function are not applicable, as it is a simple utility function for converting an item to a list. Therefore, there is no need for generating LaTeX code for mathematical equations.

## Function **`_validate_output_feature_name_from_train_stats`** Overview
The purpose of the function `_validate_output_feature_name_from_train_stats` is to validate the prediction `output_feature_name` from the model train stats and return it as a list.

The function takes two parameters:
- `output_feature_name`: This parameter represents the output feature name containing the ground truth.
- `train_stats_per_model`: This parameter is a list of per model train stats.

The function performs the following mathematical operations or procedures:
1. It initializes an empty set called `output_feature_names_set`.
2. It iterates over each `train_stats` in the `train_stats_per_model` list.
3. For each `train_stats`, it iterates over the keys of the `training`, `validation`, and `test` dictionaries.
4. It adds each key to the `output_feature_names_set`.
5. It tries to check if the `output_feature_name` is in the `output_feature_names_set`.
6. If the `output_feature_name` is in the `output_feature_names_set`, it returns a list containing the `output_feature_name`.
7. If the `output_feature_name` is not in the `output_feature_names_set`, it returns the `output_feature_names_set` itself.
8. If the `output_feature_name` is an empty iterable (e.g., `[]` in `set()`), it returns the `output_feature_names_set`.

The mathematical operations or procedures can be represented using LaTeX code as follows:


$$
\text{{output\_feature\_names\_set}} = \{\}
$$


$$
\text{{for }} \text{{train\_stats}} \text{{ in }} \text{{train\_stats\_per\_model}}:
$$


$$
\quad \text{{for }} \text{{key}} \text{{ in }} \text{{itertools.chain(train\_stats.training.keys(), train\_stats.validation.keys(), train\_stats.test.keys())}}:
$$


$$
\quad \quad \text{{output\_feature\_names\_set.add(key)}}
$$


$$
\text{{try:}}
$$


$$
\quad \text{{if }} \text{{output\_feature\_name}} \text{{ in }} \text{{output\_feature\_names\_set}}:
$$


$$
\quad \quad \text{{return }} [\text{{output\_feature\_name}}]
$$


$$
\quad \text{{else:}}
$$


$$
\quad \quad \text{{return }} \text{{output\_feature\_names\_set}}
$$


$$
\text{{except TypeError:}}
$$


$$
\quad \text{{return }} \text{{output\_feature\_names\_set}}
$$

## Function **`_validate_output_feature_name_from_test_stats`** Overview
The function `_validate_output_feature_name_from_test_stats` takes two parameters: `output_feature_name` and `test_stats_per_model`. 

The purpose of the `output_feature_name` parameter is to specify the name of the output feature that contains the ground truth. 

The purpose of the `test_stats_per_model` parameter is to provide a list of per model test statistics. 

The function first creates an empty set called `output_feature_names_set`. 

Then, it iterates over each element `ls` in the `test_stats_per_model` list. 

For each `ls`, it iterates over each key in `ls` and adds it to the `output_feature_names_set`. 

Next, the function checks if the `output_feature_name` is present in the `output_feature_names_set`. If it is, the function returns a list containing only the `output_feature_name`. 

If the `output_feature_name` is not present in the `output_feature_names_set`, the function returns the entire `output_feature_names_set` as a list. 

If the `output_feature_name` is an empty iterable (e.g. `[]` in `set()`), a `TypeError` is raised and the function returns the `output_feature_names_set`. 

Here is the LaTex code to display the equations in a markdown document:


$$
\text{{output\_feature\_names\_set}} = \{\}
$$


$$
\text{{for }} ls \text{{ in }} \text{{test\_stats\_per\_model}}:
$$


$$
\quad \text{{for }} \text{{key}} \text{{ in }} ls:
$$


$$
\quad \quad \text{{output\_feature\_names\_set.add(key)}}
$$


$$
\text{{try:}}
$$


$$
\quad \text{{if }} \text{{output\_feature\_name}} \text{{ in }} \text{{output\_feature\_names\_set}}:
$$


$$
\quad \quad \text{{return }} [\text{{output\_feature\_name}}]
$$


$$
\quad \text{{else:}}
$$


$$
\quad \quad \text{{return }} \text{{output\_feature\_names\_set}}
$$


$$
\text{{except TypeError:}}
$$


$$
\quad \text{{return }} \text{{output\_feature\_names\_set}}
$$

## Function **`_encode_categorical_feature`** Overview
The function `_encode_categorical_feature` takes in two parameters: `raw` and `str2idx`. 

- `raw` is a numpy array that represents the string categorical values that need to be encoded.
- `str2idx` is a dictionary that maps the string representation of the categorical values to their corresponding encoded numeric values.

The purpose of this function is to encode the raw categorical string values to their corresponding encoded numeric values using the provided `str2idx` dictionary.

The function performs the following mathematical operations or procedures:

1. It takes the `raw` categorical string value as input.
2. It uses the `str2idx` dictionary to look up the corresponding encoded numeric value for the input `raw` value.
3. It returns the encoded numeric value.

No mathematical operations are performed in this function. It simply performs a dictionary lookup to encode the categorical values.

## Function **`_get_ground_truth_df`** Overview
The function `_get_ground_truth_df` takes a string parameter `ground_truth` and returns a DataFrame object. 

The purpose of the function is to determine the format of the ground truth data and retrieve it from the source dataset using an appropriate reader. The function first determines the data format by calling the `figure_data_format_dataset` function with the `ground_truth` parameter. It then checks if the data format is supported by checking if it is in the `CACHEABLE_FORMATS` set. If the data format is not supported, a `ValueError` is raised.

Next, the function retrieves the appropriate reader for the data format by calling the `get_from_registry` function with the `data_format` parameter and the `data_reader_registry` as arguments.

If the data format is either "csv" or "tsv", the function calls the reader with the `ground_truth` parameter, `dtype=None`, and `df_lib=pd` to allow type inference. Otherwise, the function calls the reader with the `ground_truth` parameter and `df_lib=pd`.

The function returns the result of the reader function call, which is a DataFrame object containing the ground truth data.

There are no mathematical operations or procedures performed in this function.

## Function **`_extract_ground_truth_values`** Overview
The function `_extract_ground_truth_values` is a helper function that is used to extract ground truth values from a source data set. It takes several parameters:

- `ground_truth`: This parameter can be either a string representing the path to the source data containing the ground truth or a DataFrame object representing the ground truth data itself.
- `output_feature_name`: This parameter is a string representing the name of the output feature for the ground truth values.
- `ground_truth_split`: This parameter is an integer representing the dataset split to use for the ground truth. It defaults to 2.
- `split_file`: This parameter is an optional string representing the file path to split values.

The function first checks if the `ground_truth` parameter is a string or a DataFrame. If it is a string, it calls the `_get_ground_truth_df` function to retrieve the ground truth DataFrame. Otherwise, it assigns the `ground_truth` parameter directly to the `ground_truth_df` variable.

Next, the function checks if the ground truth DataFrame contains a column named `SPLIT`. If it does, it retrieves the split values from the DataFrame and assigns them to the `split` variable. It then selects the ground truth values corresponding to the specified `ground_truth_split` using boolean indexing and assigns them to the `gt` variable.

If the ground truth DataFrame does not contain a `SPLIT` column, the function checks if the `split_file` parameter is provided. If it is, the function checks the file extension of the `split_file`. If it ends with ".csv", it issues a deprecation warning and loads the split values using the `load_array` function. It then creates a boolean mask based on the `ground_truth_split` and assigns the corresponding ground truth values to the `gt` variable.

If the `split_file` parameter is not provided, the function assigns all the data in the `ground_truth_df` DataFrame to the `gt` variable.

Finally, the function returns the `gt` variable, which is a pandas Series containing the extracted ground truth values.

The mathematical operations or procedures performed by this function involve retrieving the ground truth values based on the specified parameters and conditions. There are no specific mathematical operations or equations involved in this function.

## Function **`_get_cols_from_predictions`** Overview
The function `_get_cols_from_predictions` takes three parameters: `predictions_paths`, `cols`, and `metadata`. 

- `predictions_paths` is a list of file paths where the predictions are stored.
- `cols` is a list of column names that need to be extracted from the predictions.
- `metadata` is a dictionary containing metadata information about the features used in the predictions.

The function performs the following operations:

1. It initializes an empty list `results_per_model` to store the extracted columns from each model's predictions.
2. It iterates over each `predictions_path` in the `predictions_paths` list.
3. It reads the predictions data from the `predictions_path` using the `pd.read_parquet` function and assigns it to the `pred_df` variable.
4. It checks if a file with the extension "shapes.json" exists by replacing the file extension of `predictions_path` with "shapes.json" using the `replace_file_extension` function. If the file exists, it loads the JSON data from the file into the `column_shapes` variable using the `load_json` function.
5. It unflattens the `pred_df` DataFrame using the `unflatten_df` function, passing the `pred_df`, `column_shapes`, and `LOCAL_BACKEND.df_engine` as arguments. This step is performed to restore the original shape of the DataFrame if it was flattened during the prediction process.
6. It iterates over each `col` in the `cols` list.
7. If the `col` ends with the `_PREDICTIONS_SUFFIX` (a constant string), it extracts the feature name by removing the `_PREDICTIONS_SUFFIX` from the `col` string. It then retrieves the corresponding feature metadata from the `metadata` dictionary using the feature name.
8. If the feature metadata contains a key "str2idx", it maps the values in the `col` column of the `pred_df` DataFrame to their corresponding indices using a lambda function. The mapping is performed to convert categorical features back to indices.
9. It converts the `pred_df` DataFrame to a numpy dataset using the `to_numpy_dataset` function, passing the `pred_df` and `LOCAL_BACKEND` as arguments. The resulting numpy dataset is assigned back to the `pred_df` variable.
10. It appends the extracted columns from the `pred_df` DataFrame to the `results_per_model` list using a list comprehension.
11. After iterating over all the `predictions_paths`, it returns the `results_per_model` list.

The mathematical operations or procedures performed by this function are not explicitly mentioned in the code. Therefore, there are no specific mathematical equations or procedures to document using LaTeX code.

## Function **`generate_filename_template_path`** Overview
The `generate_filename_template_path` function takes two parameters: `output_dir` and `filename_template`. 

The purpose of the `output_dir` parameter is to specify the directory where the `filename_template` file will be located. 

The purpose of the `filename_template` parameter is to specify the name of the file template that will be appended to the filename template path.

The function first checks if the `output_dir` parameter is not None. If it is not None, it creates the output directory if it does not exist using the `os.makedirs` function with the `exist_ok=True` parameter. 

Then, it returns the path to the filename template inside the output directory by using the `os.path.join` function to concatenate the `output_dir` and `filename_template` parameters.

If the `output_dir` parameter is None, the function returns None.

There are no mathematical operations or procedures performed in this function.

## Function **`compare_performance_cli`** Overview
The `compare_performance_cli` function is a Python function that serves as a command-line interface (CLI) for the `compare_performance` function. It takes in a path to an experiment test statistics file and additional parameters for the requested visualizations.

Parameters:
- `test_statistics`: A string or a list of strings representing the path(s) to the experiment test statistics file(s).
- `**kwargs`: Additional keyword arguments that are passed to the `compare_performance` function.

The function first loads the data from the experiment test statistics file(s) using the `load_data_for_viz` function with the "load_json" method. The loaded data is then passed to the `compare_performance` function along with the additional keyword arguments.

The purpose of this function is to provide a convenient way to compare the performance of different models using the `compare_performance` function through the command-line interface. It abstracts away the data loading process and allows users to specify the test statistics file(s) and visualization parameters as command-line arguments.

The `compare_performance` function is not provided in the code snippet, so the specific mathematical operations or procedures it performs cannot be documented. However, it can be assumed that the `compare_performance` function performs some calculations or visualizations based on the loaded test statistics data to compare the performance of different models.

## Function **`learning_curves_cli`** Overview
The `learning_curves_cli` function is a Python function that loads model data from files and displays learning curves using the `learning_curves` function. 

Parameters:
- `training_statistics` (Union[str, List[str]]): This parameter specifies the path to the experiment training statistics file. It can be either a string representing a single file path or a list of strings representing multiple file paths.
- `kwargs` (dict): This parameter is used to pass additional parameters for the requested visualizations. It is a dictionary that can contain any number of key-value pairs.

The function first calls the `load_training_stats_for_viz` function to load the training statistics data from the specified file(s). The `load_training_stats_for_viz` function is not shown in the code snippet, but it is assumed to return a data structure containing the training statistics for each model.

Once the training statistics data is loaded, the function calls the `learning_curves` function, passing the loaded data and the additional parameters specified in `kwargs`. The `learning_curves` function is not shown in the code snippet, but it is assumed to be a separate function that generates and displays learning curves based on the provided data.

The function does not return any value (`None`).

Mathematical operations or procedures:
The `learning_curves_cli` function does not perform any mathematical operations or procedures. It is mainly responsible for loading the training statistics data and calling the `learning_curves` function to display the learning curves.

## Function **`compare_classifiers_performance_from_prob_cli`** Overview
The `compare_classifiers_performance_from_prob_cli` function is a Python function that compares the performance of different classifiers based on their predicted probabilities. It takes several parameters as input and performs various operations to load the model data and visualize the performance of the classifiers.

Here is a description of each parameter:

- `probabilities`: A list of prediction results file names or a single file name (str) to extract probabilities from.
- `ground_truth`: The path to the ground truth file.
- `ground_truth_split`: The type of ground truth split - `0` for the training split, `1` for the validation split, or `2` for the test split.
- `split_file`: The file path to a CSV file containing split values. This parameter is optional and can be set to `None`.
- `ground_truth_metadata`: The file path to a feature metadata JSON file created during training.
- `output_feature_name`: The name of the output feature to visualize.
- `output_directory`: The name of the output directory containing training results.
- `kwargs`: Additional parameters for the requested visualizations.

The function starts by loading the feature metadata from the `ground_truth_metadata` file using the `load_json` function.

Next, it retrieves the ground truth values from the source dataset using the `_extract_ground_truth_values` function. This function takes the `ground_truth`, `output_feature_name`, `ground_truth_split`, and `split_file` as input parameters.

The function then defines a variable `col` as the concatenation of `output_feature_name` and `_PROBABILITIES_SUFFIX`. This variable is used to specify the column name to extract from the probabilities.

The function calls the `_get_cols_from_predictions` function to extract the probabilities per model. This function takes the `probabilities`, `[col]`, and `metadata` as input parameters.

Finally, the function calls the `compare_classifiers_performance_from_prob` function to compare the performance of the classifiers. This function takes the `probabilities_per_model`, `ground_truth`, `metadata`, `output_feature_name`, `output_directory`, and `kwargs` as input parameters.

The purpose of this function is to provide a command-line interface (CLI) for comparing the performance of classifiers based on their predicted probabilities. It loads the necessary data and calls the appropriate functions to generate the visualizations.

## Function **`compare_classifiers_performance_from_pred_cli`** Overview
The `compare_classifiers_performance_from_pred_cli` function is a Python function that compares the performance of different classifiers based on their prediction results. It takes several parameters as input and performs various operations to load the necessary data and visualize the performance.

Parameters:
- `predictions`: A list of prediction results file names to extract predictions from.
- `ground_truth`: The path to the ground truth file.
- `ground_truth_metadata`: The path to the ground truth metadata file.
- `ground_truth_split`: The type of ground truth split - `0` for training split, `1` for validation split, or `2` for test split.
- `split_file`: The file path to a CSV file containing split values.
- `output_feature_name`: The name of the output feature to visualize.
- `output_directory`: The name of the output directory containing training results.
- `kwargs`: Additional parameters for the requested visualizations.

The function first loads the feature metadata from the `ground_truth_metadata` file using the `load_json` function. This metadata is used to convert raw predictions to encoded values.

Next, it retrieves the ground truth values from the source dataset using the `_extract_ground_truth_values` function. This function takes the `ground_truth`, `output_feature_name`, `ground_truth_split`, and `split_file` as input and returns the corresponding ground truth values.

The function then creates a column name based on the `output_feature_name` and a suffix `_PREDICTIONS_SUFFIX`. It uses this column name to extract the predictions for each model from the `predictions` using the `_get_cols_from_predictions` function. This function takes the `predictions`, a list of column names (in this case, only the `col`), and the metadata as input and returns the predictions for each model.

Finally, the function calls the `compare_classifiers_performance_from_pred` function to compare the performance of the classifiers. It passes the `predictions_per_model`, `ground_truth`, `metadata`, `output_feature_name`, `output_directory`, and `kwargs` as input to this function.

The purpose of this function is to provide a command-line interface (CLI) for comparing the performance of classifiers based on their prediction results. It loads the necessary data, extracts the predictions and ground truth values, and calls the appropriate function for visualization.

## Function **`compare_classifiers_performance_subset_cli`** Overview
The `compare_classifiers_performance_subset_cli` function is used to load model data from files and display the performance of different classifiers on a subset of the data. It takes several parameters:

- `probabilities`: A list of prediction results file names or a single file name as a string. These files contain the predicted probabilities for each class.
- `ground_truth`: The path to the ground truth file, which contains the true labels for the data.
- `ground_truth_split`: The type of ground truth split to consider. It can be `0` for the training split, `1` for the validation split, or `2` for the test split.
- `split_file`: The file path to a CSV file containing split values. This is optional and can be set to `None`.
- `ground_truth_metadata`: The file path to a JSON file containing feature metadata that was created during training.
- `output_feature_name`: The name of the output feature to visualize. This is typically the target variable or the feature being predicted.
- `output_directory`: The name of the output directory where the training results are stored.
- `kwargs`: Additional parameters for the requested visualizations.

The function first loads the feature metadata from the `ground_truth_metadata` file using the `load_json` function. Then, it extracts the ground truth values for the specified output feature, split, and split file using the `_extract_ground_truth_values` function.

Next, it retrieves the predicted probabilities for each model from the `probabilities` files using the `_get_cols_from_predictions` function. The function extracts the column with the name `output_feature_name` followed by the `_PROBABILITIES_SUFFIX` from each file.

Finally, it calls the `compare_classifiers_performance_subset` function with the retrieved probabilities, ground truth, metadata, output feature name, output directory, and any additional parameters specified in `kwargs`.

The function does not return any value (`None`).

## Function **`compare_classifiers_performance_changing_k_cli`** Overview
The `compare_classifiers_performance_changing_k_cli` function is a Python function that loads model data from files and calls the `compare_classifiers_performance_changing_k` function to visualize and compare the performance of different classifiers.

Parameters:
- `probabilities`: A list of prediction results file names or a single file name as a string.
- `ground_truth`: The path to the ground truth file.
- `ground_truth_split`: The type of ground truth split. It can be `0` for the training split, `1` for the validation split, or `2` for the test split.
- `split_file`: The file path to a CSV file containing split values.
- `ground_truth_metadata`: The file path to a feature metadata JSON file created during training.
- `output_feature_name`: The name of the output feature to visualize.
- `output_directory`: The name of the output directory containing training results.
- `**kwargs`: Additional parameters for the requested visualizations.

The function first loads the feature metadata from the `ground_truth_metadata` file using the `load_json` function. Then, it extracts the ground truth values from the source dataset using the `_extract_ground_truth_values` function, passing the `ground_truth`, `output_feature_name`, `ground_truth_split`, and `split_file` as parameters.

Next, the function retrieves the probabilities per model by calling the `_get_cols_from_predictions` function, passing the `probabilities`, `[col]` (where `col` is the output feature name with a suffix), and `metadata` as parameters.

Finally, the function calls the `compare_classifiers_performance_changing_k` function, passing the `probabilities_per_model`, `ground_truth`, `metadata`, `output_feature_name`, `output_directory`, and `**kwargs` as parameters.

The purpose of this function is to provide a command-line interface for comparing the performance of different classifiers by loading the necessary data and calling the appropriate visualization function.

## Function **`compare_classifiers_multiclass_multimetric_cli`** Overview
The `compare_classifiers_multiclass_multimetric_cli` function is a Python function that serves as a command-line interface (CLI) for the `compare_classifiers_multiclass_multimetric` function. It takes in three parameters: `test_statistics`, `ground_truth_metadata`, and `kwargs`.

- `test_statistics` is a path to the experiment test statistics file. It can be either a string representing the path to a single file or a list of strings representing multiple files.
- `ground_truth_metadata` is a path to the ground truth metadata file.
- `kwargs` is a dictionary that contains additional parameters for the requested visualizations.

The function first loads the model data from the test statistics file(s) using the `load_data_for_viz` function with the "load_json" method. The loaded data is stored in the `test_stats_per_model` variable.

Next, it loads the ground truth metadata using the `load_json` function and stores it in the `metadata` variable.

Finally, it calls the `compare_classifiers_multiclass_multimetric` function with the `test_stats_per_model`, `metadata`, and `kwargs` as arguments to perform the desired visualizations.

The function does not return any value (`None`).

No mathematical operations or procedures are performed in this function.

## Function **`compare_classifiers_predictions_cli`** Overview
The `compare_classifiers_predictions_cli` function is a command-line interface (CLI) function that loads model data from files and calls the `compare_classifiers_predictions` function to compare the predictions of different classifiers.

Parameters:
- `predictions` (List[str]): A list of prediction results file names to extract predictions from.
- `ground_truth` (str): The path to the ground truth file.
- `ground_truth_split` (int): The type of ground truth split. `0` for the training split, `1` for the validation split, or `2` for the test split.
- `split_file` (str, None): The file path to a CSV file containing split values. This parameter is optional and can be set to `None`.
- `ground_truth_metadata` (str): The file path to a feature metadata JSON file created during training.
- `output_feature_name` (str): The name of the output feature to visualize.
- `output_directory` (str): The name of the output directory containing training results.
- `**kwargs` (dict): Additional parameters for the requested visualizations.

The function first loads the feature metadata from the `ground_truth_metadata` file using the `load_json` function.

Next, it extracts the ground truth values from the source dataset using the `_extract_ground_truth_values` function, passing the `ground_truth`, `output_feature_name`, `ground_truth_split`, and `split_file` parameters.

Then, it retrieves the predictions for each model from the `predictions` files using the `_get_cols_from_predictions` function, passing the `predictions`, `[col]` (where `col` is the output feature name with a suffix), and `metadata` parameters.

Finally, it calls the `compare_classifiers_predictions` function, passing the `predictions_per_model`, `ground_truth`, `metadata`, `output_feature_name`, and `output_directory` parameters, as well as any additional parameters specified in `kwargs`.

The function does not return any value (`None`).

Mathematical operations or procedures are not performed in this function.

## Function **`compare_classifiers_predictions_distribution_cli`** Overview
The `compare_classifiers_predictions_distribution_cli` function is a command-line interface (CLI) function that loads model data from files and calls the `compare_classifiers_predictions_distribution` function to visualize and compare the predictions of different classifiers.

Parameters:
- `predictions` (List[str]): A list of prediction results file names to extract predictions from.
- `ground_truth` (str): The path to the ground truth file.
- `ground_truth_split` (int): The type of ground truth split. `0` for the training split, `1` for the validation split, or `2` for the test split.
- `split_file` (str, None): The file path to a CSV file containing split values. This parameter is optional and can be set to `None`.
- `ground_truth_metadata` (str): The file path to the feature metadata JSON file created during training.
- `output_feature_name` (str): The name of the output feature to visualize.
- `output_directory` (str): The name of the output directory containing training results.
- `**kwargs` (dict): Additional parameters for the requested visualizations.

The function first loads the feature metadata from the `ground_truth_metadata` file using the `load_json` function. Then, it extracts the ground truth values from the source dataset using the `_extract_ground_truth_values` function, passing the `ground_truth`, `output_feature_name`, `ground_truth_split`, and `split_file` parameters.

Next, the function retrieves the predictions for each model from the `predictions` files using the `_get_cols_from_predictions` function. It extracts the column with the name `output_feature_name` followed by the `_PREDICTIONS_SUFFIX` suffix. The `metadata` is passed to convert the raw predictions to encoded values.

Finally, the function calls the `compare_classifiers_predictions_distribution` function, passing the `predictions_per_model`, `ground_truth`, `metadata`, `output_feature_name`, `output_directory`, and any additional parameters specified in `kwargs`. This function visualizes and compares the predictions of different classifiers.

Mathematical Operations:
The `compare_classifiers_predictions_distribution_cli` function does not perform any mathematical operations. It mainly handles file loading, data extraction, and calls the `compare_classifiers_predictions_distribution` function for visualization.

## Function **`confidence_thresholding_cli`** Overview
The `confidence_thresholding_cli` function is used to load model data from files and display the results using the `confidence_thresholding` function. 

Parameters:
- `probabilities`: A list of prediction results file names or a single file name as a string.
- `ground_truth`: The path to the ground truth file.
- `ground_truth_split`: The type of ground truth split. It can be `0` for the training split, `1` for the validation split, or `2` for the test split.
- `split_file`: The file path to a CSV file containing split values. It can be `None` if not applicable.
- `ground_truth_metadata`: The file path to a feature metadata JSON file created during training.
- `output_feature_name`: The name of the output feature to visualize.
- `output_directory`: The name of the output directory containing training results.
- `kwargs`: Additional parameters for the requested visualizations.

The function first loads the feature metadata from the `ground_truth_metadata` file using the `load_json` function. Then, it extracts the ground truth values from the source dataset using the `_extract_ground_truth_values` function, passing the `ground_truth`, `output_feature_name`, `ground_truth_split`, and `split_file` as parameters.

Next, it retrieves the probabilities per model using the `_get_cols_from_predictions` function, passing the `probabilities`, `[col]` (where `col` is the output feature name with a suffix), and `metadata` as parameters.

Finally, it calls the `confidence_thresholding` function, passing the `probabilities_per_model`, `ground_truth`, `metadata`, `output_feature_name`, `output_directory`, and `kwargs` as parameters.

The function does not return any value (`None`).

Mathematical operations or procedures:
- No specific mathematical operations or procedures are performed in this function. It mainly involves loading data, extracting values, and calling other functions for visualization purposes.

## Function **`confidence_thresholding_data_vs_acc_cli`** Overview
The `confidence_thresholding_data_vs_acc_cli` function is used to load model data from files and display it using the `confidence_thresholding_data_vs_acc` function. 

Parameters:
- `probabilities`: A list of prediction results file names or a single file name as a string.
- `ground_truth`: The path to the ground truth file.
- `ground_truth_split`: The type of ground truth split. It can be `0` for the training split, `1` for the validation split, or `2` for the test split.
- `split_file`: The file path to a CSV file containing split values.
- `ground_truth_metadata`: The file path to a feature metadata JSON file created during training.
- `output_feature_name`: The name of the output feature to visualize.
- `output_directory`: The name of the output directory containing training results.
- `**kwargs`: Additional parameters for the requested visualizations.

The function first loads the feature metadata from the `ground_truth_metadata` file using the `load_json` function. Then, it extracts the ground truth values from the source dataset using the `_extract_ground_truth_values` function. 

Next, it retrieves the probabilities per model by calling the `_get_cols_from_predictions` function with the `probabilities` list, the `col` name (constructed from `output_feature_name` and `_PROBABILITIES_SUFFIX`), and the metadata. 

Finally, it calls the `confidence_thresholding_data_vs_acc` function with the retrieved probabilities per model, the ground truth values, the metadata, the output feature name, the output directory, and any additional parameters specified in `kwargs`.

## Function **`confidence_thresholding_data_vs_acc_subset_cli`** Overview
The `confidence_thresholding_data_vs_acc_subset_cli` function is a Python function that loads model data from files and calls the `confidence_thresholding_data_vs_acc_subset` function to visualize the data.

Parameters:
- `probabilities`: A list of prediction results file names or a single file name as a string.
- `ground_truth`: The path to the ground truth file.
- `ground_truth_split`: The type of ground truth split. It can be `0` for the training split, `1` for the validation split, or `2` for the test split.
- `split_file`: The file path to a CSV file containing split values. It can be `None` if not applicable.
- `ground_truth_metadata`: The file path to a JSON file containing feature metadata created during training.
- `output_feature_name`: The name of the output feature to visualize.
- `output_directory`: The name of the output directory containing training results.
- `**kwargs`: Additional parameters for the requested visualizations.

The function first loads the feature metadata from the `ground_truth_metadata` file using the `load_json` function. Then, it extracts the ground truth values from the source dataset using the `_extract_ground_truth_values` function, passing the `ground_truth`, `output_feature_name`, `ground_truth_split`, and `split_file` as parameters.

Next, it retrieves the probabilities per model by calling the `_get_cols_from_predictions` function, passing the `probabilities`, `[col]` (where `col` is the output feature name with a suffix), and `metadata` as parameters.

Finally, it calls the `confidence_thresholding_data_vs_acc_subset` function, passing the `probabilities_per_model`, `ground_truth`, `metadata`, `output_feature_name`, `output_directory`, and `**kwargs` as parameters to visualize the data.

The function does not return any value (`None`).

Mathematical operations or procedures are not performed in this function.

## Function **`confidence_thresholding_data_vs_acc_subset_per_class_cli`** Overview
The `confidence_thresholding_data_vs_acc_subset_per_class_cli` function is a Python function that loads model data from files and calls the `confidence_thresholding_data_vs_acc_subset_per_class` function to visualize the results.

Parameters:
- `probabilities`: A list of prediction results file names or a single file name as a string.
- `ground_truth`: The path to the ground truth file.
- `ground_truth_metadata`: The path to the ground truth metadata file.
- `ground_truth_split`: The type of ground truth split. It can be `0` for the training split, `1` for the validation split, or `2` for the test split.
- `split_file`: The file path to a CSV file containing split values.
- `output_feature_name`: The name of the output feature to visualize.
- `output_directory`: The name of the output directory containing training results.
- `**kwargs`: Additional parameters for the requested visualizations.

The function first loads the feature metadata from the ground truth metadata file using the `load_json` function. Then, it extracts the ground truth values based on the output feature name, ground truth split, and split file using the `_extract_ground_truth_values` function.

Next, it retrieves the probabilities per model by calling the `_get_cols_from_predictions` function with the probabilities, the column name based on the output feature name, and the metadata.

Finally, it calls the `confidence_thresholding_data_vs_acc_subset_per_class` function with the probabilities per model, ground truth, metadata, output feature name, output directory, and any additional parameters specified in `kwargs`.

The purpose of this function is to provide a command-line interface for visualizing the results of confidence thresholding on a per-class basis. It takes input files, extracts the necessary data, and calls the appropriate function to generate the visualization.

## Function **`confidence_thresholding_2thresholds_2d_cli`** Overview
The `confidence_thresholding_2thresholds_2d_cli` function is used to load model data from files and display the results using the `confidence_thresholding_2thresholds_2d` function. 

Parameters:
- `probabilities`: A list of prediction results file names or a single file name as a string.
- `ground_truth`: The path to the ground truth file.
- `ground_truth_split`: The type of ground truth split. It can be `0` for the training split, `1` for the validation split, or `2` for the test split.
- `split_file`: The file path to a CSV file containing split values.
- `ground_truth_metadata`: The file path to a feature metadata JSON file created during training.
- `threshold_output_feature_names`: A list of names of the output features to visualize.
- `output_directory`: The name of the output directory containing the training results.
- `**kwargs`: Additional parameters for the requested visualizations.

The function performs the following operations:

1. Loads the feature metadata from the `ground_truth_metadata` file.
2. Extracts the ground truth values for the first and second output features specified in `threshold_output_feature_names` using the `_extract_ground_truth_values` function.
3. Retrieves the columns corresponding to the output features from the prediction files specified in `probabilities` using the `_get_cols_from_predictions` function.
4. Calls the `confidence_thresholding_2thresholds_2d` function with the extracted ground truth values, probabilities per model, metadata, output feature names, and other parameters to display the results.

The function does not return any value.

## Function **`confidence_thresholding_2thresholds_3d_cli`** Overview
The `confidence_thresholding_2thresholds_3d_cli` function is used to load model data from files and display it using the `confidence_thresholding_2thresholds_3d` function. Here is a breakdown of the parameters and their purposes:

- `probabilities`: A list of prediction results file names or a single file name as a string. These files contain the predicted probabilities for each class.
- `ground_truth`: The path to the ground truth file. This file contains the true labels for the data.
- `ground_truth_split`: An integer representing the type of ground truth split. `0` for the training split, `1` for the validation split, or `2` for the test split.
- `split_file`: The file path to a CSV file containing split values. This file is used to split the data into different sets.
- `ground_truth_metadata`: The file path to a JSON file containing feature metadata. This file is created during training and is used to convert raw predictions to encoded values.
- `threshold_output_feature_names`: A list of output feature names to visualize. These are the features for which the confidence thresholds will be applied.
- `output_directory`: The name of the output directory where the training results will be stored.
- `kwargs`: Additional parameters for the requested visualizations.

The function performs the following mathematical operations or procedures:

1. It loads the feature metadata from the `ground_truth_metadata` file.
2. It extracts the ground truth values for the first and second output features using the `_extract_ground_truth_values` function.
3. It constructs a list of column names for the predicted probabilities based on the `threshold_output_feature_names`.
4. It retrieves the predicted probabilities per model using the `_get_cols_from_predictions` function.
5. It calls the `confidence_thresholding_2thresholds_3d` function with the retrieved probabilities, ground truth values, metadata, output feature names, and other parameters.

The `confidence_thresholding_2thresholds_3d` function is responsible for visualizing the data using confidence thresholds. The specific mathematical operations performed by this function are not described in the provided code snippet.

## Function **`binary_threshold_vs_metric_cli`** Overview
The `binary_threshold_vs_metric_cli` function is used to load model data from files and visualize the binary threshold vs metric plot. Here is a breakdown of the function and its parameters:

Parameters:
- `probabilities`: A list of prediction results file names or a single file name as a string.
- `ground_truth`: The path to the ground truth file.
- `ground_truth_split`: The type of ground truth split. It can be `0` for the training split, `1` for the validation split, or `2` for the test split.
- `split_file`: The file path to a CSV file containing split values. It can be `None` if not applicable.
- `ground_truth_metadata`: The file path to the feature metadata JSON file created during training.
- `output_feature_name`: The name of the output feature to visualize.
- `output_directory`: The name of the output directory containing training results.
- `**kwargs`: Additional parameters for the requested visualizations.

The function first loads the feature metadata from the `ground_truth_metadata` file using the `load_json` function. Then, it extracts the ground truth values from the source dataset using the `_extract_ground_truth_values` function, passing the `ground_truth`, `output_feature_name`, `ground_truth_split`, and `split_file` parameters.

Next, it retrieves the column name for the output feature probabilities by appending the `_PROBABILITIES_SUFFIX` to the `output_feature_name`. It then calls the `_get_cols_from_predictions` function to extract the probabilities per model from the `probabilities` files, passing the list of column names and the metadata.

Finally, it calls the `binary_threshold_vs_metric` function to visualize the binary threshold vs metric plot, passing the probabilities per model, ground truth, metadata, output feature name, output directory, and any additional parameters specified in `kwargs`.

The function does not return any value (`None`).

Mathematical operations or procedures:
- No specific mathematical operations or procedures are performed in this function. It mainly focuses on loading data, extracting values, and calling other functions for visualization.

## Function **`precision_recall_curves_cli`** Overview
The `precision_recall_curves_cli` function is used to load model data from files and display precision-recall curves for binary classification tasks. Here is a breakdown of its parameters and operations:

Parameters:
- `probabilities`: A list of prediction results file names or a single file name as a string. These files contain the predicted probabilities for the positive class.
- `ground_truth`: The path to the ground truth file, which contains the true labels for the data.
- `ground_truth_split`: An integer indicating the type of ground truth split. `0` represents the training split, `1` represents the validation split, and `2` represents the test split.
- `split_file`: The file path to a CSV file containing split values. This parameter is optional and can be set to `None`.
- `ground_truth_metadata`: The file path to a JSON file containing feature metadata that was created during training.
- `output_feature_name`: The name of the output feature to visualize. This is typically the target variable or the feature being predicted.
- `output_directory`: The name of the output directory where the training results are stored.
- `**kwargs`: Additional parameters for the requested visualizations.

Operations:
1. The function first loads the feature metadata from the `ground_truth_metadata` file using the `load_json` function.
2. It then extracts the ground truth values from the `ground_truth` file using the `_extract_ground_truth_values` function, passing the `output_feature_name`, `ground_truth_split`, and `split_file` as arguments.
3. The function retrieves the column name for the predicted probabilities by appending the `_PROBABILITIES_SUFFIX` to the `output_feature_name`.
4. It uses the `_get_cols_from_predictions` function to extract the predicted probabilities from the `probabilities` files, passing the column name and the metadata as arguments.
5. Finally, the function calls the `precision_recall_curves` function, passing the extracted probabilities, ground truth values, metadata, `output_feature_name`, `output_directory`, and any additional parameters specified in `kwargs`.

The `precision_recall_curves` function is responsible for generating the precision-recall curves based on the provided data. The specific mathematical operations or procedures performed within this function are not described in the given code snippet.

## Function **`roc_curves_cli`** Overview
The `roc_curves_cli` function is used to load model data from files and display ROC curves for binary classification models. Here is a breakdown of the function's parameters and their purposes:

- `probabilities`: A list of prediction results file names or a single file name as a string. These files contain the predicted probabilities for the positive class.
- `ground_truth`: The path to the ground truth file. This file contains the true labels for the data.
- `ground_truth_split`: An integer representing the type of ground truth split. `0` indicates the training split, `1` indicates the validation split, and `2` indicates the test split.
- `split_file`: The file path to a CSV file containing split values. This file is used to split the ground truth data into training, validation, and test sets.
- `ground_truth_metadata`: The file path to a JSON file containing feature metadata that was created during training. This metadata is used to convert raw predictions to encoded values.
- `output_feature_name`: The name of the output feature to visualize. This is the feature for which the ROC curves will be generated.
- `output_directory`: The name of the output directory where the training results will be stored.
- `**kwargs`: Additional parameters for the requested visualizations.

The function first loads the feature metadata from the `ground_truth_metadata` file using the `load_json` function. Then, it extracts the ground truth values for the specified output feature, split, and split file using the `_extract_ground_truth_values` function.

Next, the function retrieves the predicted probabilities for each model from the `probabilities` files using the `_get_cols_from_predictions` function. The probabilities are extracted for the specified output feature and converted to encoded values using the metadata.

Finally, the `roc_curves` function is called with the extracted probabilities, ground truth values, metadata, output feature name, and other optional parameters. This function generates the ROC curves and saves them in the specified output directory.

Here is the LaTex code for the mathematical operations performed in the function:


$$
\text{{col}} = \text{{output\_feature\_name}} + \text{{\_PROBABILITIES\_SUFFIX}}
$$


$$
\text{{probabilities\_per\_model}} = \text{{\_get\_cols\_from\_predictions}}(\text{{probabilities}}, [\text{{col}}], \text{{metadata}})
$$


$$
\text{{roc\_curves}}(\text{{probabilities\_per\_model}}, \text{{ground\_truth}}, \text{{metadata}}, \text{{output\_feature\_name}}, \text{{output\_directory}}, \ldots)
$$

## Function **`roc_curves_from_test_statistics_cli`** Overview
The `roc_curves_from_test_statistics_cli` function is a Python function that serves as a command-line interface (CLI) for generating ROC curves from test statistics. It takes in the following parameters:

- `test_statistics` (Union[str, List[str]]): This parameter specifies the path to the experiment test statistics file. It can be either a string representing a single file path or a list of strings representing multiple file paths.

- `**kwargs` (dict): This parameter allows for additional parameters to be passed to the function. These parameters are used for the requested visualizations.

The function first loads the model data from the test statistics file(s) using the `load_data_for_viz` function with the "load_json" option. The loaded data is stored in the `test_stats_per_model` variable.

Finally, the function calls the `roc_curves_from_test_statistics` function, passing in the `test_stats_per_model` variable and the additional parameters specified in `**kwargs`.

The purpose of this function is to provide a convenient way to generate ROC curves from test statistics through a command-line interface. It abstracts away the details of loading the data and calling the `roc_curves_from_test_statistics` function, making it easier to use for users who want to generate ROC curves from test statistics.

## Function **`precision_recall_curves_from_test_statistics_cli`** Overview
The `precision_recall_curves_from_test_statistics_cli` function is a command-line interface (CLI) function that is used to load model data from files and display precision-recall curves based on the test statistics.

Parameters:
- `test_statistics` (Union[str, List[str]]): This parameter specifies the path to the experiment test statistics file. It can be a single string or a list of strings if multiple files need to be loaded.
- `**kwargs` (dict): This parameter allows for additional parameters to be passed to the function for the requested visualizations.

The function first calls the `load_data_for_viz` function to load the test statistics data from the file(s) specified by `test_statistics`. The `load_data_for_viz` function is not shown in the code snippet, but it is assumed to be a helper function that loads the data and returns it in a suitable format.

Once the test statistics data is loaded, the function calls the `precision_recall_curves_from_test_statistics` function to generate and display the precision-recall curves based on the loaded data. The `**kwargs` parameter is passed to this function to provide any additional parameters that were specified.

The function does not return any value (`None`) as indicated by the `-> None` in the function signature. It is assumed that the precision-recall curves are displayed directly within the function or through some other means not shown in the code snippet.

The mathematical operations or procedures performed by this function are not explicitly shown in the code snippet. However, based on the function name and the assumption that the `precision_recall_curves_from_test_statistics` function is responsible for generating the curves, it can be inferred that the function involves calculations related to precision and recall. The specific mathematical operations or procedures would be implemented in the `precision_recall_curves_from_test_statistics` function, which is not shown in the code snippet.

## Function **`calibration_1_vs_all_cli`** Overview
The `calibration_1_vs_all_cli` function is used to load model data from files and display calibration plots for a binary classification problem. Here is a breakdown of the function's parameters and their purposes:

- `probabilities`: A list of file names or a single file name containing the predicted probabilities for each class. These probabilities are used to generate the calibration plots.
- `ground_truth`: The path to the ground truth file, which contains the true labels for the data.
- `ground_truth_split`: An integer indicating the type of ground truth split. `0` represents the training split, `1` represents the validation split, and `2` represents the test split.
- `split_file`: The path to a CSV file containing split values. This file is used to split the ground truth data into training, validation, and test sets.
- `ground_truth_metadata`: The path to a JSON file containing metadata about the ground truth features. This metadata is used to convert raw predictions to encoded values.
- `output_feature_name`: The name of the output feature to visualize. This feature is the target variable in the binary classification problem.
- `output_directory`: The name of the output directory where the calibration plots will be saved.
- `output_feature_proc_name` (optional): The name of the output feature column in the ground truth file. If the ground truth file is a preprocessed Parquet or HDF5 file, the column name will be `<output_feature>_<hash>`.
- `ground_truth_apply_idx` (optional): A boolean indicating whether to use the metadata's `str2idx` mapping in `np.vectorize` when vectorizing the ground truth values.
- `kwargs` (optional): Additional parameters for the requested visualizations.

The function performs the following mathematical operations or procedures:

1. It loads the feature metadata from the `ground_truth_metadata` file.
2. It extracts the ground truth values from the `ground_truth` file using the `_extract_ground_truth_values` function.
3. It vectorizes the ground truth values using the `_vectorize_ground_truth` function, which converts the ground truth labels to encoded values based on the feature metadata.
4. It retrieves the predicted probabilities for each model from the `probabilities` files using the `_get_cols_from_predictions` function.
5. It calls the `calibration_1_vs_all` function to generate the calibration plots using the probabilities, ground truth values, metadata, output feature name, output directory, and any additional parameters specified in `kwargs`.

The mathematical operations or procedures performed by the `calibration_1_vs_all` function are not explicitly described in the code provided.

## Function **`calibration_multiclass_cli`** Overview
The `calibration_multiclass_cli` function is used to load model data from files and display the calibration results for a multiclass classification problem. 

Parameters:
- `probabilities`: A list of prediction results file names or a single file name as a string. These files contain the predicted probabilities for each class.
- `ground_truth`: The path to the ground truth file, which contains the true labels for the data.
- `ground_truth_split`: The type of ground truth split. It can be `0` for the training split, `1` for the validation split, or `2` for the test split.
- `split_file`: The file path to a CSV file containing split values. This is an optional parameter and can be set to `None` if not needed.
- `ground_truth_metadata`: The file path to a JSON file containing feature metadata created during training.
- `output_feature_name`: The name of the output feature to visualize. This is the feature for which the calibration results will be displayed.
- `output_directory`: The name of the output directory containing the training results.
- `**kwargs`: Additional parameters for the requested visualizations.

The function first loads the feature metadata from the `ground_truth_metadata` file using the `load_json` function. Then, it extracts the ground truth values for the specified output feature, split, and split file using the `_extract_ground_truth_values` function.

Next, it retrieves the predicted probabilities for each model from the `probabilities` files using the `_get_cols_from_predictions` function. The function extracts the probabilities for the specified output feature and stores them in the `probabilities_per_model` variable.

Finally, the `calibration_multiclass` function is called with the extracted probabilities, ground truth values, metadata, output feature name, output directory, and any additional parameters specified in `kwargs`. This function performs the calibration calculations and displays the calibration results.

The mathematical operations or procedures performed by the `calibration_multiclass_cli` function involve loading data from files, extracting relevant information, and passing it to the `calibration_multiclass` function for further processing. There are no specific mathematical operations or equations performed within this function.

## Function **`confusion_matrix_cli`** Overview
The `confusion_matrix_cli` function is a Python function that loads model data from files and displays a confusion matrix. It takes three parameters: `test_statistics`, `ground_truth_metadata`, and `kwargs`.

The purpose of each parameter is as follows:

- `test_statistics`: This parameter can be either a string or a list of strings. It represents the path to the experiment test statistics file(s). If it is a string, it represents the path to a single file. If it is a list of strings, it represents the paths to multiple files.

- `ground_truth_metadata`: This parameter is a string that represents the path to the ground truth metadata file. This file contains information about the ground truth data.

- `kwargs`: This parameter is a dictionary that contains additional parameters for the requested visualizations. The specific parameters depend on the visualization being used.

The function first calls the `load_data_for_viz` function to load the test statistics data from the file(s) specified by the `test_statistics` parameter. The `load_data_for_viz` function is not shown in the code snippet, but it is assumed to be a helper function that loads data from a file using a specified method (in this case, "load_json").

Next, the function calls the `load_json` function to load the ground truth metadata from the file specified by the `ground_truth_metadata` parameter. The `load_json` function is also assumed to be a helper function that loads JSON data from a file.

Finally, the function calls the `confusion_matrix` function, passing the loaded test statistics data, the loaded metadata, and the additional parameters specified in `kwargs`. The `confusion_matrix` function is not defined in the code snippet, but it is assumed to be a separate function that generates and displays a confusion matrix based on the provided data.

The function does not return any value. It simply performs the necessary operations to load the data and display the confusion matrix.

## Function **`frequency_vs_f1_cli`** Overview
The `frequency_vs_f1_cli` function is a Python function that serves as a command-line interface for the `frequency_vs_f1` function. It takes in three parameters: `test_statistics`, `ground_truth_metadata`, and `kwargs`.

- `test_statistics` is a path to an experiment test statistics file. It can be either a string representing a single file path or a list of strings representing multiple file paths.
- `ground_truth_metadata` is a path to a ground truth metadata file.
- `kwargs` is a dictionary that contains additional parameters for the requested visualizations.

The function first loads the model data from the files specified by `test_statistics` and `ground_truth_metadata` using the `load_data_for_viz` and `load_json` functions, respectively. The loaded data is then passed to the `frequency_vs_f1` function along with the additional parameters specified in `kwargs`.

The purpose of this function is to provide a convenient command-line interface for generating visualizations using the `frequency_vs_f1` function. It abstracts away the data loading process and allows users to specify the necessary files and parameters through command-line arguments.

Here is the LaTex code for the mathematical operations or procedures performed by this function:


$$
\text{{test\_stats\_per\_model}} = \text{{load\_data\_for\_viz}}("load\_json", \text{{test\_statistics}})
$$


$$
\text{{metadata}} = \text{{load\_json}}(\text{{ground\_truth\_metadata}})
$$


$$
\text{{frequency\_vs\_f1}}(\text{{test\_stats\_per\_model}}, \text{{metadata}}, **\text{{kwargs}})
$$

## Function **`learning_curves`** Overview
The `learning_curves` function takes in several parameters and generates learning curves for each model and output feature.

Parameters:
- `train_stats_per_model`: A list containing dictionaries of training statistics per model.
- `output_feature_name`: The name of the output feature to use for the visualization. If `None`, all output features are used.
- `model_names`: The model name or a list of model names to use as labels.
- `output_directory`: The directory where to save the plots. If not specified, plots will be displayed in a window.
- `file_format`: The file format of the output plots, either `'pdf'` or `'png'`.
- `callbacks`: A list of `ludwig.callbacks.Callback` objects that provide hooks into the Ludwig pipeline.

The function first generates a filename template based on the output feature name and metric, and the specified file format. It then converts the `train_stats_per_model` and `model_names` parameters into lists if they are not already. The output feature names are validated based on the training statistics.

The function then iterates over each output feature name and metric, and checks if the metric exists in the training statistics. If it does, the function generates a filename based on the template, extracts the training and validation statistics for the metric, and retrieves the evaluation frequency.

Finally, the function calls the `learning_curves_plot` function from the `visualization_utils` module, passing in the training and validation statistics, metric, evaluation frequency, model names, and other parameters. This function generates the learning curves plot for the specified metric and output feature.

The function does not return any value.

## Function **`compare_performance`** Overview
The `compare_performance` function takes in several parameters and produces a bar plot visualization comparing the performance of different models based on evaluation statistics.

Parameters:
- `test_stats_per_model` (List[dict]): A list of dictionaries containing evaluation performance statistics for each model.
- `output_feature_name` (Union[str, None], default: None): The name of the output feature to use for the visualization. If None, all output features are used.
- `model_names` (Union[str, List[str]], default: None): The model name or list of model names to use as labels.
- `output_directory` (str, default: None): The directory where the plots will be saved. If not specified, the plots will be displayed in a window.
- `file_format` (str, default: 'pdf'): The file format of the output plots - 'pdf' or 'png'.
- `**kwargs`: Additional keyword arguments.

The function first defines a list of names to ignore, such as "overall_stats", "confusion_matrix", etc. Then, it generates a filename template for the output plots based on the specified file format and output directory.

Next, it converts the input parameters `test_stats_per_model` and `model_names` into lists if they are not already. It also validates the output feature name from the test statistics.

The function then iterates over each output feature name and performs the following steps:
1. Collects the unique metric names available in the test statistics for the current output feature.
2. Removes the "LOSS" metric name and any ignored metric names from the list of metric names.
3. Creates an empty dictionary `metrics_dict` to store the metric values for each model.
4. Iterates over each test statistics dictionary in `test_stats_per_model_list` and appends the metric values to `metrics_dict`.
5. Checks if there are any metrics to compare in `metrics_dict`.
6. If there are metrics, it creates empty lists `metrics` and `metrics_names` to store the metric values and names, respectively. It also initializes `min_val` and `max_val` variables to track the minimum and maximum metric values.
7. Iterates over each metric name and its corresponding values in `metrics_dict` and appends them to `metrics` and `metrics_names`. It also updates `min_val` and `max_val` if necessary.
8. If a filename template path is specified, it creates the filename for the current output feature.
9. Calls the `visualization_utils.compare_classifiers_plot` function to generate the bar plot visualization, passing in the metrics, metric names, model names, and other parameters.

The function does not return any value (`None`).

Example usage:
```python
model_a = LudwigModel(config)
model_a.train(dataset)
a_evaluation_stats, _, _ = model_a.evaluate(eval_set)
model_b = LudwigModel.load("path/to/model/")
b_evaluation_stats, _, _ = model_b.evaluate(eval_set)
compare_performance([a_evaluation_stats, b_evaluation_stats], model_names=["A", "B"])
```

The `compare_performance` function is used to compare the performance of two models (`model_a` and `model_b` in the example) based on their evaluation statistics and generate a bar plot visualization.

## Function **`compare_classifiers_performance_from_prob`** Overview
The function `compare_classifiers_performance_from_prob` takes in several parameters and produces a bar plot visualization comparing the performance of different classifiers based on their predicted probabilities.

Parameters:
- `probabilities_per_model` (List[np.ndarray]): A list of numpy arrays containing the predicted probabilities for each model.
- `ground_truth` (Union[pd.Series, np.ndarray]): The ground truth values.
- `metadata` (dict): A dictionary containing metadata about the features.
- `output_feature_name` (str): The name of the output feature.
- `top_n_classes` (Union[List[int], int]): A list or integer specifying the number of classes to plot.
- `labels_limit` (int): An upper limit on the numeric encoded label value. Labels higher than this limit are considered "rare" labels.
- `model_names` (Union[str, List[str]]): The name or list of names of the models to use as labels.
- `output_directory` (str): The directory where the plots will be saved. If not specified, the plots will be displayed in a window.
- `file_format` (str): The file format of the output plots (either "pdf" or "png").
- `ground_truth_apply_idx` (bool): Whether to use the metadata['str2idx'] in np.vectorize.

The function performs the following mathematical operations or procedures:

1. If the `ground_truth` is not a numpy array, it assumes that the raw values need to be translated to encoded values using the metadata.
2. Converts `top_n_classes` and `model_names` to lists if they are not already.
3. If `labels_limit` is greater than 0, it sets all label values in `ground_truth` that are higher than `labels_limit` to `labels_limit`.
4. Assigns the `probabilities_per_model` to the variable `probs`.
5. Initializes empty lists `accuracies`, `hits_at_ks`, and `mrrs` to store the computed metrics for each model.
6. Iterates over each model's predicted probabilities:
   - If `labels_limit` is greater than 0 and the number of classes in the probabilities is greater than `labels_limit + 1`, it limits the probabilities to the first `labels_limit + 1` classes and sums the probabilities of the remaining classes into the last class.
   - Sorts the probabilities in ascending order and gets the top 1 prediction and top k predictions.
   - Computes the accuracy by comparing the top 1 prediction with the ground truth and calculates the proportion of correct predictions.
   - Computes the hits@k metric by counting the number of ground truth labels that are present in the top k predictions and calculates the proportion.
   - Computes the mean reciprocal rank (MRR) by finding the position of the ground truth label in the sorted probabilities and calculates the reciprocal rank.
7. If `output_directory` is specified, it creates the directory if it doesn't exist and sets the filename for the plot.
8. Calls the `compare_classifiers_plot` function from the `visualization_utils` module to generate the bar plot visualization, passing in the computed metrics, metric names, model names, and the filename for saving the plot.

The mathematical operations are as follows:

- Accuracy: \(\text{{accuracy}} = \frac{{\text{{number of correct predictions}}}}{{\text{{total number of predictions}}}}\)
- Hits@k: \(\text{{hits@k}} = \frac{{\text{{number of ground truth labels in top k predictions}}}}{{\text{{total number of predictions}}}}\)
- Mean Reciprocal Rank (MRR): \(\text{{MRR}} = \frac{{1}}{{\text{{rank of the first correct prediction}}}}\)

## Function **`compare_classifiers_performance_from_pred`** Overview
The function `compare_classifiers_performance_from_pred` takes in several parameters and produces a bar plot visualization comparing the performance of different classifiers based on their predictions.

Parameters:
- `predictions_per_model`: A list of numpy arrays containing the predictions made by each model.
- `ground_truth`: The ground truth values.
- `metadata`: A dictionary containing feature metadata.
- `output_feature_name`: The name of the output feature to visualize.
- `labels_limit`: An upper limit on the numeric encoded label value.
- `model_names`: The name or list of names of the models to use as labels.
- `output_directory`: The directory where to save the plots.
- `file_format`: The file format of the output plots.
- `ground_truth_apply_idx`: Whether to use metadata['str2idx'] in np.vectorize.

The function first checks if the ground truth values are not already a numpy array and if so, it converts the raw values to encoded values using the feature metadata. It then flattens the predictions for each model and stores them in the `predictions_per_model` variable.

If `labels_limit` is greater than 0, it sets any ground truth values higher than `labels_limit` to `labels_limit`.

Next, it maps the predictions to numeric labels using the metadata if available. It calculates the accuracy, precision, recall, and F1 score for each model using the ground truth and predictions. These metrics are stored in the `accuracies`, `precisions`, `recalls`, and `f1s` lists, respectively.

If an `output_directory` is specified, it creates the directory if it doesn't exist and sets the `filename` variable to the path of the output file.

Finally, it calls the `compare_classifiers_plot` function from the `visualization_utils` module to generate the bar plot visualization, passing in the metrics, model names, and output filename.

Mathematical operations or procedures:
- Converting raw ground truth values to encoded values using the feature metadata.
- Flattening the predictions for each model.
- Setting ground truth values higher than `labels_limit` to `labels_limit`.
- Mapping predictions to numeric labels using the metadata.
- Calculating accuracy, precision, recall, and F1 score for each model using the ground truth and predictions.
- Generating a bar plot visualization of the metrics.

## Function **`compare_classifiers_performance_subset`** Overview
The `compare_classifiers_performance_subset` function takes in several parameters and produces a bar plot visualization comparing the performance of different models on a subset of the training set.

Parameters:
- `probabilities_per_model`: A list of numpy arrays representing the predicted probabilities for each model.
- `ground_truth`: The ground truth values.
- `metadata`: A dictionary containing feature metadata.
- `output_feature_name`: The name of the output feature.
- `top_n_classes`: A list containing the number of classes to plot.
- `labels_limit`: An upper limit on the numeric encoded label value.
- `subset`: A string specifying the type of subset filtering. Valid values are "ground_truth" or "predictions".
- `model_names`: The name or list of names of the models to use as labels.
- `output_directory`: The directory where to save the plots.
- `file_format`: The file format of the output plots.
- `ground_truth_apply_idx`: Whether to use metadata['str2idx'] in np.vectorize.

The function first checks if the `ground_truth` parameter is not a numpy array and if so, it assumes that the raw values need to be translated to encoded values using the `metadata['str2idx']` dictionary.

Next, it converts the `top_n_classes` and `model_names` parameters to lists if they are not already.

If the `labels_limit` parameter is greater than 0, it sets any ground truth values higher than `labels_limit` to `labels_limit`.

The function then determines the subset indices based on the `subset` parameter. If `subset` is "ground_truth", it selects the subset of ground truth values that are less than `k`, where `k` is the first element of `top_n_classes`. If `subset` is "predictions", it selects the subset of ground truth values where the predicted class index is less than `k`.

The function then performs some operations on the probabilities for each model. If `labels_limit` is greater than 0 and the number of classes in the probabilities is greater than `labels_limit + 1`, it limits the probabilities to the first `labels_limit + 1` classes and sums the probabilities for the remaining classes.

For each model, it calculates the accuracy and hits@k metrics based on the subset of ground truth values and the corresponding predicted probabilities.

Finally, it generates a title for the plot based on the subset type and the number of classes, and saves the plot to a file if `output_directory` is specified.

The function does not perform any mathematical operations that require LaTex code.

## Function **`compare_classifiers_performance_changing_k`** Overview
The `compare_classifiers_performance_changing_k` function takes in several parameters and produces a line plot that shows the Hits@K metric while changing K from 1 to `top_k` for each model.

The parameters of the function are as follows:

- `probabilities_per_model` (List[np.array]): A list of model probabilities.
- `ground_truth` (Union[pd.Series, np.ndarray]): The ground truth values.
- `metadata` (dict): A feature metadata dictionary.
- `output_feature_name` (str): The output feature name.
- `top_k` (int): The number of elements in the ranklist to consider.
- `labels_limit` (int): An upper limit on the numeric encoded label value. Encoded numeric label values in the dataset that are higher than `labels_limit` are considered to be "rare" labels.
- `model_names` (Union[str, List[str]], default: `None`): The model name or list of model names to use as labels.
- `output_directory` (str, default: `None`): The directory where to save the plots. If not specified, the plots will be displayed in a window.
- `file_format` (str, default: `'pdf'`): The file format of the output plots - `'pdf'` or `'png'`.
- `ground_truth_apply_idx` (bool, default: `True`): Whether to use `metadata['str2idx']` in `np.vectorize`.

The function performs the following mathematical operations or procedures:

1. If the `ground_truth` is not an instance of `np.ndarray`, it assumes that the raw value needs to be translated to an encoded value using the `feature_metadata["str2idx"]` and `ground_truth_apply_idx` parameters.
2. Sets `k` equal to `top_k`.
3. If `labels_limit` is greater than 0, it sets all values in `ground_truth` that are higher than `labels_limit` to `labels_limit`.
4. Assigns the `probabilities_per_model` to the `probs` variable.
5. Initializes an empty list `hits_at_ks` to store the Hits@K values for each model.
6. Converts the `model_names` to a list if it is not already.
7. Iterates over each model's probabilities and performs the following operations:
   - If `labels_limit` is greater than 0 and the shape of `prob` is larger than `labels_limit + 1`, it limits the probabilities to the first `labels_limit + 1` columns and sums the remaining columns to the last column.
   - Sorts the probabilities in descending order along the second axis.
   - Initializes a list `hits_at_k` with zeros of length `k`.
   - Iterates over each ground truth value and each value of `j` from 0 to `k-1` and performs the following operations:
     - Uses `np.in1d` to check if the ground truth value is in the last `j+1` values of the sorted probabilities.
     - Adds 1 to the corresponding index in `hits_at_k` if the ground truth value is found.
   - Appends the normalized `hits_at_k` values to `hits_at_ks`.
8. If `output_directory` is specified, it creates the directory if it doesn't exist and sets the `filename` variable to the path of the output file.
9. Calls the `compare_classifiers_line_plot` function from the `visualization_utils` module to generate the line plot using the `np.arange(1, k + 1)` as the x-axis values, `hits_at_ks` as the y-axis values, "hits@k" as the y-axis label, `model_names_list` as the legend labels, and "Classifier comparison (hits@k)" as the title. The plot is saved to the `filename` if specified, otherwise it is displayed in a window.

## Function **`compare_classifiers_multiclass_multimetric`** Overview
The `compare_classifiers_multiclass_multimetric` function compares the performance of multiple classifiers on a multiclass classification task using precision, recall, and F1 score metrics. It generates plots to visualize the performance of each classifier on different classes.

Parameters:
- `test_stats_per_model`: A list of dictionaries containing evaluation performance statistics for each model.
- `metadata`: A dictionary containing intermediate preprocess structures created during training, including mappings of the input dataset.
- `output_feature_name`: The name of the output feature to use for the visualization. If `None`, all output features are used.
- `top_n_classes`: A list of integers specifying the number of classes to plot.
- `model_names`: The name or list of names of the models to use as labels. Default is `None`.
- `output_directory`: The directory where the plots will be saved. If not specified, the plots will be displayed in a window. Default is `None`.
- `file_format`: The file format of the output plots, either `'pdf'` or `'png'`. Default is `'pdf'`.
- `**kwargs`: Additional keyword arguments.

The function performs the following operations:
1. It generates a filename template for the plots based on the output directory, model name, output feature name, and file format.
2. It converts the `test_stats_per_model` and `model_names` parameters to lists if they are not already.
3. It validates the output feature name from the test statistics and retrieves the corresponding output feature names.
4. It iterates over each test statistics and output feature name combination.
5. For each combination, it retrieves the per-class statistics (precision, recall, and F1 score) from the test statistics.
6. It sorts the per-class statistics based on the class names.
7. It selects the top `k` classes based on the specified `top_n_classes` parameter.
8. It generates plots for the precision, recall, and F1 score of the selected classes.
9. It generates additional plots for the best `k` and worst `k` classes based on the F1 score.
10. It generates a plot for the precision, recall, and F1 score of all classes sorted by the F1 score.
11. It logs the model name, the best and worst classes based on the F1 score, and the number of classes with F1 score greater than 0 and equal to 0.

The mathematical operations performed in the function include sorting the per-class statistics based on the F1 score, selecting the top `k` classes, and generating plots for the precision, recall, and F1 score. However, there are no specific mathematical equations or procedures that can be represented using LaTex code.

## Function **`compare_classifiers_predictions`** Overview
The `compare_classifiers_predictions` function compares the predictions of two models for a specified output feature. It takes the following parameters:

- `predictions_per_model`: A list containing the model predictions for the specified output feature.
- `ground_truth`: The ground truth values for the output feature.
- `metadata`: A dictionary containing feature metadata.
- `output_feature_name`: The name of the output feature.
- `labels_limit`: An upper limit on the numeric encoded label value. Labels higher than this limit are considered "rare" labels.
- `model_names`: The name or list of names of the models to use as labels. (default: `None`)
- `output_directory`: The directory where to save the plots. If not specified, plots will be displayed in a window. (default: `None`)
- `file_format`: The file format of the output plots - `'pdf'` or `'png'`. (default: `'pdf'`)
- `ground_truth_apply_idx`: Whether to use `metadata['str2idx']` in `np.vectorize`. (default: `True`)

The function performs the following mathematical operations or procedures:

1. If the `ground_truth` is not a numpy array, it assumes that the raw values need to be translated to encoded values using the `feature_metadata["str2idx"]` dictionary and the `ground_truth_apply_idx` flag.
2. It converts the `model_names` parameter to a list if it is not already a list.
3. It assigns names to the two models being compared (`name_c1` and `name_c2`) based on the `model_names` list.
4. It assigns the predictions of the two models to `pred_c1` and `pred_c2`.
5. If `labels_limit` is greater than 0, it sets any label values in `ground_truth`, `pred_c1`, and `pred_c2` that are higher than `labels_limit` to `labels_limit`.
6. It initializes variables to count the number of correct predictions for both models (`both_right`), both models making the same wrong prediction (`both_wrong_same`), both models making different wrong predictions (`both_wrong_different`), model 1 making the correct prediction and model 2 making the wrong prediction (`c1_right_c2_wrong`), and model 1 making the wrong prediction and model 2 making the correct prediction (`c1_wrong_c2_right`).
7. It iterates over all the data points and updates the count variables based on the predictions and ground truth values.
8. It calculates the number of data points where only one model made the correct prediction (`one_right`) and the number of data points where both models made the wrong prediction (`both_wrong`).
9. It logs the results, including the number and percentage of data points where both models were right, one model was right, both models were wrong, and the breakdown of wrong predictions.
10. If an `output_directory` is specified, it creates the directory if it doesn't exist and generates a filename for the plot.
11. It calls the `donut` function from the `visualization_utils` module to create a donut plot showing the comparison of the models' predictions.
12. The plot is either saved to the specified `output_directory` or displayed in a window.

The mathematical operations or procedures performed by the function are not explicitly represented by equations, so there is no LaTex code to generate for them.

## Function **`compare_classifiers_predictions_distribution`** Overview
The `compare_classifiers_predictions_distribution` function compares the distributions of predictions made by different classifiers for a specified output feature. It visualizes the comparison using a radar plot.

The function takes the following parameters:

- `predictions_per_model`: A list containing the model predictions for the specified output feature.
- `ground_truth`: The ground truth values for the output feature.
- `metadata`: A dictionary containing feature metadata.
- `output_feature_name`: The name of the output feature.
- `labels_limit`: An upper limit on the numeric encoded label value. Labels with values higher than `labels_limit` are considered rare.
- `model_names`: The name or list of names of the models to use as labels. (default: `None`)
- `output_directory`: The directory where the plots will be saved. If not specified, the plots will be displayed in a window. (default: `None`)
- `file_format`: The file format of the output plots - `'pdf'` or `'png'`. (default: `'pdf'`)
- `ground_truth_apply_idx`: Whether to use `metadata['str2idx']` in `np.vectorize`. (default: `True`)

The function performs the following mathematical operations or procedures:

1. If the `ground_truth` is not a numpy array, it assumes that the raw values need to be translated to encoded values using the `str2idx` mapping in the `metadata` dictionary.
2. Converts `model_names` to a list if it is not already a list.
3. If `labels_limit` is greater than 0, it sets any ground truth values higher than `labels_limit` to `labels_limit`. It also sets any predictions higher than `labels_limit` to `labels_limit` for each model.
4. Finds the maximum value in the ground truth and predictions, and adds 1 to get the maximum value for the radar plot axis.
5. Computes the counts of each label in the ground truth using `np.bincount` and calculates the probability distribution of the ground truth.
6. Computes the counts of each label in the predictions for each model using `np.bincount` and calculates the probability distribution of the predictions for each model.
7. If `output_directory` is specified, it creates the directory if it doesn't exist and generates a filename for the radar plot.
8. Calls the `radar_chart` function from the `visualization_utils` module to generate the radar plot using the ground truth and predictions probability distributions, model names, and the filename.

The mathematical operations can be represented using LaTex code as follows:

Let $N$ be the number of models, $M$ be the number of classes, and $L$ be the `labels_limit`.

1. Translate raw ground truth values to encoded values:
   - If `ground_truth` is not a numpy array, use `metadata[output_feature_name]["str2idx"]` to translate the raw values to encoded values.

2. Set labels higher than `labels_limit` to `labels_limit`:
   - Set `ground_truth[ground_truth > L]` to `L`.
   - For each model $i$ in `predictions_per_model`, set `predictions_per_model[i][predictions_per_model[i] > L]` to `L`.

3. Find the maximum value in the ground truth and predictions:
   - Let $max_{gt}$ be the maximum value in `ground_truth`.
   - Let $max_{pred}$ be the maximum value among all predictions in `predictions_per_model`.
   - Let $max_{val} = \max(max_{gt}, max_{pred}) + 1$.

4. Compute the counts of each label in the ground truth:
   - Let `counts_gt` be the result of `np.bincount(ground_truth, minlength=max_{val})`.
   - Let `prob_gt` be the probability distribution of the ground truth: `prob_gt = counts_gt / counts_gt.sum()`.

5. Compute the counts of each label in the predictions for each model:
   - For each model $i$ in `predictions_per_model`, let `counts_predictions[i]` be the result of `np.bincount(predictions_per_model[i], minlength=max_{val})`.
   - For each model $i$ in `predictions_per_model`, let `prob_predictions[i]` be the probability distribution of the predictions for model $i`: `prob_predictions[i] = counts_predictions[i] / counts_predictions[i].sum()`.

6. Generate the radar plot:
   - If `output_directory` is specified, create the directory if it doesn't exist and generate a filename for the radar plot.
   - Call `visualization_utils.radar_chart(prob_gt, prob_predictions, model_names_list, filename=filename)` to generate the radar plot.

## Function **`confidence_thresholding`** Overview
The `confidence_thresholding` function takes in several parameters and performs mathematical operations to show models accuracy and data coverage while increasing a threshold on the probabilities of predictions for a specified output feature.

Parameters:
- `probabilities_per_model` (List[np.array]): A list of model probabilities.
- `ground_truth` (Union[pd.Series, np.ndarray]): Ground truth values.
- `metadata` (dict): Feature metadata dictionary.
- `output_feature_name` (str): The name of the output feature.
- `labels_limit` (int): An upper limit on the numeric encoded label value. Labels higher than this limit are considered "rare" labels.
- `model_names` (Union[str, List[str]], default: `None`): The name or list of names of the models to use as labels.
- `output_directory` (str, default: `None`): The directory where to save plots. If not specified, plots will be displayed in a window.
- `file_format` (str, default: `'pdf'`): The file format of the output plots.
- `ground_truth_apply_idx` (bool, default: `True`): Whether to use metadata['str2idx'] in np.vectorize.

Mathematical Operations:
1. If `ground_truth` is not an instance of `np.ndarray`, it assumes that raw values need to be translated to encoded values using the `metadata` dictionary.
2. If `labels_limit` is greater than 0, it sets all ground truth values higher than `labels_limit` to `labels_limit`.
3. Assigns `probabilities_per_model` to `probs`.
4. Converts `model_names` to a list if it is not already a list.
5. Generates a list of thresholds from 0 to 1 with a step size of 0.05.
6. Initializes empty lists `accuracies` and `dataset_kept`.
7. For each model in `probs`:
   - If `labels_limit` is greater than 0 and the number of columns in `prob` is greater than `labels_limit` + 1, it limits the columns of `prob` to `labels_limit` + 1 and sums the remaining columns to the last column.
   - Calculates the maximum probability for each row in `prob` and assigns it to `max_prob`.
   - Calculates the predicted labels for each row in `prob` and assigns it to `predictions`.
   - Initializes empty lists `accuracies_alg` and `dataset_kept_alg`.
   - For each threshold in `thresholds`:
     - If `threshold` is greater than or equal to 1, it sets `threshold` to 0.999.
     - Filters the indices where `max_prob` is greater than or equal to `threshold`.
     - Filters the ground truth values and predictions based on the filtered indices.
     - Calculates the accuracy as the sum of matching ground truth and predictions divided by the length of filtered ground truth.
     - Appends the accuracy and the ratio of filtered ground truth length to the total ground truth length to `accuracies_alg` and `dataset_kept_alg`, respectively.
   - Appends `accuracies_alg` and `dataset_kept_alg` to `accuracies` and `dataset_kept`, respectively.
8. If `output_directory` is specified, it creates the directory if it doesn't exist and assigns the filename for the output plot.
9. Calls the `confidence_filtering_plot` function from the `visualization_utils` module to generate the plot with the given thresholds, accuracies, dataset kept ratios, model names, title, and filename.

## Function **`confidence_thresholding_data_vs_acc`** Overview
The `confidence_thresholding_data_vs_acc` function takes in several parameters and produces a line plot comparing the data coverage and accuracy of different models as the threshold on the probabilities of predictions increases.

Parameters:
- `probabilities_per_model`: A list of numpy arrays representing the probabilities predicted by each model.
- `ground_truth`: The ground truth values.
- `metadata`: A dictionary containing feature metadata.
- `output_feature_name`: The name of the output feature.
- `labels_limit`: An upper limit on the numeric encoded label value.
- `model_names`: The name or list of names of the models to use as labels.
- `output_directory`: The directory where to save the plots.
- `file_format`: The file format of the output plots.
- `ground_truth_apply_idx`: Whether to use metadata['str2idx'] in np.vectorize.

The function first checks if the ground truth values are not a numpy array and if so, it translates the raw values to encoded values using the feature metadata. If `labels_limit` is greater than 0, it limits the ground truth values to be less than or equal to `labels_limit`.

The function then calculates the maximum probability and predictions for each model. It iterates over a range of thresholds and for each threshold, filters the indices where the maximum probability is greater than or equal to the threshold. It calculates the accuracy by comparing the filtered ground truth values with the filtered predictions. The accuracy and the ratio of the filtered data to the total ground truth data are stored for each threshold.

Finally, the function calls the `confidence_filtering_data_vs_acc_plot` function from the `visualization_utils` module to generate the line plot comparing the accuracies and data coverage for each model.

Mathematical operations or procedures:
- Translating raw ground truth values to encoded values using metadata.
- Limiting ground truth values to be less than or equal to `labels_limit`.
- Calculating the maximum probability and predictions for each model.
- Iterating over a range of thresholds and filtering the data based on the threshold.
- Calculating the accuracy by comparing the filtered ground truth values with the filtered predictions.
- Calculating the ratio of the filtered data to the total ground truth data.
- Storing the accuracies and data coverage for each model and threshold.
- Generating a line plot comparing the accuracies and data coverage for each model.

## Function **`confidence_thresholding_data_vs_acc_subset`** Overview
The `confidence_thresholding_data_vs_acc_subset` function compares the confidence threshold data vs accuracy for multiple models on a subset of data. It produces a line plot indicating the accuracy of each model and the data coverage while increasing the threshold on the probabilities of predictions for a specified output feature.

Parameters:
- `probabilities_per_model`: A list of numpy arrays representing the probabilities predicted by each model.
- `ground_truth`: The ground truth values.
- `metadata`: A dictionary containing feature metadata.
- `output_feature_name`: The name of the output feature.
- `top_n_classes`: A list containing the number of classes to plot.
- `labels_limit`: An upper limit on the numeric encoded label value.
- `subset`: A string specifying the type of subset filtering. Valid values are `'ground_truth'` or `'predictions'`.
- `model_names`: The name or list of names of the models to use as labels.
- `output_directory`: The directory where to save the plots.
- `file_format`: The file format of the output plots.
- `ground_truth_apply_idx`: Whether to use metadata['str2idx'] in np.vectorize.

The function performs the following mathematical operations or procedures:

1. Translates the ground truth values to encoded values if they are not already in numpy array format.
2. Converts `top_n_classes` to a list and assigns the first element to `k`.
3. If `labels_limit` is greater than 0, sets all ground truth values higher than `labels_limit` to `labels_limit`.
4. Assigns the probabilities to `probs`.
5. Converts `model_names` to a list and assigns it to `model_names_list`.
6. Generates a list of thresholds from 0 to 1 with a step size of 0.05.
7. Initializes empty lists `accuracies` and `dataset_kept`.
8. Determines the subset indices based on the value of `subset` and assigns the subset ground truth values to `gt_subset`.
9. Loops over each model's probabilities and performs the following operations:
   - If `labels_limit` is greater than 0 and the number of classes in the probabilities is higher than `labels_limit` + 1, limits the probabilities to `labels_limit` + 1 classes and sums the probabilities of the remaining classes.
   - If `subset` is `'predictions'`, determines the subset indices based on the maximum predicted class and assigns the subset ground truth values to `gt_subset`.
   - Filters the probabilities and ground truth values based on the subset indices.
   - Calculates the maximum probability and predicted class for each data point in the subset.
   - Initializes empty lists `accuracies_alg` and `dataset_kept_alg`.
   - Loops over each threshold and performs the following operations:
     - Sets the threshold to 0.999 if it is greater than or equal to 1.
     - Filters the data points based on the maximum probability threshold.
     - Calculates the accuracy by comparing the filtered ground truth values with the filtered predicted classes.
     - Appends the accuracy and the percentage of data points kept to `accuracies_alg` and `dataset_kept_alg`, respectively.
   - Appends `accuracies_alg` and `dataset_kept_alg` to `accuracies` and `dataset_kept`, respectively.
10. Creates a filename for the output plot if `output_directory` is specified.
11. Calls the `confidence_filtering_data_vs_acc_plot` function from the `visualization_utils` module to generate the plot.

The mathematical operations or procedures can be represented using LaTex code as follows:

1. Translating ground truth values:

$$
\text{{ground\_truth}} = \text{{\_vectorize\_ground\_truth}}(\text{{ground\_truth}}, \text{{feature\_metadata}}[\text{{"str2idx"}}], \text{{ground\_truth\_apply\_idx}})
$$

2. Converting `top_n_classes`:

$$
k = \text{{top\_n\_classes\_list}}[0]
$$

3. Limiting labels:

$$
\text{{ground\_truth}}[\text{{ground\_truth}} > \text{{labels\_limit}}] = \text{{labels\_limit}}
$$

4. Assigning probabilities:

$$
\text{{probs}} = \text{{probabilities\_per\_model}}
$$

5. Converting `model_names`:

$$
\text{{model\_names\_list}} = \text{{convert\_to\_list}}(\text{{model\_names}})
$$

6. Generating thresholds:

$$
\text{{thresholds}} = \left[ \frac{t}{100} \text{{ for }} t \text{{ in range}}(0, 101, 5) \right]
$$

7. Initializing lists:

$$
\text{{accuracies}} = []
$$

$$
\text{{dataset\_kept}} = []
$$

8. Determining subset indices:

$$
\text{{subset\_indices}} = \text{{ground\_truth}} > 0
$$

$$
\text{{gt\_subset}} = \text{{ground\_truth}}
$$

$$
\text{{if }} \text{{subset}} == \text{{"ground\_truth"}}:
$$

$$
\text{{subset\_indices}} = \text{{ground\_truth}} < k
$$

$$
\text{{gt\_subset}} = \text{{ground\_truth}}[\text{{subset\_indices}}]
$$

$$
\text{{logger.info}}(\text{{f"Subset is {len(gt\_subset) / len(ground\_truth) * 100:.2f}% of the data"}})
$$

9. Looping over models:

$$
\text{{for }} i, \text{{prob}} \text{{ in enumerate}}(\text{{probs}}):
$$

$$
\text{{if }} \text{{labels\_limit}} > 0 \text{{ and }} \text{{prob.shape[1]}} > \text{{labels\_limit}} + 1:
$$

$$
\text{{prob\_limit}} = \text{{prob}}[:, : \text{{labels\_limit}} + 1]
$$

$$
\text{{prob\_limit}}[:, \text{{labels\_limit}}] = \text{{prob}}[:, \text{{labels\_limit}}:].\text{{sum}}(1)
$$

$$
\text{{prob}} = \text{{prob\_limit}}
$$

$$
\text{{if }} \text{{subset}} == \text{{PREDICTIONS}}:
$$

$$
\text{{subset\_indices}} = \text{{np.argmax}}(\text{{prob}}, \text{{axis=1}}) < k
$$

$$
\text{{gt\_subset}} = \text{{ground\_truth}}[\text{{subset\_indices}}]
$$

$$
\text{{logger.info}}(\text{{"Subset for model\_name {} is {:.2f}% of the data".format(}}\text{{model\_names}}[i] \text{{ if }} \text{{model\_names}} \text{{ and }} i < \text{{len(model\_names)}} \text{{ else }} i, \text{{len(gt\_subset) / len(ground\_truth) * 100,}}\text{{)}})
$$

$$
\text{{prob\_subset}} = \text{{prob}}[\text{{subset\_indices}}]
$$

$$
\text{{max\_prob}} = \text{{np.max}}(\text{{prob\_subset}}, \text{{axis=1}})
$$

$$
\text{{predictions}} = \text{{np.argmax}}(\text{{prob\_subset}}, \text{{axis=1}})
$$

$$
\text{{accuracies\_alg}} = []
$$

$$
\text{{dataset\_kept\_alg}} = []
$$

$$
\text{{for }} \text{{threshold}} \text{{ in }} \text{{thresholds}}:
$$

$$
\text{{threshold}} = \text{{threshold if threshold < 1 else 0.999}}
$$

$$
\text{{filtered\_indices}} = \text{{max\_prob}} \geq \text{{threshold}}
$$

$$
\text{{filtered\_gt}} = \text{{gt\_subset}}[\text{{filtered\_indices}}]
$$

$$
\text{{filtered\_predictions}} = \text{{predictions}}[\text{{filtered\_indices}}]
$$

$$
\text{{accuracy}} = \left( \text{{filtered\_gt}} == \text{{filtered\_predictions}} \right).\text{{sum}}() / \text{{len}}(\text{{filtered\_gt}})
$$

$$
\text{{accuracies\_alg}}.\text{{append}}(\text{{accuracy}})
$$

$$
\text{{dataset\_kept\_alg}}.\text{{append}}(\text{{len}}(\text{{filtered\_gt}}) / \text{{len}}(\text{{ground\_truth}}))
$$

$$
\text{{accuracies}}.\text{{append}}(\text{{accuracies\_alg}})
$$

$$
\text{{dataset\_kept}}.\text{{append}}(\text{{dataset\_kept\_alg}})
$$

10. Creating filename:

$$
\text{{filename}} = \text{{None}}
$$

$$
\text{{if }} \text{{output\_directory}}:
$$

$$
\text{{os.makedirs}}(\text{{output\_directory}}, \text{{exist\_ok=True}})
$$

$$
\text{{filename}} = \text{{os.path.join}}(\text{{output\_directory}}, \text{{"confidence\_thresholding\_data\_vs\_acc\_subset."}} + \text{{file\_format}})
$$

11. Generating plot:

$$
\text{{visualization\_utils.confidence\_filtering\_data\_vs\_acc\_plot}}(\text{{accuracies}}, \text{{dataset\_kept}}, \text{{model\_names\_list}}, \text{{title="Confidence\_Thresholding (Data vs Accuracy)"}}, \text{{filename=filename}})
$$

## Function **`confidence_thresholding_data_vs_acc_subset_per_class`** Overview
The function `confidence_thresholding_data_vs_acc_subset_per_class` takes in several parameters and produces a line plot comparing the accuracy of different models with the data coverage while increasing a threshold on the probabilities of predictions for a specified output feature. The purpose of each parameter is as follows:

- `probabilities_per_model`: A list of numpy arrays representing the probabilities predicted by each model.
- `ground_truth`: The ground truth values.
- `metadata`: Intermediate preprocess structure created during training containing mappings of the input dataset.
- `output_feature_name`: The name of the output feature to use for the visualization.
- `top_n_classes`: The number of top classes or a list containing the number of top classes to plot.
- `labels_limit`: An upper limit on the numeric encoded label value.
- `subset`: A string specifying the type of subset filtering. Valid values are "ground_truth" or "predictions".
- `model_names`: The name of the model or a list of model names to use as labels.
- `output_directory`: The directory where to save the plots.
- `file_format`: The file format of the output plots.
- `ground_truth_apply_idx`: Whether to use metadata['str2idx'] in np.vectorize.

The function performs the following mathematical operations or procedures:

1. It checks if the ground truth is not a numpy array and converts it to an encoded value if necessary.
2. It generates a filename template for saving the plots.
3. It converts the `top_n_classes` parameter to a list.
4. It checks if `top_n_classes` is greater than the maximum number of tokens and truncates it if necessary.
5. It applies a label limit to the ground truth if `labels_limit` is greater than 0.
6. It initializes variables for storing probabilities and model names.
7. It generates a list of thresholds from 0 to 1 with a step size of 0.05.
8. It iterates over each class within the top_n_classes.
9. It initializes lists for storing accuracies and dataset coverage.
10. It applies subset filtering based on the `subset` parameter and updates the subset indices and ground truth subset accordingly.
11. It iterates over each model and performs the following steps:
    - If `labels_limit` is greater than 0 and the number of classes in the probabilities is greater than `labels_limit` + 1, it limits the probabilities to `labels_limit` + 1 classes and sums the probabilities of the remaining classes.
    - If `subset` is "predictions", it applies subset filtering based on the current class and updates the subset indices and ground truth subset accordingly.
    - It selects the probabilities and ground truth subset based on the subset indices.
    - It calculates the maximum probability and predicted class for each data point.
    - It iterates over each threshold and performs the following steps:
        - It filters the data points based on the maximum probability threshold.
        - It calculates the accuracy by comparing the filtered ground truth and predictions.
        - It appends the accuracy and dataset coverage to the respective lists.
12. It generates the output feature name based on the current class.
13. It generates the filename for saving the plot.
14. It calls the `confidence_filtering_data_vs_acc_plot` function from the `visualization_utils` module to create the plot.

The mathematical operations or procedures performed by the function are not explicitly defined in the code, but they involve filtering and calculating accuracies based on probability thresholds.

## Function **`confidence_thresholding_2thresholds_2d`** Overview
The `confidence_thresholding_2thresholds_2d` function takes in several parameters and performs various mathematical operations to generate plots that show confidence threshold data vs accuracy for two output feature names.

Parameters:
- `probabilities_per_model`: A list of numpy arrays representing the probabilities predicted by each model.
- `ground_truths`: A list of numpy arrays or pandas Series containing the ground truth data.
- `metadata`: A dictionary containing feature metadata.
- `threshold_output_feature_names`: A list of two output feature names for visualization.
- `labels_limit`: An integer representing the upper limit on the numeric encoded label value.
- `model_names`: A string or list of strings representing the model names to use as labels.
- `output_directory`: A string representing the directory where to save the plots.
- `file_format`: A string representing the file format of the output plots.

Mathematical Operations:
1. Validate the input probabilities and threshold output feature names.
2. Convert the model names to a list if it is a string.
3. Generate a filename template for saving the plots.
4. If the ground truths are not numpy arrays, encode the raw values to encoded values using the feature metadata.
5. Apply label limit to the ground truth arrays if the label limit is greater than 0.
6. Generate a list of thresholds from 0 to 1 with a step size of 0.05.
7. Initialize empty lists for accuracies, dataset kept, and interps.
8. If the number of classes in the probabilities arrays is greater than the label limit, modify the probabilities arrays to include a "rare" label.
9. Compute the maximum probability and predicted class for each model.
10. Iterate over the thresholds for the first output feature name.
11. Iterate over the thresholds for the second output feature name.
12. Filter the indices based on the maximum probabilities and thresholds.
13. Compute the coverage and accuracy for the filtered data.
14. Append the accuracy and coverage to the respective lists.
15. Compute the interpolated accuracies for fixed step coverage.
16. Log the CSV table.
17. Generate a multiline plot showing coverage vs accuracy for each model.
18. Generate a max line plot showing the maximum accuracy for each coverage threshold.
19. Generate a max line plot with thresholds showing the maximum accuracy for each coverage threshold and the corresponding thresholds.
20. Save the plots in the specified output directory.

The mathematical operations are not explicitly shown in the code, but they involve filtering and computing accuracy and coverage based on the thresholds and predicted probabilities.

## Function **`confidence_thresholding_2thresholds_3d`** Overview
The `confidence_thresholding_2thresholds_3d` function takes in several parameters and performs mathematical operations to generate a 3D plot showing the relationship between confidence thresholds and accuracy for two output feature names.

Parameters:
- `probabilities_per_model`: A list of numpy arrays representing the model probabilities.
- `ground_truths`: A list of numpy arrays or pandas Series containing the ground truth data.
- `metadata`: A dictionary containing feature metadata.
- `threshold_output_feature_names`: A list containing two output feature names for visualization.
- `labels_limit`: An integer representing the upper limit on the numeric encoded label value.
- `output_directory`: A string representing the directory where the plots will be saved.
- `file_format`: A string representing the file format of the output plots.

The function first validates the input probabilities and thresholds using the `validate_conf_thresholds_and_probabilities_2d_3d` function. If the validation fails, the function returns.

Next, the function processes the ground truth data. If the ground truth data is not a numpy array, it assumes that the raw values need to be translated to encoded values. It uses the feature metadata to perform this translation.

The function then applies a label limit to the ground truth data if `labels_limit` is greater than 0. Any label values higher than the limit are considered "rare" labels.

The function defines a list of thresholds ranging from 0 to 1 with a step size of 0.05.

It then calculates the maximum probabilities and predictions for each output feature name.

The function iterates over the thresholds and calculates the accuracy and dataset coverage for each combination of thresholds. It filters the data based on the confidence thresholds and calculates the accuracy by comparing the filtered ground truth values with the filtered predictions.

The accuracies and dataset coverage values are stored in lists.

Finally, the function generates a filename for the output plot if `output_directory` is specified. It then calls the `confidence_filtering_3d_plot` function from the `visualization_utils` module to generate the 3D plot using the threshold values, accuracies, dataset coverage, and other parameters.

The function does not return any value.

## Function **`binary_threshold_vs_metric`** Overview
The `binary_threshold_vs_metric` function takes in several parameters and produces a line chart that plots a threshold on the confidence of a model against a specified metric for a given output feature.

The parameters of the function are as follows:

- `probabilities_per_model` (List[np.array]): A list of model probabilities.
- `ground_truth` (Union[pd.Series, np.ndarray]): The ground truth values.
- `metadata` (dict): A feature metadata dictionary.
- `output_feature_name` (str): The name of the output feature.
- `metrics` (List[str]): The metrics to display, which can be `'f1'`, `'precision'`, `'recall'`, or `'accuracy'`.
- `positive_label` (int, default: `1`): The numeric encoded value for the positive class.
- `model_names` (List[str], default: `None`): A list of the names of the models to use as labels.
- `output_directory` (str, default: `None`): The directory where to save the plots. If not specified, the plots will be displayed in a window.
- `file_format` (str, default: `'pdf'`): The file format of the output plots, which can be `'pdf'` or `'png'`.
- `ground_truth_apply_idx` (bool, default: `True`): Whether to use `metadata['str2idx']` in `np.vectorize`.

The function first checks if the `ground_truth` parameter is not an instance of `np.ndarray`. If it is not, it assumes that the raw value needs to be translated to the encoded value using the `metadata` and `positive_label` parameters.

Next, the function assigns the `probabilities_per_model` parameter to the `probs` variable and converts the `model_names` and `metrics` parameters to lists if they are not already.

The function then generates a filename template for saving the plots based on the `file_format` parameter and the specified output directory.

A list of thresholds is created, ranging from 0 to 1 with a step size of 0.05.

The function then iterates over each metric specified in the `metrics` parameter. If the metric is not one of the supported metrics (`'f1'`, `'precision'`, `'recall'`, `'accuracy'`), an error message is logged and the iteration continues to the next metric.

For each metric, the function calculates the scores for each model and threshold combination. It iterates over each model's probabilities and for each threshold, it calculates the predictions based on whether the probability is greater than or equal to the threshold. The metric score is then calculated using the appropriate sklearn metric function (`f1_score`, `precision_score`, `recall_score`, or `accuracy_score`).

The scores for each model are appended to a list, and this list is appended to the `scores` list.

Finally, the `threshold_vs_metric_plot` function from the `visualization_utils` module is called to generate the line chart. The thresholds, scores, model names, and a title are passed as arguments. If an output directory is specified, the plot is saved with the corresponding filename.

The function does not return any value (`None`).

## Function **`precision_recall_curves`** Overview
The `precision_recall_curves` function takes in several parameters and generates precision-recall curves for output features in specified models. Here is a breakdown of the purpose of each parameter:

- `probabilities_per_model` (List[np.array]): A list of model probabilities.
- `ground_truth` (Union[pd.Series, np.ndarray]): The ground truth values.
- `metadata` (dict): A dictionary containing feature metadata.
- `output_feature_name` (str): The name of the output feature.
- `positive_label` (int, default: `1`): The numeric encoded value for the positive class.
- `model_names` (Union[str, List[str]], default: `None`): The name or list of names of the models to use as labels.
- `output_directory` (str, default: `None`): The directory where the plots will be saved. If not specified, the plots will be displayed in a window.
- `file_format` (str, default: `'pdf'`): The file format of the output plots ('pdf' or 'png').
- `ground_truth_apply_idx` (bool, default: `True`): Whether to use metadata['str2idx'] in np.vectorize.

The function first checks if the `ground_truth` parameter is not an instance of `np.ndarray`. If it is not, it assumes that the raw value needs to be translated to an encoded value using the `_convert_ground_truth` function.

Next, the function assigns the `probabilities_per_model` to the `probs` variable and converts the `model_names` parameter to a list if it is not already.

Then, the function iterates over the `probs` and calculates the precision, recall, and thresholds using the `sklearn.metrics.precision_recall_curve` function. The precision and recall values are stored in a list called `precision_recalls`.

Finally, the function creates a filename for the output plot if the `output_directory` parameter is specified. It then calls the `visualization_utils.precision_recall_curves_plot` function to generate the precision-recall curves plot.

The function does not perform any specific mathematical operations or procedures beyond calculating precision and recall values using the `sklearn.metrics.precision_recall_curve` function.

## Function **`precision_recall_curves_from_test_statistics`** Overview
The `precision_recall_curves_from_test_statistics` function takes in several parameters and produces a precision-recall curve visualization for the specified models.

Parameters:
- `test_stats_per_model` (List[dict]): A list of dictionaries containing evaluation performance statistics for each model.
- `output_feature_name` (str): The name of the output feature to use for the visualization. This feature should be binary.
- `model_names` (Union[str, List[str]], default: `None`): The name or list of names of the models to use as labels in the visualization.
- `output_directory` (str, default: `None`): The directory where the plots will be saved. If not specified, the plots will be displayed in a window.
- `file_format` (str, default: `'pdf'`): The file format of the output plots, either `'pdf'` or `'png'`.

The function first converts the `model_names` parameter to a list if it is not already a list. It then generates a filename template for the output plots based on the specified `file_format` parameter and the function `generate_filename_template_path`.

Next, the function iterates over each dictionary in `test_stats_per_model` and retrieves the precision and recall values for the specified `output_feature_name`. These values are stored in a list of dictionaries called `precision_recalls`.

Finally, the function calls the `precision_recall_curves_plot` function from the `visualization_utils` module, passing in the `precision_recalls`, `model_names_list`, a title for the plot, and the filename template path for saving the plot.

The function does not perform any mathematical operations or procedures.

## Function **`roc_curves`** Overview
The `roc_curves` function is used to plot ROC curves for output features in specified models. It takes the following parameters:

- `probabilities_per_model` (List[np.array]): A list of model probabilities.
- `ground_truth` (Union[pd.Series, np.ndarray]): Ground truth values.
- `metadata` (dict): Feature metadata dictionary.
- `output_feature_name` (str): Output feature name.
- `positive_label` (int, default: `1`): Numeric encoded value for the positive class.
- `model_names` (Union[str, List[str]], default: `None`): Model name or list of model names to use as labels.
- `output_directory` (str, default: `None`): Directory where to save plots. If not specified, plots will be displayed in a window.
- `file_format` (str, default: `'pdf'`): File format of output plots - `'pdf'` or `'png'`.
- `ground_truth_apply_idx` (bool, default: `True`): Whether to use metadata['str2idx'] in np.vectorize.

The function first checks if the `ground_truth` parameter is an instance of `np.ndarray`. If not, it assumes that the raw values need to be translated to encoded values using the `_convert_ground_truth` function. The `positive_label` parameter is also updated accordingly.

Next, the function assigns the `probabilities_per_model` parameter to the `probs` variable and converts the `model_names` parameter to a list using the `convert_to_list` function.

Then, the function iterates over the `probs` list and for each probability array, it checks if the shape of the array is greater than 1. If so, it selects the column corresponding to the `positive_label`. It then calculates the false positive rate (fpr), true positive rate (tpr), and thresholds using the `sklearn.metrics.roc_curve` function and appends them to the `fpr_tprs` list.

Finally, the function creates a filename for the output plot if the `output_directory` parameter is specified. It then calls the `visualization_utils.roc_curves` function to plot the ROC curves using the `fpr_tprs`, `model_names_list`, and `filename` parameters.

The function does not perform any mathematical operations or procedures beyond the calculations of fpr, tpr, and thresholds using the `sklearn.metrics.roc_curve` function.

## Function **`roc_curves_from_test_statistics`** Overview
The `roc_curves_from_test_statistics` function takes in several parameters and produces a line chart plotting the ROC curves for the specified models' output binary feature.

Parameters:
- `test_stats_per_model` (List[dict]): A list of dictionaries containing evaluation performance statistics for each model.
- `output_feature_name` (str): The name of the output feature to use for the visualization.
- `model_names` (Union[str, List[str]], default: `None`): The model name or a list of model names to use as labels for the ROC curves.
- `output_directory` (str, default: `None`): The directory where the plots will be saved. If not specified, the plots will be displayed in a window.
- `file_format` (str, default: `'pdf'`): The file format of the output plots, either `'pdf'` or `'png'`.

The function first converts the `model_names` parameter to a list using the `convert_to_list` function. It then generates a filename template based on the `file_format` parameter and the output directory using the `generate_filename_template_path` function.

Next, the function iterates over each dictionary in the `test_stats_per_model` list. For each dictionary, it retrieves the false positive rate (FPR) and true positive rate (TPR) values from the `roc_curve` sub-dictionary corresponding to the `output_feature_name`. These FPR and TPR values are then appended to the `fpr_tprs` list.

Finally, the `roc_curves` function from the `visualization_utils` module is called with the `fpr_tprs`, `model_names_list`, title, and filename template path as arguments to generate the ROC curves visualization.

Mathematical operations or procedures:
- None. The function mainly performs data retrieval and visualization using the provided parameters and functions.

## Function **`calibration_1_vs_all`** Overview
The `calibration_1_vs_all` function takes in several parameters and performs the following mathematical operations:

1. It checks the type of `ground_truth` and converts it to a numpy array if it is not already.
2. It assigns the input probabilities to the variable `probs`.
3. It limits the number of classes in `ground_truth` and `probs` if `labels_limit` is specified.
4. It calculates the number of classes and initializes an empty list `brier_scores`.
5. It creates a list of class names based on the feature metadata.
6. It iterates over each class and performs the following operations:
   - Initializes empty lists for `fraction_positives_class`, `mean_predicted_vals_class`, `probs_class`, and `brier_scores_class`.
   - Iterates over each probability and performs the following operations:
     - Creates a binary vector `gt_class` indicating whether the ground truth matches the current class index.
     - Extracts the probabilities for the current class from `prob`.
     - Calculates the fraction of positives and mean predicted values using the `calibration_curve` function.
     - Appends the results to the respective lists.
     - Calculates the Brier score using the `brier_score_loss` function and appends it to `brier_scores_class`.
   - Appends `brier_scores_class` to `brier_scores`.
   - Generates filenames for saving the calibration plot and prediction distribution plot.
   - Calls the `calibration_plot` function from the `visualization_utils` module to generate the calibration plot.
   - Calls the `predictions_distribution_plot` function from the `visualization_utils` module to generate the prediction distribution plot.
7. Generates a filename for saving the Brier plot.
8. Calls the `brier_plot` function from the `visualization_utils` module to generate the Brier plot.

The mathematical operations involve calculating the fraction of positives, mean predicted values, and Brier scores for each class and model. These values are used to generate calibration plots, prediction distribution plots, and a Brier plot.

## Function **`calibration_multiclass`** Overview
The `calibration_multiclass` function takes in several parameters and performs various mathematical operations to visualize and compare the calibration of multiple models for multiclass classification.

Parameters:
- `probabilities_per_model`: A list of numpy arrays representing the predicted probabilities of each model.
- `ground_truth`: The ground truth values for the classification task, either as a pandas Series or numpy array.
- `metadata`: A dictionary containing metadata for the features.
- `output_feature_name`: The name of the output feature.
- `labels_limit`: An upper limit on the numeric encoded label value. Labels higher than this limit are considered "rare" labels.
- `model_names`: (Optional) A list of names for the models.
- `output_directory`: (Optional) The directory where the plots will be saved.
- `file_format`: (Optional) The file format of the output plots.
- `ground_truth_apply_idx`: (Optional) Whether to use metadata['str2idx'] in np.vectorize.

Mathematical Operations:
1. If the `ground_truth` is not a numpy array, it is assumed that the raw values need to be translated to encoded values using the metadata.
2. The `probs` variable is assigned the value of `probabilities_per_model`.
3. The `model_names_list` is generated by converting `model_names` to a list.
4. The `filename_template` is set as a string template for the output plot filenames.
5. The `filename_template_path` is generated by combining the `output_directory` and `filename_template`.
6. If `labels_limit` is greater than 0, any ground truth values higher than `labels_limit` are set to `labels_limit`.
7. The variable `prob_classes` is initialized to 0.
8. For each model's predicted probabilities in `probs`:
   - If `labels_limit` is greater than 0 and the number of classes in the probabilities is greater than `labels_limit + 1`, the probabilities are limited to `labels_limit + 1` classes. The probabilities for the remaining classes are summed and assigned to the last class.
   - If the number of classes in the probabilities is greater than `prob_classes`, `prob_classes` is updated to the number of classes in the probabilities.
9. The dimensions of the one-hot encoded ground truth array `gt_one_hot` are determined based on the maximum number of classes (`prob_classes`) and the maximum value in the ground truth array.
10. The ground truth array `gt_one_hot` is one-hot encoded using numpy indexing.
11. The one-hot encoded ground truth array is flattened to be compared with the flattened predicted probabilities.
12. Empty lists `fraction_positives`, `mean_predicted_vals`, and `brier_scores` are initialized.
13. For each set of predicted probabilities in `probs`:
   - The probabilities are flattened.
   - The calibration curve is computed using the `calibration_curve` function from scikit-learn, and the fraction of positives and mean predicted values are appended to `fraction_positives` and `mean_predicted_vals` respectively.
   - The Brier score is computed using the `brier_score_loss` function from scikit-learn, and the score is appended to `brier_scores`.
14. If `output_directory` is specified, the output directory is created if it doesn't exist, and the filename for the calibration plot is set.
15. The `calibration_plot` function from the `visualization_utils` module is called to generate the calibration plot using `fraction_positives`, `mean_predicted_vals`, `model_names_list`, and the filename.
16. If `output_directory` is specified, the filename for the Brier score plot is set.
17. The `compare_classifiers_plot` function from the `visualization_utils` module is called to generate the Brier score plot using `brier_scores`, the label "brier", `model_names`, and the filename.
18. For each Brier score in `brier_scores`, the score is logged using the `logger.info` function.

The function does not return any value.

## Function **`confusion_matrix`** Overview
The `confusion_matrix` function in Python is used to display the confusion matrix in the models' predictions for each output feature. It takes several parameters:

- `test_stats_per_model`: A list of dictionaries containing evaluation performance statistics.
- `metadata`: A dictionary containing intermediate preprocess structures created during training.
- `output_feature_name`: The name of the output feature to use for the visualization. If set to `None`, all output features will be used.
- `top_n_classes`: A list of integers specifying the number of top classes to plot.
- `normalize`: A boolean flag indicating whether to normalize rows in the confusion matrix.
- `model_names`: The name or list of names of the models to use as labels. Default is `None`.
- `output_directory`: The directory where the plots will be saved. If not specified, the plots will be displayed in a window. Default is `None`.
- `file_format`: The file format of the output plots, either `'pdf'` or `'png'`. Default is `'pdf'`.

The function first converts the `model_names` parameter to a list and defines a filename template for saving the plots. It then validates the `output_feature_name` parameter and checks if a confusion matrix is present in the test statistics for each model and output feature. If a confusion matrix is found, it retrieves the matrix and the corresponding labels.

For each specified `top_n_classes`, the function selects the top `k` classes from the confusion matrix and normalizes the matrix if `normalize` is set to `True`. It then generates a filename for saving the plot, if an output directory is specified, and calls the `confusion_matrix_plot` function from the `visualization_utils` module to create the heatmap plot.

After plotting the confusion matrix, the function calculates the entropy of each row in the matrix and sorts the classes based on their entropy. It then generates a filename for saving the entropy plot, if an output directory is specified, and calls the `bar_plot` function from the `visualization_utils` module to create a bar plot of the class entropies.

If no confusion matrix is found in the evaluation data, an error is logged and a `FileNotFoundError` is raised.

The mathematical operations performed in this function include selecting the top `k` classes from the confusion matrix, normalizing the matrix by dividing each row by its sum, calculating the entropy of each row, and sorting the classes based on their entropy.

Here is the LaTeX code for the mathematical operations:

1. Selecting top `k` classes from the confusion matrix:

$$
cm = \_confusion\_matrix[:k, :k]
$$

2. Normalizing the matrix:

$$
cm_{norm} = \frac{cm}{cm.sum(1)[:, np.newaxis]}
$$

3. Calculating the entropy of each row:

$$
\text{{entropies}} = \text{{entropy}}(row)
$$

4. Sorting the classes based on entropy:

$$
\text{{class\_desc\_entropy}} = \text{{argsort}}(\text{{class\_entropy}})[::-1]
$$

5. Generating the filename for saving the entropy plot:

$$
\text{{filename}} = \text{{filename\_template\_path}}.\text{{format}}("entropy\_" + \text{{model\_name\_name}}, \text{{output\_feature\_name}}, "top" + str(k))
$$

## Function **`frequency_vs_f1`** Overview
The `frequency_vs_f1` function takes in several parameters and produces two plots that show prediction statistics for the specified `output_feature_name` for each model.

The parameters of the function are as follows:

- `test_stats_per_model` (List[dict]): A list of dictionaries containing evaluation performance statistics for each model.
- `metadata` (dict): An intermediate preprocess structure created during training containing the mappings of the input dataset.
- `output_feature_name` (Union[str, None]): The name of the output feature to use for the visualization. If `None`, all output features are used.
- `top_n_classes` (List[int]): The number of top classes or a list containing the number of top classes to plot.
- `model_names` (Union[str, List[str]], default: `None`): The model name or a list of model names to use as labels.
- `output_directory` (str, default: `None`): The directory where to save the plots. If not specified, the plots will be displayed in a window.
- `file_format` (str, default: `'pdf'`): The file format of the output plots - `'pdf'` or `'png'`.

The function first converts the `model_names` parameter to a list if it is not already a list. It then sets up a filename template for saving the plots based on the `output_directory` and `file_format` parameters.

Next, the function validates the `output_feature_name` parameter and retrieves the output feature names from the `test_stats_per_model` list.

The function then iterates over the `test_stats_per_model` list and for each model and output feature, it performs the following steps:

1. Determines the model name and sets up the filename for saving the plot.
2. Retrieves the per-class statistics and class names from the `test_stats` and `metadata` dictionaries.
3. Creates a dictionary mapping class names to frequencies.
4. Sorts the class names and frequencies by the F1 score.
5. Creates a visualization of the F1 score vs frequency plot using the sorted data.
6. Sorts the class names and frequencies by the frequency.
7. Creates a visualization of the frequency vs F1 score plot using the sorted data.

The function uses the `visualization_utils.double_axis_line_plot` function to create the plots.

The function does not return any value.

## Function **`hyperopt_report_cli`** Overview
The `hyperopt_report_cli` function is a Python function that generates a report about hyperparameter optimization. It takes in several parameters:

- `hyperopt_stats_path`: This is the path to the hyperopt results JSON file. It is a required parameter and specifies the location of the file containing the results of the hyperparameter optimization.

- `output_directory`: This parameter specifies the path where the output plots will be saved. It is an optional parameter and if not provided, the plots will be saved in the current working directory.

- `file_format`: This parameter specifies the format of the output plots. It can be either "pdf" or "png". It is an optional parameter and if not provided, the default format is set to "pdf".

The function calls another function called `hyperopt_report` to generate the actual report. This function creates one graph per hyperparameter to show the distribution of results and one additional graph of pairwise hyperparameters interactions.

The `hyperopt_report` function is not shown in the code snippet, but it is assumed to be defined elsewhere. It takes in the same parameters as `hyperopt_report_cli` and is responsible for generating the report based on the hyperparameter optimization results.

The mathematical operations or procedures performed by the `hyperopt_report_cli` function are not explicitly mentioned in the code snippet. It is assumed that the `hyperopt_report` function performs the necessary calculations and data analysis to generate the graphs and visualizations for the report. Without the actual implementation of the `hyperopt_report` function, it is not possible to provide specific details about the mathematical operations or procedures involved.

## Function **`hyperopt_report`** Overview
The `hyperopt_report` function is used to produce a report about hyperparameter optimization. It creates graphs to show the distribution of results for each hyperparameter and an additional graph to visualize pairwise hyperparameter interactions.

The function takes the following parameters:

- `hyperopt_stats_path` (str): The path to the hyperopt results JSON file.
- `output_directory` (str, default: `None`): The directory where the plots will be saved. If not specified, the plots will be displayed in a window.
- `file_format` (str, default: `'pdf'`): The file format of the output plots, which can be `'pdf'` or `'png'`.

The function does not return any value.

The function first generates a filename template based on the specified `file_format`. It then loads the hyperopt stats from the JSON file specified by `hyperopt_stats_path`.

The `visualization_utils.hyperopt_report` function is then called with the following arguments:

- `hyperopt_stats["hyperopt_config"]["parameters"]`: The hyperparameter configuration.
- `hyperopt_results_to_dataframe(...)`: A function that converts the hyperopt results to a dataframe.
- `metric=hyperopt_stats["hyperopt_config"]["metric"]`: The metric used for optimization.
- `filename_template=filename_template_path`: The path to the output file.

The `hyperopt_report` function essentially serves as a wrapper for the `visualization_utils.hyperopt_report` function, providing the necessary arguments and handling the file saving/displaying logic.

## Function **`hyperopt_hiplot_cli`** Overview
The `hyperopt_hiplot_cli` function is a Python function that generates a parallel coordinate plot about hyperparameter optimization using the HiPlot library. It takes the following parameters:

- `hyperopt_stats_path`: This is the path to the hyperopt results JSON file. This file contains the results of the hyperparameter optimization process.
- `output_directory` (optional): This is the path where the output plots will be saved. If not provided, the plots will be saved in the current working directory.

The function calls the `hyperopt_hiplot` function, passing the `hyperopt_stats_path` and `output_directory` parameters. The `hyperopt_hiplot` function is responsible for generating the parallel coordinate plot using the HiPlot library.

The purpose of the `hyperopt_hiplot_cli` function is to provide a command-line interface for generating the parallel coordinate plot. It takes the necessary input parameters and calls the `hyperopt_hiplot` function to generate the plot.

There are no mathematical operations or procedures performed in this function. It simply calls the `hyperopt_hiplot` function to generate the plot based on the provided hyperparameter optimization results.

## Function **`hyperopt_hiplot`** Overview
The `hyperopt_hiplot` function is used to produce a parallel coordinate plot about hyperparameter optimization. It takes in the path to a hyperopt results JSON file and an optional output directory where the plots can be saved.

The function first generates a filename for the HTML file that will contain the plot. It then loads the hyperopt results from the JSON file using the `load_json` function. The loaded results are then converted into a pandas DataFrame using the `hyperopt_results_to_dataframe` function, which takes in the hyperopt results, the hyperopt parameters, and the metric used for optimization.

Finally, the `hyperopt_hiplot` function from the `visualization_utils` module is called to generate the parallel coordinate plot. The resulting plot is saved as an HTML file using the generated filename.

Mathematical operations or procedures are not performed in this function.

## Function **`_convert_space_to_dtype`** Overview
The function `_convert_space_to_dtype` takes a parameter `space` of type string and returns a string. 

The purpose of the function is to convert a given space into a data type. It checks if the given `space` is present in the `RAY_TUNE_FLOAT_SPACES` list. If it is, the function returns the string "float". If the `space` is present in the `RAY_TUNE_INT_SPACES` list, the function returns the string "int". If the `space` is not present in either of the lists, the function returns the string "object".

There are no mathematical operations or procedures performed in this function. It simply checks if the given `space` is present in the predefined lists and returns the corresponding data type.

## Function **`hyperopt_results_to_dataframe`** Overview
The `hyperopt_results_to_dataframe` function takes three parameters: `hyperopt_results`, `hyperopt_parameters`, and `metric`. 

- `hyperopt_results` is a list of dictionaries containing the results of hyperparameter optimization. Each dictionary represents a set of hyperparameters and their corresponding metric score.
- `hyperopt_parameters` is a dictionary that maps hyperparameter names to their corresponding search spaces and data types.
- `metric` is a string representing the name of the metric used to evaluate the performance of the hyperparameters.

The function first creates an empty pandas DataFrame `df`. It then iterates over each dictionary in `hyperopt_results` and creates a new dictionary that includes the metric score and all the hyperparameters as key-value pairs. This new dictionary is appended to `df`.

Next, the function converts the data types of the hyperparameters in `df` based on the search spaces defined in `hyperopt_parameters`. It uses the `_convert_space_to_dtype` function to determine the appropriate data type for each hyperparameter.

Finally, the function returns the resulting DataFrame `df`.

The mathematical operations or procedures performed by this function involve creating a DataFrame from the hyperopt results and converting the data types of the hyperparameters. There are no explicit mathematical equations involved in this function.

Here is the LaTex code for displaying the equations in a markdown document:

```latex
\begin{align*}
df &= \text{pd.DataFrame}([\{metric: \text{res["metric_score"]}, **\text{res["parameters"]}\} \text{ for res in hyperopt_results}]) \\
df &= df.\text{astype}(\{hp\_name: \_convert\_space\_to\_dtype(hp\_params[SPACE]) \text{ for hp\_name, hp\_params in hyperopt\_parameters.items()}\})
\end{align*}
```

## Function **`get_visualizations_registry`** Overview
The `get_visualizations_registry` function returns a dictionary that maps visualization names to corresponding visualization functions. The function does not take any parameters and returns a dictionary of type `Dict[str, Callable]`.

The purpose of this function is to provide a registry of available visualizations in a Python program. Each key in the dictionary represents the name of a visualization, and the corresponding value is the function that generates that visualization.

Here is a description of each visualization and its corresponding function:

1. "compare_performance": compare_performance_cli
   - This visualization compares the performance of different models.
   
2. "compare_classifiers_performance_from_prob": compare_classifiers_performance_from_prob_cli
   - This visualization compares the performance of different classifiers using probability predictions.
   
3. "compare_classifiers_performance_from_pred": compare_classifiers_performance_from_pred_cli
   - This visualization compares the performance of different classifiers using predicted labels.
   
4. "compare_classifiers_performance_subset": compare_classifiers_performance_subset_cli
   - This visualization compares the performance of different classifiers on a subset of data.
   
5. "compare_classifiers_performance_changing_k": compare_classifiers_performance_changing_k_cli
   - This visualization compares the performance of different classifiers while changing a parameter k.
   
6. "compare_classifiers_multiclass_multimetric": compare_classifiers_multiclass_multimetric_cli
   - This visualization compares the performance of different classifiers for multiclass classification with multiple metrics.
   
7. "compare_classifiers_predictions": compare_classifiers_predictions_cli
   - This visualization compares the predictions of different classifiers.
   
8. "compare_classifiers_predictions_distribution": compare_classifiers_predictions_distribution_cli
   - This visualization compares the distribution of predictions from different classifiers.
   
9. "confidence_thresholding": confidence_thresholding_cli
   - This visualization analyzes the effect of confidence thresholding on classifier performance.
   
10. "confidence_thresholding_data_vs_acc": confidence_thresholding_data_vs_acc_cli
    - This visualization compares the effect of confidence thresholding on data size and accuracy.
    
11. "confidence_thresholding_data_vs_acc_subset": confidence_thresholding_data_vs_acc_subset_cli
    - This visualization compares the effect of confidence thresholding on data size and accuracy for a subset of data.
    
12. "confidence_thresholding_data_vs_acc_subset_per_class": confidence_thresholding_data_vs_acc_subset_per_class_cli
    - This visualization compares the effect of confidence thresholding on data size and accuracy for each class individually.
    
13. "confidence_thresholding_2thresholds_2d": confidence_thresholding_2thresholds_2d_cli
    - This visualization analyzes the effect of two confidence thresholds on classifier performance in a 2D plot.
    
14. "confidence_thresholding_2thresholds_3d": confidence_thresholding_2thresholds_3d_cli
    - This visualization analyzes the effect of two confidence thresholds on classifier performance in a 3D plot.
    
15. "binary_threshold_vs_metric": binary_threshold_vs_metric_cli
    - This visualization compares a binary threshold with a performance metric.
    
16. "roc_curves": roc_curves_cli
    - This visualization plots ROC curves for different classifiers.
    
17. "roc_curves_from_test_statistics": roc_curves_from_test_statistics_cli
    - This visualization plots ROC curves using test statistics.
    
18. "precision_recall_curves": precision_recall_curves_cli
    - This visualization plots precision-recall curves for different classifiers.
    
19. "precision_recall_curves_from_test_statistics": precision_recall_curves_from_test_statistics_cli
    - This visualization plots precision-recall curves using test statistics.
    
20. "calibration_1_vs_all": calibration_1_vs_all_cli
    - This visualization compares calibration of one class against all other classes.
    
21. "calibration_multiclass": calibration_multiclass_cli
    - This visualization compares calibration for multiclass classification.
    
22. "confusion_matrix": confusion_matrix_cli
    - This visualization plots a confusion matrix for classifier predictions.
    
23. "frequency_vs_f1": frequency_vs_f1_cli
    - This visualization compares the frequency of classes with F1 scores.
    
24. "learning_curves": learning_curves_cli
    - This visualization plots learning curves for different classifiers.
    
25. "hyperopt_report": hyperopt_report_cli
    - This visualization generates a report for hyperparameter optimization.
    
26. "hyperopt_hiplot": hyperopt_hiplot_cli
    - This visualization generates a HiPlot visualization for hyperparameter optimization.

The mathematical operations or procedures performed by these visualization functions are not explicitly mentioned in the code snippet provided. Therefore, it is not possible to generate LaTeX code for displaying equations in a markdown document.

## Function **`cli`** Overview
The `cli` function is a Python function that takes a single parameter `sys_argv`. This function is used to parse command line arguments and execute a visualization function based on the provided arguments.

The purpose of each parameter is as follows:

- `sys_argv`: A list of command line arguments passed to the script.

The function uses the `argparse` module to define and parse command line arguments. Here is a description of each argument:

- `-g`, `--ground_truth`: Specifies the ground truth file.
- `-gm`, `--ground_truth_metadata`: Specifies the input metadata JSON file.
- `-sf`, `--split_file`: Specifies a file containing split values used in conjunction with the ground truth file.
- `-od`, `--output_directory`: Specifies the directory where to save plots. If not specified, plots will be displayed in a window.
- `-ff`, `--file_format`: Specifies the file format of output plots. Default is "pdf" with choices of "pdf" or "png".
- `-v`, `--visualization`: Specifies the type of visualization to generate. This argument is required.
- `-ofn`, `--output_feature_name`: Specifies the name of the output feature to visualize.
- `-gts`, `--ground_truth_split`: Specifies the ground truth split - 0:train, 1:validation, 2:test split.
- `-tf`, `--threshold_output_feature_names`: Specifies the names of output features for 2d threshold.
- `-pred`, `--predictions`: Specifies the predictions files.
- `-prob`, `--probabilities`: Specifies the probabilities files.
- `-trs`, `--training_statistics`: Specifies the training stats files.
- `-tes`, `--test_statistics`: Specifies the test stats files.
- `-hs`, `--hyperopt_stats_path`: Specifies the hyperopt stats file.
- `-mn`, `--model_names`: Specifies the names of the models to use as labels.
- `-tn`, `--top_n_classes`: Specifies the number of classes to plot.
- `-k`, `--top_k`: Specifies the number of elements in the ranklist to consider.
- `-ll`, `--labels_limit`: Specifies the maximum numbers of labels. Encoded numeric label values in dataset that are higher than labels_limit are considered to be "rare" labels.
- `-ss`, `--subset`: Specifies the type of subset filtering. Choices are "ground_truth" or "PREDICTIONS".
- `-n`, `--normalize`: Specifies whether to normalize rows in confusion matrix.
- `-m`, `--metrics`: Specifies the metrics to display in threshold_vs_metric.
- `-pl`, `--positive_label`: Specifies the label of the positive class for the roc curve.
- `-l`, `--logging_level`: Specifies the level of logging to use. Choices are "critical", "error", "warning", "info", "debug", or "notset".

After parsing the command line arguments, the function sets up logging and executes the visualization function based on the provided arguments.

The mathematical operations or procedures performed by this function are not explicitly stated in the code. The function is mainly responsible for parsing command line arguments and executing the appropriate visualization function based on those arguments.

