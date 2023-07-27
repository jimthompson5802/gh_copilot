# Module:`multivariate_ols.py` Overview

The module contains functions and classes for performing multivariate linear regression analysis. It includes functions for fitting the model, conducting hypothesis tests, and generating summary statistics. The main class in the module is `_MultivariateOLS`, which represents the multivariate linear regression model. The `fit` method of this class fits the model to the data. The `_MultivariateOLSResults` class represents the results of the fitted model and provides methods for conducting hypothesis tests and generating summary statistics. The `mv_test` method of this class conducts hypothesis tests on the model coefficients. The `summary` method generates a summary of the model results. The module also includes helper functions for performing the regression analysis and calculating test statistics.

## Function **`_multivariate_ols_fit`** Overview
The function `_multivariate_ols_fit` is used to perform multivariate ordinary least squares (OLS) regression. It takes the following parameters:

- `endog`: The dependent variable(s) in the regression model. It should be a 2-dimensional array-like object with shape `(nobs, k_endog)`, where `nobs` is the number of observations and `k_endog` is the number of dependent variables.

- `exog`: The independent variable(s) in the regression model. It should be a 2-dimensional array-like object with shape `(nobs, k_exog)`, where `nobs` is the number of observations and `k_exog` is the number of independent variables.

- `method` (optional): The method used to calculate the regression coefficients and other statistics. It can be either `'pinv'` (default) or `'svd'`.

- `tolerance` (optional): The tolerance level used to check for singularity of the covariance matrix. It is set to `1e-8` by default.

The function performs the following mathematical operations or procedures:

1. It checks if the number of observations in `endog` and `exog` are the same. If not, it raises a `ValueError` indicating that the number of rows should be the same.

2. It calculates the degrees of freedom for the residuals (`df_resid`) as the difference between the number of observations and the number of independent variables.

3. If the `method` is `'pinv'`, it calculates the regression coefficients (`params`) using the Moore-Penrose pseudo-inverse of `exog` (`pinv_x`). It also calculates the inverse of the covariance matrix (`inv_cov`) as the product of the pseudo-inverse and its transpose. If the rank of `inv_cov` is less than the number of independent variables, it raises a `ValueError` indicating that the covariance of `exog` is singular. Finally, it calculates the sums of squares and cross-products of residuals (`sscpr`) as the difference between the total sum of squares (`y.T.dot(y)`) and the sum of squares of the predicted values (`t.T.dot(t)`).

4. If the `method` is `'svd'`, it performs singular value decomposition (SVD) on `exog` to obtain the matrices `u`, `s`, and `v`. It checks if any of the singular values (`s`) are smaller than the tolerance level. If so, it raises a `ValueError` indicating that the covariance of `exog` is singular. It then calculates the regression coefficients (`params`) using the SVD matrices, the inverse of the covariance matrix (`inv_cov`), and the sums of squares and cross-products of residuals (`sscpr`) in a similar manner as the `'pinv'` method.

5. If the `method` is neither `'pinv'` nor `'svd'`, it raises a `ValueError` indicating that the method is not supported.

The function returns a tuple containing the regression coefficients (`params`), the degrees of freedom for the residuals (`df_resid`), the inverse of the covariance matrix (`inv_cov`), and the sums of squares and cross-products of residuals (`sscpr`).

## Function **`multivariate_stats`** Overview
The `multivariate_stats` function takes in several parameters:

- `eigenvals`: A numpy array of eigenvalues.
- `r_err_sscp`: A scalar representing the residual error sum of squares and cross products.
- `r_contrast`: A scalar representing the contrast sum of squares and cross products.
- `df_resid`: A scalar representing the degrees of freedom for the residuals.
- `tolerance`: A scalar representing the tolerance value for determining if an eigenvalue is greater than the tolerance.

The function performs several mathematical operations and procedures to calculate various statistics related to multivariate analysis. Here is a breakdown of the mathematical operations performed:

1. The function initializes variables `v`, `p`, `q`, and `s` based on the input parameters.
2. It creates a boolean array `ind` by checking if each eigenvalue is greater than the tolerance.
3. It calculates the number of eigenvalues greater than the tolerance and assigns it to `n_e`.
4. It creates two new arrays `eigv2` and `eigv1` by filtering the eigenvalues based on the boolean array `ind` and performing element-wise operations on the filtered eigenvalues.
5. It calculates the values for four different statistics: Wilks' lambda, Pillai's trace, Hotelling-Lawley trace, and Roy's greatest root. These values are stored in a pandas DataFrame called `results`.
6. It calculates the degrees of freedom and F-values for each statistic using the calculated values and stores them in the `results` DataFrame.
7. It calculates the p-values for each statistic using the F-values and the degrees of freedom.
8. Finally, it returns the `results` DataFrame.

Here is the LaTex code for the equations used in the function:

1. Wilks' lambda:

$$
\text{{Wilks' lambda}} = \prod_{i=1}^{n_e} (1 - \text{{eigv2}}_i)
$$

2. Pillai's trace:

$$
\text{{Pillai's trace}} = \sum_{i=1}^{n_e} \text{{eigv2}}_i
$$

3. Hotelling-Lawley trace:

$$
\text{{Hotelling-Lawley trace}} = \sum_{i=1}^{n_e} \frac{{\text{{eigv2}}_i}}{{1 - \text{{eigv2}}_i}}
$$

4. Roy's greatest root:

$$
\text{{Roy's greatest root}} = \max_{i=1}^{n_e} \frac{{\text{{eigv2}}_i}}{{1 - \text{{eigv2}}_i}}
$$

Note: The LaTex code provided above assumes that the variables `eigv2` and `eigv1` are arrays. If they are scalars, the LaTex code needs to be modified accordingly.

## Function **`_multivariate_ols_test`** Overview
The function `_multivariate_ols_test` is a Python function that performs a multivariate ordinary least squares (OLS) test. It takes four parameters: `hypotheses`, `fit_results`, `exog_names`, and `endog_names`.

- `hypotheses` is a list of hypothesis matrices. Each hypothesis matrix represents a linear combination of the regression coefficients that is being tested. The dimensions of each hypothesis matrix should match the dimensions of the regression coefficient matrix.
- `fit_results` is a tuple containing the results of the OLS regression. It includes the estimated regression coefficients (`params`), the degrees of freedom of the residuals (`df_resid`), the inverse of the covariance matrix (`inv_cov`), and the sum of squares and cross-products matrix (`sscpr`).
- `exog_names` is a list of names for the exogenous variables (independent variables) in the regression model.
- `endog_names` is a list of names for the endogenous variable (dependent variable) in the regression model.

The function defines an inner function `fn` that takes three parameters: `L`, `M`, and `C`. These parameters represent the hypothesis matrix, the projection matrix, and the constant matrix, respectively.

The function then performs the following mathematical operations:

1. Calculate `t1` by multiplying the hypothesis matrix `L` with the estimated regression coefficients `params` and the projection matrix `M`, and subtracting the constant matrix `C`. This can be represented as:


$$
t1 = L \cdot \text{{params}} \cdot M - C
$$

2. Calculate `t2` by multiplying the hypothesis matrix `L` with the inverse of the covariance matrix `inv_cov`, and then multiplying the result by the transpose of `L`. This can be represented as:


$$
t2 = L \cdot \text{{inv_cov}} \cdot L^T
$$

3. Calculate the rank of `t2` using the `matrix_rank` function.
4. Calculate `H` by taking the dot product of `t1` (transposed) with the inverse of `t2`, and then taking the dot product of the result with `t1`. This can be represented as:


$$
H = t1^T \cdot \text{{inv}}(t2) \cdot t1
$$

5. Calculate `E` by taking the dot product of the transpose of the projection matrix `M` with the sum of squares and cross-products matrix `sscpr`, and then taking the dot product of the result with `M`. This can be represented as:


$$
E = M^T \cdot \text{{sscpr}} \cdot M
$$

6. Return the values of `E`, `H`, the rank `q` of `t2`, and the degrees of freedom of the residuals `df_resid`.

Finally, the function returns the result of calling the `_multivariate_test` function with the `hypotheses`, `exog_names`, `endog_names`, and `fn` as parameters.

## Function **`_multivariate_test`** Overview
The `_multivariate_test` function takes in four parameters: `hypotheses`, `exog_names`, `endog_names`, and `fn`. 

- `hypotheses` is a list of tuples, where each tuple represents a hypothesis to be tested. The length of each tuple can be 2, 3, or 4, depending on the number of constraints specified. The first element of the tuple is the name of the hypothesis, the second element is the contrast matrix `L`, the third element is the transform matrix `M`, and the fourth element is the constant matrix `C`.

- `exog_names` is a list of names for the exogenous variables.

- `endog_names` is a list of names for the endogenous variables.

- `fn` is a function that performs the mathematical operations or procedures required to calculate the test statistics. It takes in the contrast matrix `L`, transform matrix `M`, and constant matrix `C` as parameters and returns the error matrix `E`, hypothesis matrix `H`, degrees of freedom `q`, and residual degrees of freedom `df_resid`.

The function iterates over each hypothesis in the `hypotheses` list and performs the following steps:

1. Checks the length of the hypothesis tuple to determine the number of constraints specified.

2. If the length is 2, assigns the name and contrast matrix `L` from the tuple. Sets `M` and `C` to `None`.

3. If the length is 3, assigns the name, contrast matrix `L`, and transform matrix `M` from the tuple. Sets `C` to `None`.

4. If the length is 4, assigns the name, contrast matrix `L`, transform matrix `M`, and constant matrix `C` from the tuple.

5. Checks the type of `L` and converts it to a contrast matrix if it is a string. Otherwise, checks if `L` is a 2-dimensional numpy array with the correct number of columns.

6. If `M` is `None`, assigns the identity matrix with the number of rows equal to the number of endogenous variables. Otherwise, checks the type of `M` and converts it to a transform matrix if it is a string. Otherwise, checks if `M` is a 2-dimensional numpy array with the correct number of rows.

7. If `C` is `None`, assigns a zero matrix with the same number of rows as `L` and the same number of columns as `M`. Otherwise, checks if `C` is a 2-dimensional numpy array.

8. Checks if the number of rows of `L` and `C` are the same. Checks if the number of columns of `M` and `C` are the same.

9. Calls the `fn` function with the contrast matrix `L`, transform matrix `M`, and constant matrix `C` as parameters. Retrieves the error matrix `E`, hypothesis matrix `H`, degrees of freedom `q`, and residual degrees of freedom `df_resid`.

10. Adds the error matrix `E` and hypothesis matrix `H` element-wise to obtain `EH`.

11. Calculates the rank of `EH` to obtain the number of non-zero eigenvalues.

12. Sorts the eigenvalues of the inverse of `EH` multiplied by `H`.

13. Calls the `multivariate_stats` function with the sorted eigenvalues, number of non-zero eigenvalues, degrees of freedom `q`, and residual degrees of freedom `df_resid` to obtain the test statistics.

14. Stores the test statistics, contrast matrix `L`, transform matrix `M`, constant matrix `C`, error matrix `E`, and hypothesis matrix `H` in the `results` dictionary with the hypothesis name as the key.

15. Returns the `results` dictionary.

The mathematical operations performed by the function include checking the dimensions and types of the input matrices, calculating the error matrix `E`, hypothesis matrix `H`, degrees of freedom `q`, and residual degrees of freedom `df_resid`, adding matrices element-wise, calculating the rank of a matrix, sorting eigenvalues, and calculating test statistics.

## Class **`_MultivariateOLS`** Overview
The `_MultivariateOLS` class is a subclass of the `Model` class in Python. It is used to fit multivariate ordinary least squares (OLS) models. 

The class has an attribute `_formula_max_endog` which is set to `None`. 

The `__init__` method is the constructor of the class. It takes the arguments `endog`, `exog`, `missing`, `hasconst`, and `**kwargs`. It checks if the shape of `endog` is either 1-dimensional or has only 1 column. If this condition is met, it raises a `ValueError` with the message "There must be more than one dependent variable to fit multivariate OLS!". Otherwise, it calls the constructor of the `Model` class with the given arguments.

The `fit` method is used to fit the multivariate OLS model. It takes an optional argument `method` which defaults to 'svd'. It calls the `_multivariate_ols_fit` function with the `endog`, `exog`, and `method` arguments. The result of the fitting process is stored in the `_fittedmod` attribute. Finally, it returns an instance of the `_MultivariateOLSResults` class with the current instance of `_MultivariateOLS` as an argument.

### Method **`__init__`** Overview
The `__init__` method is a special method in Python classes that is automatically called when an object is created. It is used to initialize the object's attributes. In the case of the `_MultivariateOLS` class, the `__init__` method takes several parameters:

- `self`: This parameter refers to the instance of the class itself. It is automatically passed when the method is called and is used to access the attributes and methods of the class.

- `endog`: This parameter represents the dependent variable(s) in the multivariate OLS regression. It can be a 1-dimensional or 2-dimensional array-like object.

- `exog`: This parameter represents the independent variable(s) in the multivariate OLS regression. It can be a 1-dimensional or 2-dimensional array-like object.

- `missing`: This parameter specifies how missing values in the data should be handled. The default value is `'none'`, which means missing values are not allowed.

- `hasconst`: This parameter specifies whether a constant term should be included in the regression model. The default value is `None`, which means the presence of a constant term is determined automatically based on the data.

- `**kwargs`: This parameter allows for additional keyword arguments to be passed to the method. It is used to handle any additional parameters that are not explicitly defined.

The purpose of the `__init__` method in this context is to initialize the `_MultivariateOLS` object by calling the `__init__` method of its superclass (`super(_MultivariateOLS, self).__init__`). This ensures that the attributes and methods of the superclass are properly initialized.

The mathematical operations or procedures performed by the `__init__` method are not explicitly stated in the code provided. However, based on the context, it can be inferred that the method is responsible for setting up the multivariate OLS regression model by initializing the dependent and independent variables, handling missing values, and determining the presence of a constant term. The actual mathematical operations for fitting the multivariate OLS regression model are likely performed in other methods of the `_MultivariateOLS` class.

### Method **`fit`** Overview
The `fit` method in Python is used to fit a multivariate ordinary least squares (OLS) regression model to the data. It takes an optional parameter `method` which specifies the method to be used for fitting the model. The default method is 'svd'.

The purpose of each parameter in the `fit` method is as follows:

- `self`: It refers to the instance of the class that the method is being called on. In this case, it refers to the instance of the multivariate OLS regression model.

- `method`: It is an optional parameter that specifies the method to be used for fitting the model. The default value is 'svd'. Other possible values for this parameter could be 'qr' or 'pinv', depending on the desired method for fitting the model.

The `fit` method calls the `_multivariate_ols_fit` function to perform the actual fitting of the model. The fitted model is stored in the `_fittedmod` attribute of the instance.

The `_multivariate_ols_fit` function performs the mathematical operations or procedures required for fitting the multivariate OLS regression model. The specific details of these operations or procedures are not provided in the given code snippet.

The `fit` method then returns an instance of the `_MultivariateOLSResults` class, which encapsulates the results of the fitted model.

To display the equations in a markdown document, the mathematical operations or procedures performed by the `_multivariate_ols_fit` function can be represented using LaTeX code. However, since the specific details of these operations are not provided, it is not possible to generate the LaTeX code for the equations.

## Class **`_MultivariateOLSResults`** Overview
The `_MultivariateOLSResults` class is a Python class that represents the results of a multivariate ordinary least squares (OLS) regression. It has the following attributes and methods:

Attributes:
- `design_info`: The design information of the fitted multivariate OLS model, which includes information about the terms used in the regression.
- `exog_names`: The names of the exogenous variables used in the regression.
- `endog_names`: The names of the endogenous variables used in the regression.
- `_fittedmod`: The fitted multivariate OLS model.

Methods:
- `__str__()`: Returns a string representation of the summary of the multivariate OLS results.
- `mv_test(hypotheses=None, skip_intercept_test=False)`: Performs a multivariate hypothesis test on the regression coefficients. The `hypotheses` parameter allows the user to specify custom hypotheses to test. If `hypotheses` is not provided, default hypotheses are generated based on the design information or the number of exogenous variables. The `skip_intercept_test` parameter determines whether to skip the hypothesis test for the intercept term. Returns a `MultivariateTestResults` object.
- `summary()`: Raises a `NotImplementedError`.

### Method **`__init__`** Overview
The `__init__` method is a special method in Python classes that is automatically called when an object is created from a class. It is used to initialize the attributes of the object.

In the given code snippet, the `__init__` method takes one parameter `fitted_mv_ols`. This parameter represents a fitted multivariate ordinary least squares (OLS) model.

The purpose of the `__init__` method is to initialize the attributes of the object. Here is a breakdown of what each line of code in the method does:

1. `if (hasattr(fitted_mv_ols, 'data') and hasattr(fitted_mv_ols.data, 'design_info')):` checks if the `fitted_mv_ols` object has attributes `data` and `design_info`. This is done using the `hasattr` function.

2. If the condition in the previous line is `True`, it means that the `fitted_mv_ols` object has the required attributes. In this case, `self.design_info` is assigned the value of `fitted_mv_ols.data.design_info`. This line initializes the `design_info` attribute of the object.

3. If the condition in line 1 is `False`, it means that the `fitted_mv_ols` object does not have the required attributes. In this case, `self.design_info` is assigned the value `None`. This line initializes the `design_info` attribute of the object.

4. `self.exog_names = fitted_mv_ols.exog_names` assigns the value of `fitted_mv_ols.exog_names` to the `exog_names` attribute of the object. This line initializes the `exog_names` attribute.

5. `self.endog_names = fitted_mv_ols.endog_names` assigns the value of `fitted_mv_ols.endog_names` to the `endog_names` attribute of the object. This line initializes the `endog_names` attribute.

6. `self._fittedmod = fitted_mv_ols._fittedmod` assigns the value of `fitted_mv_ols._fittedmod` to the `_fittedmod` attribute of the object. This line initializes the `_fittedmod` attribute.

In terms of mathematical operations or procedures, the `__init__` method does not perform any explicit mathematical operations. It mainly initializes the attributes of the object based on the provided `fitted_mv_ols` parameter.

### Method **`__str__`** Overview
The `__str__` method in Python is a special method that is automatically called when we use the `str()` function or the `print()` function on an object. It is used to return a string representation of the object.

In the given code snippet, the `__str__` method is defined within a class and it calls another method called `summary()` to get the string representation of the object. The `summary()` method is assumed to be defined within the same class.

The purpose of the `__str__` method is to provide a human-readable string representation of the object. It is often used for debugging or displaying information about the object.

As for the mathematical operations or procedures performed in the `summary()` method, it is not specified in the given code snippet. Therefore, we cannot generate LaTeX code for any equations or mathematical operations.

### Method **`mv_test`** Overview
The `mv_test` method in Python is used to perform multivariate hypothesis tests in a linear regression model. It takes two parameters: `hypotheses` and `skip_intercept_test`.

- `hypotheses` is a list of hypotheses to be tested. Each hypothesis is represented as a list containing three elements: the name of the hypothesis, the contrast matrix `L_contrast`, and the restriction matrix (which is set to `None` by default).
- `skip_intercept_test` is a boolean parameter that determines whether the intercept term should be skipped in the hypothesis tests. It is set to `False` by default.

The method first checks the number of exogenous variables (`k_xvar`) in the model. If no hypotheses are provided, it generates a list of hypotheses based on the design information of the model. If the design information is available, it iterates over the terms and creates a contrast matrix `L_contrast` for each term. If the `skip_intercept_test` parameter is `True`, it skips the hypothesis test for the intercept term. If the design information is not available, it generates hypotheses for each exogenous variable individually, with a contrast matrix `L` that has a single 1 in the corresponding position.

The method then calls the `_multivariate_ols_test` function to perform the actual hypothesis tests. This function takes the list of hypotheses, the fitted model (`self._fittedmod`), the names of the exogenous variables (`self.exog_names`), and the names of the endogenous variables (`self.endog_names`) as input. The results of the hypothesis tests are stored in the `results` variable.

Finally, the method returns an instance of the `MultivariateTestResults` class, which is initialized with the `results`, `self.endog_names`, and `self.exog_names` as arguments.

The mathematical operations or procedures performed by the `mv_test` method involve creating contrast matrices for the hypotheses and performing hypothesis tests using the `_multivariate_ols_test` function. The contrast matrices are used to specify the restrictions on the coefficients of the linear regression model. The `_multivariate_ols_test` function then calculates the test statistics and p-values for each hypothesis using the fitted model.

### Method **`summary`** Overview
The `summary` method is a placeholder method that raises a `NotImplementedError` exception. It does not take any parameters.

The purpose of this method is to indicate that the functionality of summarizing the object's data or state has not been implemented yet. It serves as a reminder for the developer to implement this method in a subclass or to provide a proper implementation.

As for the mathematical operations or procedures, since the method is not implemented, there are no specific mathematical operations or procedures to document.

Regarding the LaTex code, since there are no mathematical operations or procedures to document, there is no need to generate LaTex code for equations.

## Class **`MultivariateTestResults`** Overview
The `MultivariateTestResults` class is a Python class that represents the results of a multivariate linear model. It has the following attributes and methods:

Attributes:
- `results`: A dictionary containing the results of the multivariate linear model.
- `endog_names`: A list of the names of the endogenous variables.
- `exog_names`: A list of the names of the exogenous variables.

Methods:
- `__init__(self, results, endog_names, exog_names)`: Initializes the `MultivariateTestResults` object with the given results, endogenous variable names, and exogenous variable names.
- `__str__(self)`: Returns a string representation of the `MultivariateTestResults` object.
- `__getitem__(self, item)`: Returns the result corresponding to the given item from the `results` dictionary.
- `summary_frame(self)`: Returns the results as a multiindex dataframe.
- `summary(self, show_contrast_L=False, show_transform_M=False, show_constant_C=False)`: Returns a summary of the multivariate linear model results. The summary includes the statistics for each variable, as well as optional information about contrast, transformation, and constant.

Overall, the `MultivariateTestResults` class provides a convenient way to store and access the results of a multivariate linear model, and also provides methods to generate a summary of the results.

### Method **`__init__`** Overview
The `__init__` method in Python is a special method that is automatically called when an object is created from a class. It is used to initialize the attributes of the object.

In the given code snippet, the `__init__` method takes three parameters: `results`, `endog_names`, and `exog_names`. Here is the purpose of each parameter:

- `results`: This parameter represents the results of a mathematical operation or procedure. It could be a list, array, or any other data structure that holds the results.
- `endog_names`: This parameter represents the names of the endogenous variables in the mathematical model. It could be a list or any other iterable containing the names.
- `exog_names`: This parameter represents the names of the exogenous variables in the mathematical model. Similar to `endog_names`, it could be a list or any other iterable containing the names.

Inside the `__init__` method, the given code snippet performs the following operations:

1. `self.results = results`: This line assigns the value of the `results` parameter to the `results` attribute of the object. This allows the object to store and access the results of the mathematical operation.

2. `self.endog_names = list(endog_names)`: This line converts the `endog_names` parameter into a list and assigns it to the `endog_names` attribute of the object. This allows the object to store and access the names of the endogenous variables.

3. `self.exog_names = list(exog_names)`: This line converts the `exog_names` parameter into a list and assigns it to the `exog_names` attribute of the object. This allows the object to store and access the names of the exogenous variables.

To display the equations in a markdown document using LaTeX, you can use the following code:

```latex

$$
\text{{self.results}} = \text{{results}}
$$


$$
\text{{self.endog\_names}} = \text{{list(endog\_names)}}
$$


$$
\text{{self.exog\_names}} = \text{{list(exog\_names)}}
$$
```

This LaTeX code will render the equations in a mathematical format when used in a markdown document.

### Method **`__str__`** Overview
The `__str__` method in Python is a special method that is automatically called when we use the `str()` function or the `print()` function on an object. It is used to return a string representation of the object.

In the given code snippet, the `__str__` method is defined within a class and it calls another method called `summary()` to get the string representation of the object. The `summary()` method is assumed to be defined within the same class.

The purpose of the `__str__` method is to provide a human-readable string representation of the object. It is often used for debugging or displaying information about the object.

As for the mathematical operations or procedures performed in the `summary()` method, it is not specified in the given code snippet. Therefore, we cannot generate LaTeX code for any equations or mathematical operations.

### Method **`__getitem__`** Overview
The `__getitem__` method in Python is a special method that allows instances of a class to be accessed using the square bracket notation. It is used to define the behavior of the object when it is indexed or sliced using square brackets.

The method takes two parameters:
- `self`: It is a reference to the instance of the class.
- `item`: It represents the index or slice that is being accessed.

The purpose of the `__getitem__` method is to retrieve the value associated with the given index or slice from the object. In the provided code snippet, it returns the value of `self.results[item]`.

The mathematical operations or procedures performed by the `__getitem__` method depend on the specific implementation of the class. It could be used to retrieve a specific element from a list, dictionary, or any other data structure stored in the `self.results` attribute.

To display mathematical equations in a markdown document, you can use LaTeX code. Here's an example of how you can represent a mathematical equation using LaTeX:

```latex

$$ equation $$
```

For example, if the `__getitem__` method is used to retrieve an element from a list, you can represent it in LaTeX as:

```latex

$$ \text{{element}} = self.results[item] $$
```

This will display the equation as:


$$ \text{element} = self.results[item] $$

### Method **`summary_frame`** Overview
The `summary_frame` method in Python is used to return the results of a statistical analysis as a multiindex dataframe. 

Parameters:
- `self`: The instance of the class that the method is being called on.

Mathematical operations or procedures:
1. Create an empty list called `df`.
2. Iterate over each key in the `results` dictionary.
3. Create a copy of the `stat` dataframe for the current key and assign it to the variable `tmp`.
4. Add a new column called 'Effect' to the `tmp` dataframe and set its values to the current key.
5. Reset the index of the `tmp` dataframe and append it to the `df` list.
6. Concatenate all the dataframes in the `df` list along the axis 0 to create a single dataframe.
7. Set the index of the dataframe to be a multiindex with levels 'Effect' and 'index'.
8. Set the names of the index levels to 'Effect' and 'Statistic'.
9. Return the resulting dataframe.

LaTeX code for the mathematical operations or procedures:

1. Create an empty list called `df`: 


$$
\text{{df = []}}
$$

2. Iterate over each key in the `results` dictionary: 


$$
\text{{for key in self.results:}}
$$

3. Create a copy of the `stat` dataframe for the current key and assign it to the variable `tmp`: 


$$
\text{{tmp = self.results[key]['stat'].copy()}}
$$

4. Add a new column called 'Effect' to the `tmp` dataframe and set its values to the current key: 


$$
\text{{tmp.loc[:, 'Effect'] = key}}
$$

5. Reset the index of the `tmp` dataframe and append it to the `df` list: 


$$
\text{{df.append(tmp.reset\_index())}}
$$

6. Concatenate all the dataframes in the `df` list along the axis 0 to create a single dataframe: 


$$
\text{{df = pd.concat(df, axis=0)}}
$$

7. Set the index of the dataframe to be a multiindex with levels 'Effect' and 'index': 


$$
\text{{df = df.set\_index(['Effect', 'index'])}}
$$

8. Set the names of the index levels to 'Effect' and 'Statistic': 


$$
\text{{df.index.set\_names(['Effect', 'Statistic'], inplace=True)}}
$$

9. Return the resulting dataframe: 


$$
\text{{return df}}
$$

### Method **`summary`** Overview
The `summary` method is a method of a Python class. It takes three optional parameters: `show_contrast_L`, `show_transform_M`, and `show_constant_C`. These parameters determine whether or not to include certain information in the summary.

The purpose of this method is to generate a summary of a multivariate linear model. It creates an instance of the `Summary` class and adds various components to it, such as titles, dataframes, and dictionaries.

The method iterates over the `results` attribute of the class, which is assumed to be a dictionary. For each key in the `results` dictionary, it performs the following steps:

1. Creates a new dataframe `df` by copying the `stat` attribute of the corresponding value in the `results` dictionary.
2. Resets the index of `df`.
3. Modifies the column names of `df` by replacing the first column name with the current key.
4. Modifies the index of `df` by setting it to four empty strings.
5. Adds `df` to the `summ` instance of the `Summary` class.

If the `show_contrast_L` parameter is `True`, the method performs the following steps:

1. Adds a dictionary entry to `summ` with the key as the current key and the value as `' contrast L='`.
2. Creates a new dataframe `df` using the `'contrast_L'` attribute of the current value in the `results` dictionary.
3. Adds `df` to the `summ` instance of the `Summary` class.

If the `show_transform_M` parameter is `True`, the method performs the following steps:

1. Adds a dictionary entry to `summ` with the key as the current key and the value as `' transform M='`.
2. Creates a new dataframe `df` using the `'transform_M'` attribute of the current value in the `results` dictionary.
3. Sets the index of `df` to the `endog_names` attribute of the class.
4. Adds `df` to the `summ` instance of the `Summary` class.

If the `show_constant_C` parameter is `True`, the method performs the following steps:

1. Adds a dictionary entry to `summ` with the key as the current key and the value as `' constant C='`.
2. Creates a new dataframe `df` using the `'constant_C'` attribute of the current value in the `results` dictionary.
3. Adds `df` to the `summ` instance of the `Summary` class.

Finally, the method returns the `summ` instance of the `Summary` class.

The mathematical operations or procedures performed by this method are not explicitly stated in the code. It appears to be mainly focused on creating a summary of the multivariate linear model and organizing the relevant information into a `Summary` object.

