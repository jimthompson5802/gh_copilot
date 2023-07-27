# Module:`multivariate_ols.py` Overview

The module contains functions and classes for performing multivariate linear regression analysis. It includes functions for fitting the model, conducting hypothesis tests, and generating summary statistics. The main class in the module is `_MultivariateOLS`, which represents the multivariate linear regression model. The `fit` method of this class fits the model to the data. The `mv_test` method conducts hypothesis tests on the model. The module also includes a class `_MultivariateOLSResults` for storing the results of the model fitting and hypothesis tests. The `summary` method of this class generates a summary of the results. The module also includes a class `MultivariateTestResults` for storing the results of multiple hypothesis tests. The `summary` method of this class generates a summary of the test results.

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
- `df_resid`: A scalar representing the residual degrees of freedom.
- `tolerance`: A scalar representing the tolerance value for determining if an eigenvalue is greater than zero.

The function performs several mathematical operations and procedures to calculate various statistics related to multivariate analysis. Here is a breakdown of the mathematical operations performed:

1. The function initializes variables `v`, `p`, `q`, and `s` based on the input parameters.
2. It creates a boolean array `ind` by checking if each eigenvalue is greater than the tolerance value.
3. It calculates the number of eigenvalues greater than the tolerance value and assigns it to `n_e`.
4. It creates two new arrays `eigv2` and `eigv1` by filtering the eigenvalues based on the boolean array `ind` and performing element-wise operations.
5. It calculates the values for the "Wilks' lambda", "Pillai's trace", "Hotelling-Lawley trace", and "Roy's greatest root" statistics using the filtered eigenvalues and assigns them to the corresponding cells in a pandas DataFrame called `results`.
6. It calculates the values for the degrees of freedom and F-statistics for each statistic using the input parameters and the calculated values from the previous step.
7. It calculates the p-values for each F-statistic using the calculated F-values and the degrees of freedom.
8. It assigns the calculated degrees of freedom, F-values, and p-values to the corresponding cells in the `results` DataFrame.
9. Finally, it returns the `results` DataFrame.

Here is the LaTex code for the equations used in the function:

1. Wilks' lambda:

$$
\text{{Wilks' lambda}} = \prod_{i=1}^{n_e} (1 - \text{{eigv2}}_i)
$$

2. Pillai's trace:
$
\text{{Pillai's trace}} = \sum_{i=1}^{n_e} \text{{eigv2}}_i
$

3. Hotelling-Lawley trace:
\[
\text{{Hotelling-Lawley trace}} = \sum_{i=1}^{n_e} \frac{{\text{{eigv2}}_i}}{{1 - \text{{eigv2}}_i}}
\]

4. Roy's greatest root:
\[
\text{{Roy's greatest root}} = \max_{i=1}^{n_e} \frac{{\text{{eigv2}}_i}}{{1 - \text{{eigv2}}_i}}
\]

5. Wilks' lambda F-value:
\[
F = \frac{{(1 - \text{{Wilks' lambda}})}}{{\text{{Wilks' lambda}}}} \times \frac{{\text{{df2}}}}{{\text{{df1}}}}
\]

6. Pillai's trace F-value:
\[
F = \frac{{\text{{df2}}}}{{\text{{df1}}}} \times \frac{{\text{{Pillai's trace}}}}{{(\text{{s}} - \text{{Pillai's trace}})}}
\]

7. Hotelling-Lawley trace F-value:
\[
F = \frac{{\text{{df2}}}}{{\text{{df1}}}} \times \frac{{\text{{Hotelling-Lawley trace}}}}{{\text{{c}}}}
\]

8. Roy's greatest root F-value:
\[
F = \frac{{\text{{df2}}}}{{\text{{df1}}}} \times \text{{Roy's greatest root}}
\]

Note: The LaTex code provided above can be used to display the equations in a markdown document.

## Function **`_multivariate_ols_test`** Overview
The function `_multivariate_ols_test` is a Python function that performs a multivariate ordinary least squares (OLS) test. It takes four parameters: `hypotheses`, `fit_results`, `exog_names`, and `endog_names`.

- `hypotheses` is a list of hypothesis matrices. Each hypothesis matrix represents a linear combination of the regression coefficients that is being tested. The dimensions of each hypothesis matrix should match the dimensions of the regression coefficient matrix.
- `fit_results` is a tuple containing the results of the OLS regression. It includes the estimated regression coefficients (`params`), the degrees of freedom of the residuals (`df_resid`), the inverse of the covariance matrix (`inv_cov`), and the sum of squares and cross-products matrix (`sscpr`).
- `exog_names` is a list of names for the exogenous variables (independent variables) in the regression model.
- `endog_names` is a list of names for the endogenous variable (dependent variable) in the regression model.

The function defines an inner function `fn` that takes three parameters: `L`, `M`, and `C`. These parameters represent the hypothesis matrix, the projection matrix, and the constant term, respectively.

The function then performs the following mathematical operations:

1. Calculate `t1` by multiplying the hypothesis matrix `L` with the estimated regression coefficients `params` and the projection matrix `M`, and subtracting the constant term `C`. This can be represented mathematically as:

\[
t1 = L \cdot \text{{params}} \cdot M - C
\]

2. Calculate `t2` by multiplying the hypothesis matrix `L` with the inverse of the covariance matrix `inv_cov`, and then multiplying the result by the transpose of `L`. This can be represented mathematically as:

\[
t2 = L \cdot \text{{inv_cov}} \cdot L^T
\]

3. Calculate the rank of `t2` using the `matrix_rank` function.
4. Calculate `H` by taking the dot product of `t1` (transposed) with the inverse of `t2`, and then taking the dot product of the result with `t1`. This can be represented mathematically as:

\[
H = t1^T \cdot \text{{inv}}(t2) \cdot t1
\]

5. Calculate `E` by taking the dot product of `M` (transposed) with the sum of squares and cross-products matrix `sscpr`, and then taking the dot product of the result with `M`. This can be represented mathematically as:

\[
E = M^T \cdot \text{{sscpr}} \cdot M
\]

6. Return the values of `E`, `H`, `q` (rank of `t2`), and `df_resid` (degrees of freedom of the residuals).

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

6. If `M` is `None`, assigns it as an identity matrix with the number of rows equal to the number of endogenous variables. If `M` is a string, converts it to a transform matrix. Otherwise, checks if `M` is a 2-dimensional numpy array with the correct number of rows.

7. If `C` is `None`, assigns it as a zero matrix with the same number of rows as `L` and the same number of columns as `M`. Otherwise, checks if `C` is a 2-dimensional numpy array.

8. Checks if the number of rows of `C` is the same as the number of rows of `L` and if the number of columns of `C` is the same as the number of columns of `M`.

9. Calls the `fn` function with the contrast matrix `L`, transform matrix `M`, and constant matrix `C` as parameters to calculate the error matrix `E`, hypothesis matrix `H`, degrees of freedom `q`, and residual degrees of freedom `df_resid`.

10. Adds the error matrix `E` and hypothesis matrix `H` element-wise to obtain `EH`.

11. Calculates the rank of `EH` to obtain the number of non-zero eigenvalues.

12. Sorts the eigenvalues of the inverse of `EH` multiplied by `H` in ascending order.

13. Calls the `multivariate_stats` function with the sorted eigenvalues, number of non-zero eigenvalues, degrees of freedom `q`, and residual degrees of freedom `df_resid` to obtain the test statistics.

14. Stores the test statistics, contrast matrix `L`, transform matrix `M`, constant matrix `C`, error matrix `E`, and hypothesis matrix `H` in the `results` dictionary with the hypothesis name as the key.

15. Returns the `results` dictionary.

The mathematical operations performed by the function include checking the dimensions and types of the input matrices, calculating the error matrix, hypothesis matrix, and test statistics, and storing the results in a dictionary.

## Class **`_MultivariateOLS`** Overview
The `_MultivariateOLS` class is a subclass of the `Model` class in Python. It is used to fit multivariate ordinary least squares (OLS) models. 

The class has an attribute `_formula_max_endog` which is set to `None`. 

The `__init__` method is the constructor of the class. It takes the arguments `endog`, `exog`, `missing`, `hasconst`, and `**kwargs`. It checks if the shape of `endog` is either 1-dimensional or has only 1 column. If this condition is met, it raises a `ValueError` indicating that there must be more than one dependent variable to fit multivariate OLS. It then calls the constructor of the superclass `Model` with the given arguments.

The `fit` method is used to fit the multivariate OLS model. It takes an optional argument `method` which defaults to `'svd'`. It fits the model using the `_multivariate_ols_fit` function, passing the `endog`, `exog`, and `method` arguments. It then returns an instance of the `_MultivariateOLSResults` class, passing itself as an argument.

### Method **`__init__`** Overview
The `__init__` method is a special method in Python classes that is automatically called when an object is created. It is used to initialize the object's attributes. In the case of the `_MultivariateOLS` class, the `__init__` method is used to initialize the object with the given parameters.

Parameters:
- `self`: The object itself.
- `endog`: The dependent variable(s) of the model. It can be a 1-dimensional or 2-dimensional array-like object.
- `exog`: The independent variable(s) of the model. It can be a 1-dimensional or 2-dimensional array-like object.
- `missing`: Specifies how missing values are handled. The default value is `'none'`, which means missing values are not handled.
- `hasconst`: Specifies whether a constant term should be included in the model. The default value is `None`, which means it is determined automatically.
- `**kwargs`: Additional keyword arguments that can be passed to the superclass constructor.

The `__init__` method performs the following mathematical operations or procedures:
1. It checks the shape of the `endog` variable to ensure that it has more than one dependent variable. If it is a 1-dimensional array or has only one column, it raises a `ValueError` with an appropriate error message.
2. It calls the `__init__` method of the superclass (`super(_MultivariateOLS, self).__init__`) to initialize the object with the given parameters. The superclass is likely the `OLS` class, which is a base class for ordinary least squares regression models.

Unfortunately, the provided code does not contain any mathematical equations or procedures that can be represented using LaTeX code.

### Method **`fit`** Overview
The `fit` method in Python is used to fit a multivariate ordinary least squares (OLS) regression model. It takes one parameter, `method`, which specifies the method to be used for fitting the model. The default value for `method` is 'svd'.

The purpose of the `fit` method is to calculate the coefficients of the OLS regression model using the specified method. It calls the `_multivariate_ols_fit` function, passing the endogenous variable (`self.endog`), the exogenous variable (`self.exog`), and the method as arguments. The result of the fitting process is stored in the `_fittedmod` attribute of the object.

The `_multivariate_ols_fit` function performs the mathematical operations required to fit the OLS regression model. The specific mathematical operations or procedures depend on the method chosen. The available methods include 'svd' (singular value decomposition), 'qr' (QR decomposition), and 'pinv' (Moore-Penrose pseudoinverse). The function returns the fitted model.

To display the equations in a markdown document, you can use LaTeX code. Here is an example of LaTeX code that can be used to display the equations for the OLS regression model:

```latex
\[
\hat{y} = X\hat{\beta}
\]

\[
\hat{\beta} = (X^TX)^{-1}X^Ty
\]

where:
- \(\hat{y}\) is the predicted values
- \(X\) is the design matrix of the exogenous variables
- \(\hat{\beta}\) is the estimated coefficients
- \(y\) is the endogenous variable
\]
```

Note that the actual equations may vary depending on the specific method used for fitting the model.

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

The `__init__` method does not perform any mathematical operations or procedures. It is solely responsible for initializing the attributes of the object based on the provided `fitted_mv_ols` parameter.

Here is the LaTex code for the equations in a markdown document:

\[
\text{{self.design\_info}} = \text{{fitted\_mv\_ols.data.design\_info}}
\]

\[
\text{{self.exog\_names}} = \text{{fitted\_mv\_ols.exog\_names}}
\]

\[
\text{{self.endog\_names}} = \text{{fitted\_mv\_ols.endog\_names}}
\]

\[
\text{{self.\_fittedmod}} = \text{{fitted\_mv\_ols.\_fittedmod}}
\]

### Method **`__str__`** Overview
The `__str__` method in Python is a special method that is automatically called when we use the `str()` function or the `print()` function on an object. It is used to return a string representation of the object.

In the given code snippet, the `__str__` method is defined within a class and it calls another method called `summary()` to get the string representation of the object. The `summary()` method is assumed to be defined within the same class.

The purpose of the `__str__` method is to provide a human-readable string representation of the object. It is often used for debugging or displaying information about the object.

As for the mathematical operations or procedures performed in the `summary()` method, it is not specified in the given code snippet. Therefore, we cannot generate LaTeX code for any equations or mathematical operations.

### Method **`mv_test`** Overview
The `mv_test` method in Python is used to perform multivariate hypothesis tests in a linear regression model. It takes two parameters: `hypotheses` and `skip_intercept_test`.

- `hypotheses` (optional): This parameter allows the user to specify the hypotheses to be tested. It is a list of lists, where each inner list contains three elements: the name of the hypothesis, the contrast matrix `L_contrast`, and the restriction matrix `R_restriction`. If no hypotheses are provided, the method will automatically generate them based on the design information of the model or the number of exogenous variables.
- `skip_intercept_test` (optional): This parameter is a boolean value that determines whether the intercept term should be included in the hypothesis tests. If set to `True`, the method will skip the hypothesis test for the intercept term.

The method first checks the number of exogenous variables (`k_xvar`) in the model. If no hypotheses are provided, it generates them based on the design information or the number of exogenous variables. For each exogenous variable, it creates a hypothesis with a contrast matrix `L` that has a single 1 in the corresponding position and zeros elsewhere.

The method then calls the `_multivariate_ols_test` function to perform the actual hypothesis tests. This function takes the hypotheses, the fitted model (`self._fittedmod`), the names of the exogenous variables (`self.exog_names`), and the names of the endogenous variables (`self.endog_names`) as input. It returns the results of the hypothesis tests.

Finally, the method creates a `MultivariateTestResults` object using the results of the hypothesis tests, the names of the endogenous variables, and the names of the exogenous variables. This object is then returned by the method.

The mathematical operations performed by the method include generating contrast matrices `L_contrast` and restriction matrices `R_restriction` for each hypothesis. These matrices are used in the hypothesis tests to compare the coefficients of the exogenous variables. The method also calls the `_multivariate_ols_test` function to perform the hypothesis tests using the provided or generated hypotheses.

### Method **`summary`** Overview
The `summary` method is a placeholder method that raises a `NotImplementedError`. It does not have any parameters and does not perform any mathematical operations or procedures.

LaTeX code for displaying the method in a markdown document:

```latex
\begin{verbatim}
def summary(self):
    raise NotImplementedError
\end{verbatim}
```

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

- `results`: This parameter represents the results of a mathematical operation or procedure. It can be any data type that holds the results.
- `endog_names`: This parameter represents the names of the endogenous variables in the mathematical model. It is expected to be an iterable (e.g., list, tuple) containing the names.
- `exog_names`: This parameter represents the names of the exogenous variables in the mathematical model. It is also expected to be an iterable containing the names.

The method performs the following mathematical operations or procedures:

1. Assigns the value of the `results` parameter to the `self.results` attribute. This allows the results to be accessed and manipulated within the object.
2. Converts the `endog_names` parameter to a list and assigns it to the `self.endog_names` attribute. This allows the endogenous variable names to be accessed and manipulated within the object.
3. Converts the `exog_names` parameter to a list and assigns it to the `self.exog_names` attribute. This allows the exogenous variable names to be accessed and manipulated within the object.

Here is the LaTex code to display the equations in a markdown document:

1. $self.results = results$
2. $self.endog\_names = list(endog\_names)$
3. $self.exog\_names = list(exog\_names)$

### Method **`__str__`** Overview
The `__str__` method in Python is a special method that is automatically called when we use the `str()` function or the `print()` function on an object. It is used to return a string representation of the object.

In the given code snippet, the `__str__` method is defined within a class and it calls another method called `summary()` to get the string representation of the object. The `summary()` method is assumed to be defined within the same class.

The purpose of the `__str__` method is to provide a human-readable string representation of the object. It is often used for debugging or displaying information about the object.

As for the mathematical operations or procedures performed in the `summary()` method, it is not specified in the given code snippet. Therefore, we cannot generate LaTeX code for mathematical equations as there is no information available about the operations or procedures involved.

### Method **`__getitem__`** Overview
The `__getitem__` method in Python is a special method that allows instances of a class to be accessed using the square bracket notation. It is used to define the behavior of the object when it is indexed or sliced using square brackets.

The method takes two parameters:
- `self`: It is a reference to the instance of the class.
- `item`: It represents the index or slice that is being accessed.

The purpose of the `__getitem__` method is to retrieve the value associated with the given index or slice from the object. In the provided code snippet, it returns the value of `self.results[item]`.

The mathematical operations or procedures performed by the `__getitem__` method depend on the specific implementation of the class. Since the code snippet only returns the value from `self.results` based on the provided index or slice, there are no specific mathematical operations or procedures documented.

To display equations in LaTeX format in a markdown document, you can use the following syntax:

```
$$
\text{{equation}}
$$
```

Replace `\text{{equation}}` with the actual LaTeX code representing the equation.

### Method **`summary_frame`** Overview
The `summary_frame` method in Python is used to return the results of a statistical analysis as a multiindex dataframe. 

Parameters:
- `self`: The instance of the class that the method belongs to.

Mathematical operations or procedures:
1. Create an empty list called `df`.
2. Iterate over each key in the `results` dictionary.
3. Create a copy of the `stat` dataframe for the current key and assign it to the variable `tmp`.
4. Add a new column called 'Effect' to the `tmp` dataframe and set its values to the current key.
5. Reset the index of the `tmp` dataframe and append it to the `df` list.
6. Concatenate all the dataframes in the `df` list along the axis 0 to create a single dataframe.
7. Set the index of the dataframe to be a multiindex with the levels 'Effect' and 'index'.
8. Set the names of the index levels to be 'Effect' and 'Statistic'.
9. Return the resulting dataframe.

LaTeX code for the mathematical operations or procedures:

1. Create an empty list called `df`: 

\[
\text{{df = []}}
\]

2. Iterate over each key in the `results` dictionary: 

\[
\text{{for key in self.results:}}
\]

3. Create a copy of the `stat` dataframe for the current key and assign it to the variable `tmp`: 

\[
\text{{tmp = self.results[key]['stat'].copy()}}
\]

4. Add a new column called 'Effect' to the `tmp` dataframe and set its values to the current key: 

\[
\text{{tmp.loc[:, 'Effect'] = key}}
\]

5. Reset the index of the `tmp` dataframe and append it to the `df` list: 

\[
\text{{df.append(tmp.reset\_index())}}
\]

6. Concatenate all the dataframes in the `df` list along the axis 0 to create a single dataframe: 

\[
\text{{df = pd.concat(df, axis=0)}}
\]

7. Set the index of the dataframe to be a multiindex with the levels 'Effect' and 'index': 

\[
\text{{df = df.set\_index(['Effect', 'index'])}}
\]

8. Set the names of the index levels to be 'Effect' and 'Statistic': 

\[
\text{{df.index.set\_names(['Effect', 'Statistic'], inplace=True)}}
\]

9. Return the resulting dataframe: 

\[
\text{{return df}}
\]

### Method **`summary`** Overview
The `summary` method is a method of a Python class. It takes three optional parameters: `show_contrast_L`, `show_transform_M`, and `show_constant_C`. These parameters determine whether or not to include certain information in the summary.

The purpose of this method is to generate a summary of a multivariate linear model. It creates an instance of the `Summary` class and adds a title to the summary indicating that it is a multivariate linear model.

For each key in the `results` attribute of the class, the method adds a dictionary with an empty key-value pair to the summary. It then creates a copy of the statistical results associated with that key and resets the index of the resulting DataFrame. The first column of the DataFrame is renamed to the current key, and the index is set to empty strings.

The method then adds the DataFrame to the summary. If the `show_contrast_L` parameter is `True`, it adds a dictionary with the key-value pair indicating that the contrast matrix `L` is included. It creates a DataFrame from the `contrast_L` attribute of the `results` dictionary and adds it to the summary.

If the `show_transform_M` parameter is `True`, it adds a dictionary with the key-value pair indicating that the transform matrix `M` is included. It creates a DataFrame from the `transform_M` attribute of the `results` dictionary and adds it to the summary.

If the `show_constant_C` parameter is `True`, it adds a dictionary with the key-value pair indicating that the constant vector `C` is included. It creates a DataFrame from the `constant_C` attribute of the `results` dictionary and adds it to the summary.

Finally, the method returns the `Summary` instance.

The mathematical operations or procedures performed by this method involve creating DataFrames from the statistical results of the multivariate linear model and adding them to the summary. The method does not perform any specific mathematical operations or procedures itself.

Here is the LaTex code for the equations in the summary:

- Contrast matrix L: $L = \begin{bmatrix} l_{11} & l_{12} & \ldots & l_{1n} \\ l_{21} & l_{22} & \ldots & l_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ l_{m1} & l_{m2} & \ldots & l_{mn} \end{bmatrix}$

- Transform matrix M: $M = \begin{bmatrix} m_{11} & m_{12} & \ldots & m_{1n} \\ m_{21} & m_{22} & \ldots & m_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ m_{p1} & m_{p2} & \ldots & m_{pn} \end{bmatrix}$

- Constant vector C: $C = \begin{bmatrix} c_1 \\ c_2 \\ \vdots \\ c_n \end{bmatrix}$

