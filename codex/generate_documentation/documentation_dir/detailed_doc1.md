# Module:`multivariate_ols.py` Overview

The module contains functions and classes for performing multivariate linear regression analysis. It includes functions for fitting the model, conducting hypothesis tests, and generating summary statistics. The main class in the module is `_MultivariateOLS`, which represents a multivariate linear regression model. The `fit` method of this class fits the model to the data. The `mv_test` method conducts hypothesis tests on the model. The module also includes a class `_MultivariateOLSResults`, which represents the results of a fitted multivariate linear regression model. The `summary` method of this class generates a summary of the results. The module also includes a class `MultivariateTestResults`, which represents the results of hypothesis tests conducted on a multivariate linear regression model. The `summary` method of this class generates a summary of the test results.

## Function **`_multivariate_ols_fit`** Overview
The `_multivariate_ols_fit` function is a Python function that performs multivariate ordinary least squares (OLS) regression. It takes the following parameters:

- `endog`: The dependent variable(s) in the regression model. It should be a 2-dimensional array-like object with shape (nobs, k_endog), where `nobs` is the number of observations and `k_endog` is the number of dependent variables.

- `exog`: The independent variable(s) in the regression model. It should be a 2-dimensional array-like object with shape (nobs, k_exog), where `nobs` is the number of observations and `k_exog` is the number of independent variables.

- `method`: The method used to calculate the regression coefficients and other statistics. It can be either 'pinv' (default) or 'svd'.

- `tolerance`: The tolerance level used to check for singularity of the covariance matrix. It is set to 1e-8 by default.

The function performs the following mathematical operations or procedures:

1. It checks if the number of observations in `endog` and `exog` are the same. If not, it raises a `ValueError` indicating that the number of rows should be the same.

2. If the `method` is 'pinv', it calculates the regression coefficients matrix (`params`) using the Moore-Penrose pseudo-inverse of `exog` (`pinv_x`). It also calculates the inverse of the covariance matrix (`inv_cov`) using the pseudo-inverse. If the rank of `inv_cov` is less than `k_exog`, it raises a `ValueError` indicating that the covariance of `exog` is singular. Finally, it calculates the sums of squares and cross-products of residuals (`sscpr`) using the formula Y'Y - (X * params)'B * params, where Y is `endog`, X is `exog`, and B is the matrix of regression coefficients.

3. If the `method` is 'svd', it performs singular value decomposition (SVD) on `exog` to obtain the matrices U, S, and V. It checks if any of the singular values in S are smaller than the tolerance level. If so, it raises a `ValueError` indicating that the covariance of `exog` is singular. It then calculates the regression coefficients matrix (`params`) using the formula V' * inv(S) * U' * Y, where V, S, and U are the SVD matrices. It also calculates the inverse of the covariance matrix (`inv_cov`) using the formula V' * inv(S^2) * V, where S^2 is obtained by squaring the singular values. Finally, it calculates the sums of squares and cross-products of residuals (`sscpr`) using the same formula as in the 'pinv' method.

4. If the `method` is neither 'pinv' nor 'svd', it raises a `ValueError` indicating that the method is not supported.

The function returns a tuple containing the regression coefficients (`params`), the degrees of freedom of the residuals (`df_resid`), the inverse of the covariance matrix (`inv_cov`), and the sums of squares and cross-products of residuals (`sscpr`).

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
5. It calculates the values for the "Wilks' lambda", "Pillai's trace", "Hotelling-Lawley trace", and "Roy's greatest root" statistics using the filtered eigenvalues and assigns them to the corresponding cells in the `results` DataFrame.
6. It calculates the values for the "Wilks' lambda" statistic using the input parameters `v`, `p`, and `q`, and assigns them to the corresponding cells in the `results` DataFrame.
7. It calculates the values for the "Pillai's trace" statistic using the input parameters `s`, `m`, `n`, and `V`, and assigns them to the corresponding cells in the `results` DataFrame.
8. It calculates the values for the "Hotelling-Lawley trace" statistic using the input parameters `n`, `p`, `q`, `U`, `b`, and `c`, and assigns them to the corresponding cells in the `results` DataFrame.
9. It calculates the values for the "Roy's greatest root" statistic using the input parameters `p`, `q`, `sigma`, `r`, and assigns them to the corresponding cells in the `results` DataFrame.
10. It calculates the p-values for each statistic using the `stats.f.sf` function and assigns them to the corresponding cells in the `results` DataFrame.
11. Finally, it returns the `results` DataFrame containing the calculated statistics.

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

Note: The LaTex code provided above can be used to display the equations in a markdown document.

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

- `fn` is a function that performs the necessary calculations to obtain the error matrix `E`, hypothesis matrix `H`, degrees of freedom `df_resid`, and the number of constraints `q`. This function takes in the contrast matrix `L`, transform matrix `M`, and constant matrix `C` as parameters.

The function then iterates over each hypothesis in the `hypotheses` list and performs the following steps:

1. Checks the length of the hypothesis tuple to determine the number of constraints specified.

2. If the contrast matrix `L` is specified as a string, it converts it to a contrast matrix using the `DesignInfo` class.

3. If the transform matrix `M` is not specified, it sets it to an identity matrix with the same number of rows as the number of endogenous variables.

4. If the transform matrix `M` is specified as a string, it converts it to a transform matrix using the `DesignInfo` class.

5. If the constant matrix `C` is not specified, it sets it to a zero matrix with the same dimensions as the contrast matrix `L`.

6. Checks the dimensions of the contrast matrix `L`, transform matrix `M`, and constant matrix `C` to ensure they are compatible.

7. Calls the provided function `fn` with the contrast matrix `L`, transform matrix `M`, and constant matrix `C` as parameters to obtain the error matrix `E`, hypothesis matrix `H`, degrees of freedom `df_resid`, and the number of constraints `q`.

8. Calculates the sum of the error matrix `E` and hypothesis matrix `H` and computes the rank of the resulting matrix.

9. Computes the eigenvalues of the inverse of the sum of the error matrix `E` and hypothesis matrix `H` multiplied by the hypothesis matrix `H`.

10. Calls the `multivariate_stats` function with the eigenvalues, number of constraints `q`, degrees of freedom `df_resid`, and the rank of the sum of the error matrix `E` and hypothesis matrix `H` to obtain a statistical table.

11. Stores the results in a dictionary with the hypothesis name as the key and the statistical table, contrast matrix `L`, transform matrix `M`, constant matrix `C`, error matrix `E`, and hypothesis matrix `H` as the values.

12. Returns the dictionary of results.

The mathematical operations performed include matrix operations such as matrix addition, matrix multiplication, matrix inversion, and eigenvalue computation. The function also checks the dimensions of the matrices to ensure compatibility.

## Class **`_MultivariateOLS`** Overview
The `_MultivariateOLS` class is a subclass of the `Model` class in Python. It is used to fit multivariate ordinary least squares (OLS) models. 

The class has an attribute `_formula_max_endog` which is set to `None`. 

The `__init__` method is the constructor of the class. It takes the arguments `endog`, `exog`, `missing`, `hasconst`, and `**kwargs`. It checks if the shape of `endog` is either 1-dimensional or has only 1 column. If this condition is met, it raises a `ValueError` indicating that there must be more than one dependent variable to fit multivariate OLS. It then calls the constructor of the superclass `Model` with the given arguments.

The `fit` method is used to fit the multivariate OLS model. It takes an optional argument `method` which defaults to `'svd'`. It fits the model using the `_multivariate_ols_fit` function, passing the `endog`, `exog`, and `method` arguments. It then returns an instance of the `_MultivariateOLSResults` class, passing itself as an argument.

### Method **`__init__`** Overview
The `__init__` method is a special method in Python classes that is automatically called when an object is created. In the case of the `_MultivariateOLS` class, this method is used to initialize an instance of the class.

The method takes several parameters:

- `self`: This parameter refers to the instance of the class itself. It is automatically passed when the method is called and is used to access the attributes and methods of the class.

- `endog`: This parameter represents the dependent variable(s) in the multivariate OLS model. It can be a 1-dimensional or 2-dimensional array-like object.

- `exog`: This parameter represents the independent variable(s) in the multivariate OLS model. It can be a 1-dimensional or 2-dimensional array-like object.

- `missing`: This parameter specifies how missing values in the data should be handled. The default value is `'none'`, which means missing values are not allowed. Other possible values include `'drop'` (to drop any rows with missing values) and `'raise'` (to raise an error if there are any missing values).

- `hasconst`: This parameter specifies whether a constant term should be included in the model. The default value is `None`, which means the presence of a constant term is determined automatically based on the data. Other possible values include `True` (to include a constant term) and `False` (to exclude a constant term).

- `**kwargs`: This parameter allows for additional keyword arguments to be passed to the superclass constructor.

The method first checks the shape of the `endog` parameter to ensure that it has more than one dependent variable. If the shape is 1-dimensional or the second dimension is 1, a `ValueError` is raised.

Next, the method calls the `__init__` method of the superclass (`super(_MultivariateOLS, self).__init__`) to initialize the instance with the `endog`, `exog`, `missing`, and `hasconst` parameters. The `super()` function is used to refer to the superclass of the current class.

The mathematical operations or procedures performed by this method are not explicitly stated in the code provided. However, based on the context, it can be inferred that the method is used to set up the multivariate OLS model by initializing the dependent and independent variables, handling missing values, and determining whether a constant term should be included. The actual mathematical operations for fitting the model and estimating the coefficients are likely performed in other methods of the class.

### Method **`fit`** Overview
The `fit` method in Python is used to fit a multivariate ordinary least squares (OLS) regression model. It takes an optional parameter `method` which specifies the method to be used for fitting the model. The default method is 'svd'.

The purpose of each parameter in the `fit` method is as follows:

- `self`: It refers to the instance of the class that the method is being called on. In this case, it refers to the instance of the multivariate OLS regression model.

- `method`: It is an optional parameter that specifies the method to be used for fitting the model. The default value is 'svd'. Other possible values for this parameter could be 'pinv' or 'qr'.

The `fit` method calls the `_multivariate_ols_fit` function to perform the actual fitting of the model. The fitted model is stored in the `_fittedmod` attribute of the instance.

The `_multivariate_ols_fit` function performs the mathematical operations or procedures required to fit the multivariate OLS regression model. Unfortunately, the specific mathematical operations or procedures performed by this function are not provided in the given code snippet. Therefore, it is not possible to generate LaTeX code for displaying the equations in a markdown document.

## Class **`_MultivariateOLSResults`** Overview
The `_MultivariateOLSResults` class is a Python class that represents the results of a multivariate ordinary least squares (OLS) regression. It has the following attributes and methods:

Attributes:
- `design_info`: The design information of the fitted multivariate OLS model, which includes information about the terms used in the regression.
- `exog_names`: The names of the exogenous variables used in the regression.
- `endog_names`: The names of the endogenous variables used in the regression.
- `_fittedmod`: The fitted multivariate OLS model.

Methods:
- `__str__()`: Returns a string representation of the summary of the multivariate OLS results.
- `mv_test(hypotheses=None, skip_intercept_test=False)`: Performs a multivariate hypothesis test on the regression coefficients. The `hypotheses` parameter allows the user to specify custom hypotheses to test. If `hypotheses` is not provided, default hypotheses are generated based on the design information or the number of exogenous variables. The `skip_intercept_test` parameter determines whether to skip the hypothesis test for the intercept term. The method returns a `MultivariateTestResults` object.
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


$$
\begin{align*}
\text{self.design\_info} &= \text{fitted\_mv\_ols.data.design\_info} \\
\text{self.exog\_names} &= \text{fitted\_mv\_ols.exog\_names} \\
\text{self.endog\_names} &= \text{fitted\_mv\_ols.endog\_names} \\
\text{self.\_fittedmod} &= \text{fitted\_mv\_ols.\_fittedmod} \\
\end{align*}
$$

### Method **`__str__`** Overview
The `__str__` method in Python is a special method that is automatically called when we use the `str()` function or the `print()` function on an object. It is used to return a string representation of the object.

In the given code snippet, the `__str__` method is defined within a class and it calls another method called `summary()` to get the string representation of the object. The `summary()` method is assumed to be defined elsewhere in the class.

The purpose of the `__str__` method is to provide a human-readable string representation of the object. It is often used for debugging or displaying information about the object.

As for the mathematical operations or procedures performed in the `__str__` method, it is not possible to determine that without knowing the implementation of the `summary()` method. The `summary()` method might perform some mathematical calculations or include mathematical equations, but without further information, it is not possible to generate LaTeX code for the equations.

### Method **`mv_test`** Overview
The `mv_test` method in Python is used to perform multivariate hypothesis tests in a linear regression model. It takes two parameters: `hypotheses` and `skip_intercept_test`.

- `hypotheses` (optional): This parameter allows the user to specify the hypotheses to be tested. It is a list of lists, where each inner list contains three elements: the name of the hypothesis, the contrast matrix `L_contrast`, and the restriction matrix `R_restriction`. If no hypotheses are provided, the method will automatically generate them based on the design information of the model or the number of exogenous variables.
- `skip_intercept_test` (optional): This parameter is a boolean value that determines whether the intercept term should be included in the hypothesis tests. If set to `True`, the method will skip the hypothesis test for the intercept term.

The method first checks the number of exogenous variables (`k_xvar`) in the model. If no hypotheses are provided, it generates them based on the design information or the number of exogenous variables. For each exogenous variable, it creates a hypothesis with a contrast matrix `L` that has a single 1 in the corresponding position and zeros elsewhere.

The method then calls the `_multivariate_ols_test` function, passing the hypotheses, the fitted model (`self._fittedmod`), the names of the exogenous variables (`self.exog_names`), and the names of the endogenous variables (`self.endog_names`). This function performs the actual multivariate hypothesis tests and returns the results.

Finally, the method creates a `MultivariateTestResults` object using the results from the hypothesis tests, the names of the endogenous variables, and the names of the exogenous variables. This object is then returned by the method.

The mathematical operations or procedures performed by the method include generating contrast matrices `L_contrast` and restriction matrices `R_restriction` for each hypothesis, and calling the `_multivariate_ols_test` function to perform the hypothesis tests. The specific mathematical operations within the `_multivariate_ols_test` function are not described in the given code snippet.

### Method **`summary`** Overview
The `summary` method is a placeholder method that raises a `NotImplementedError`. It does not take any parameters.

The purpose of this method is to provide a summary of the mathematical operations or procedures performed by the Python class or object it belongs to. However, since it is not implemented, it does not perform any mathematical operations or procedures.

In terms of LaTex code, there are no mathematical equations or procedures to document for this method.

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
- `endog_names`: This parameter represents the names of the endogenous variables in the mathematical model. Endogenous variables are the variables whose values are determined by the model itself.
- `exog_names`: This parameter represents the names of the exogenous variables in the mathematical model. Exogenous variables are the variables whose values are determined outside the model.

The `__init__` method performs the following mathematical operations or procedures:

1. Assigns the value of the `results` parameter to the `self.results` attribute. This attribute will hold the results of the mathematical operation or procedure.
2. Converts the `endog_names` parameter to a list and assigns it to the `self.endog_names` attribute. This attribute will hold the names of the endogenous variables.
3. Converts the `exog_names` parameter to a list and assigns it to the `self.exog_names` attribute. This attribute will hold the names of the exogenous variables.

Here is the LaTex code to display the equations in a markdown document:

1. $self.results = results$
2. $self.endog\_names = list(endog\_names)$
3. $self.exog\_names = list(exog\_names)$

### Method **`__str__`** Overview
The `__str__` method in Python is a special method that is automatically called when we use the `str()` function or the `print()` function on an object. It is used to provide a string representation of the object.

In the given code snippet, the `__str__` method is defined within a class and it calls another method `summary()` to get a summary of the object and then converts it to a string using the `__str__` method of the summary object.

The purpose of the `__str__` method is to provide a human-readable string representation of the object. It is often used for debugging or displaying information about the object.

As for the mathematical operations or procedures performed in the `summary()` method, it is not provided in the given code snippet. Therefore, we cannot generate LaTeX code for mathematical equations without knowing the specific operations or procedures involved.

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


$$ \text{{element}} = self.results[item] $$

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

The purpose of this method is to generate a summary of a multivariate linear model. It creates an instance of the `Summary` class and adds a title to the summary indicating that it is a multivariate linear model.

For each key in the `results` attribute of the class, the method adds a dictionary with an empty key-value pair to the summary. It then creates a copy of the statistical results associated with that key and resets the index of the resulting DataFrame. The first column of the DataFrame is renamed to the current key, and the index is set to empty strings.

The method then adds the DataFrame to the summary. If the `show_contrast_L` parameter is `True`, it adds a dictionary with the key-value pair indicating that the contrast matrix `L` is included. It creates a DataFrame from the `contrast_L` attribute of the `results` dictionary and adds it to the summary.

If the `show_transform_M` parameter is `True`, it adds a dictionary with the key-value pair indicating that the transform matrix `M` is included. It creates a DataFrame from the `transform_M` attribute of the `results` dictionary and adds it to the summary.

If the `show_constant_C` parameter is `True`, it adds a dictionary with the key-value pair indicating that the constant vector `C` is included. It creates a DataFrame from the `constant_C` attribute of the `results` dictionary and adds it to the summary.

Finally, the method returns the `Summary` instance.

The mathematical operations or procedures performed by this method involve creating DataFrames from the statistical results of the multivariate linear model and adding them to the summary. These DataFrames represent the contrast matrix `L`, the transform matrix `M`, and the constant vector `C`. The method does not perform any mathematical operations itself, but rather organizes and presents the results of the model.

