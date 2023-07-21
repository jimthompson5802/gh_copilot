# Module:`multivariate_ols.py` Overview

The code is a Python implementation of multivariate linear regression using ordinary least squares (OLS). It includes functions for fitting the regression model, performing hypothesis tests, and generating summary statistics.

The main function in the code is `_multivariate_ols_fit`, which fits the multivariate OLS model to the given data. It takes the endogenous variables (`endog`) and exogenous variables (`exog`) as input and returns the estimated regression coefficients, degrees of freedom for residuals, inverse of the covariance matrix, and sums of squares and cross-products of residuals.

The code also includes a function `multivariate_stats` that calculates various statistics for hypothesis testing, such as Wilks' lambda, Pillai's trace, Hotelling-Lawley trace, and Roy's greatest root. These statistics are used to test the null hypothesis that the coefficients of the regression model are zero.

The function `_multivariate_ols_test` performs hypothesis tests based on the given hypotheses. It takes the contrast matrix (`L`), transform matrix (`M`), and constant matrix (`C`) as input and calculates the test statistics and p-values.

The code also includes a class `_MultivariateOLS` that represents the multivariate OLS model. It has a `fit` method that fits the model using the `_multivariate_ols_fit` function and returns an instance of `_MultivariateOLSResults` class.

The `_MultivariateOLSResults` class represents the results of the multivariate OLS model. It has a `mv_test` method that performs hypothesis tests based on the fitted model and returns an instance of `MultivariateTestResults` class.

The `MultivariateTestResults` class represents the results of the hypothesis tests. It has a `summary` method that generates a summary of the test results, including the test statistics and p-values.

Overall, the code provides a comprehensive implementation of multivariate linear regression using OLS and includes functions for fitting the model, performing hypothesis tests, and generating summary statistics.

## Function **`_multivariate_ols_fit`** Overview
The function `_multivariate_ols_fit` is used to perform multivariate ordinary least squares (OLS) regression. It takes in two parameters: `endog` (the dependent variable) and `exog` (the independent variables). 

The function first checks if the number of observations in `endog` and `exog` are the same. If not, it raises a `ValueError`. 

Next, it calculates the necessary matrices for hypothesis testing. If the `method` parameter is set to 'pinv', it uses the Moore-Penrose pseudoinverse to calculate the regression coefficients matrix (`params`) and the inverse of the matrix product of `exog` and its transpose (`inv_cov`). It also calculates the sums of squares and cross-products of residuals (`sscpr`). 

If the `method` parameter is set to 'svd', it uses the singular value decomposition (SVD) to calculate the regression coefficients matrix (`params`), the inverse of the matrix product of `exog` and its transpose (`inv_cov`), and the sums of squares and cross-products of residuals (`sscpr`). 

If the `method` parameter is neither 'pinv' nor 'svd', it raises a `ValueError`. 

The function returns a tuple containing the regression coefficients (`params`), the degrees of freedom of the residuals (`df_resid`), the inverse covariance matrix (`inv_cov`), and the sums of squares and cross-products of residuals (`sscpr`).

### **Function Details**
This code defines a function called `_multivariate_ols_fit` that performs multivariate ordinary least squares (OLS) regression. 

The function takes three arguments: `endog`, `exog`, and `method`. 
- `endog` is a numpy array representing the dependent variable(s) in the regression. 
- `exog` is a numpy array representing the independent variable(s) in the regression. 
- `method` is a string specifying the method to use for calculating the regression coefficients and other statistics. The default method is 'svd', but 'pinv' is also supported. 

The function first checks if the number of observations in `endog` and `exog` are the same. If not, it raises a ValueError. 

Next, it calculates the degrees of freedom for the residuals. 

If the method is 'pinv', it calculates the regression coefficients using the Moore-Penrose pseudoinverse of `exog` and the formula `params = pinv_x.dot(y)`. It also calculates the inverse of `x'x` (where `x'` is the transpose of `x`) using the pseudoinverse and checks if it is singular. Finally, it calculates the sums of squares and cross-products of residuals using the formula `sscpr = np.subtract(y.T.dot(y), t.T.dot(t))`, where `t = x.dot(params)`.

If the method is 'svd', it performs singular value decomposition (SVD) on `exog` using the `svd` function. It checks if any of the singular values are below a tolerance value and raises a ValueError if so. It then calculates the regression coefficients using the SVD components and the formula `params = v.T.dot(np.diag(invs)).dot(u.T).dot(y)`. It also calculates the inverse of `x'x` using the SVD components and checks if it is singular. Finally, it calculates the sums of squares and cross-products of residuals using the formula `sscpr = np.subtract(y.T.dot(y), t.T.dot(t))`, where `t = np.diag(s).dot(v).dot(params)`.

The function returns a tuple containing the regression coefficients (`params`), the degrees of freedom for the residuals (`df_resid`), the inverse of `x'x` (`inv_cov`), and the sums of squares and cross-products of residuals (`sscpr`).

## Function **`multivariate_stats`** Overview
The function `multivariate_stats` takes in several parameters: `eigenvals`, `r_err_sscp`, `r_contrast`, `df_resid`, and an optional parameter `tolerance`. It performs calculations and returns a DataFrame containing various statistical values.

The function calculates different multivariate statistical measures based on the given input parameters. It computes the values for "Wilks' lambda", "Pillai's trace", "Hotelling-Lawley trace", and "Roy's greatest root". These measures are commonly used in multivariate analysis to assess the significance of differences between groups or conditions.

The function uses the input parameters to perform calculations and populate the DataFrame with the computed values. It utilizes numpy and pandas libraries for mathematical operations and data manipulation. The function also uses the `stats` module from the scipy library to calculate p-values using the F-distribution.

The resulting DataFrame contains the computed statistical values for each measure, including the value itself, the degrees of freedom, the F-value, and the p-value. The DataFrame is then returned as the output of the function.

### **Function Details**
This code defines a function called `multivariate_stats` that calculates various statistics for multivariate analysis. The function takes the following parameters:

- `eigenvals`: A numpy array of eigenvalues.
- `r_err_sscp`: A scalar representing the residual sum of squares and cross products.
- `r_contrast`: A scalar representing the contrast sum of squares and cross products.
- `df_resid`: A scalar representing the degrees of freedom for the residuals.
- `tolerance`: A scalar representing the tolerance value for determining which eigenvalues to include in the calculations.

The function first assigns the input parameters to local variables `v`, `p`, `q`, and `s`. It then calculates the number of eigenvalues greater than the tolerance value and assigns it to `n_e`. It also calculates two arrays `eigv2` and `eigv1` based on the eigenvalues.

The function then creates an empty DataFrame called `results` with specific columns and index values.

Next, the function defines a nested function called `fn` that takes a scalar value `x` and returns its real part.

The function then calculates the values for each statistic and assigns them to the corresponding cells in the `results` DataFrame.

Finally, the function calculates the F-values and p-values for each statistic using the `stats.f.sf` function from the `scipy` library, and assigns them to the corresponding cells in the `results` DataFrame.

The function returns the `results` DataFrame.

## Function **`_multivariate_ols_test`** Overview
The function `_multivariate_ols_test` is a helper function that is used to perform a multivariate ordinary least squares (OLS) test. 

The function takes four arguments: `hypotheses`, `fit_results`, `exog_names`, and `endog_names`. 

- `hypotheses` is a list of hypothesis matrices that define the null hypotheses to be tested. 
- `fit_results` is a tuple containing the results of the OLS regression, including the estimated parameters, degrees of freedom of the residuals, inverse covariance matrix, and sum of squared centered predicted residuals. 
- `exog_names` is a list of names for the exogenous variables in the regression. 
- `endog_names` is a list of names for the endogenous variables in the regression. 

The function defines an inner function `fn` that takes three arguments: `L`, `M`, and `C`. 

- `L` is a matrix that defines the linear combination of the estimated parameters to be tested. 
- `M` is a matrix that defines the linear combination of the observed variables to be tested. 
- `C` is a matrix that defines the constant term in the linear combination. 

Inside the `fn` function, the function calculates the test statistic for the null hypothesis using the provided matrices and the fit results from the OLS regression. 

- The first step is to calculate `t1`, which is the linear combination of the estimated parameters (`params`) defined by `L` and `M`, subtracted by `C`. 
- Then, `t2` is calculated as the product of `L`, the inverse covariance matrix (`inv_cov`), and the transpose of `L`. 
- The rank of `t2` is calculated and stored in `q`. 
- Finally, the test statistic `H` is calculated as the product of `t1` transposed, the inverse of `t2`, and `t1`. 

The function also calculates the error term `E` using the observed variables (`M`), the sum of squared centered predicted residuals (`sscpr`), and the transpose of `M`. 

The function returns the error term `E`, the test statistic `H`, the rank `q`, and the degrees of freedom of the residuals `df_resid`. 

The function then calls another function `_multivariate_test` with the calculated values and returns the result.

### **Function Details**
The given code defines a function `_multivariate_ols_test` that performs a multivariate test for ordinary least squares (OLS) regression models. 

The function takes four arguments:
- `hypotheses`: A list of hypothesis matrices.
- `fit_results`: A tuple containing the results of the OLS regression model fit. It includes the estimated parameters, degrees of freedom for residuals, inverse covariance matrix, and sum of squared centered predicted residuals.
- `exog_names`: A list of names for the exogenous variables.
- `endog_names`: A list of names for the endogenous variables.

The function defines an inner function `fn` that takes three arguments: `L`, `M`, and `C`. This inner function calculates the test statistics for the given hypotheses.

The function then calls another function `_multivariate_test` with the hypotheses, exogenous and endogenous variable names, and the `fn` function as arguments. The `_multivariate_test` function is not defined in the given code.

Overall, the code appears to be a part of a larger codebase that performs multivariate tests for OLS regression models.

## Function **`_multivariate_test`** Overview
The function `_multivariate_test` takes in four parameters: `hypotheses`, `exog_names`, `endog_names`, and `fn`. 

The `hypotheses` parameter is a list of tuples, where each tuple represents a hypothesis. The length of each tuple can be 2, 3, or 4. The first element of the tuple is the name of the hypothesis, the second element is the contrast matrix `L`, the third element is the transform matrix `M`, and the fourth element is the constant matrix `C`. The function checks the length of each tuple and assigns the values accordingly.

The `exog_names` parameter is a list of names for the exogenous variables, and the `endog_names` parameter is a list of names for the endogenous variables.

The `fn` parameter is a function that takes in the contrast matrix `L`, transform matrix `M`, and constant matrix `C` as arguments and returns four values: `E`, `H`, `q`, and `df_resid`.

The function then iterates over each hypothesis in the `hypotheses` list. It checks the type of each element in `L`, `M`, and `C` and performs necessary validations. It calculates the values of `E`, `H`, `q`, and `df_resid` by calling the `fn` function. It calculates the rank of the sum of `E` and `H` and sorts the eigenvalues of the inverse of `E + H` multiplied by `H`. It then calls the `multivariate_stats` function to calculate the statistical values based on the eigenvalues, rank, `q`, and `df_resid`. Finally, it stores the results in a dictionary with the hypothesis name as the key and the calculated values as the value.

The function returns the dictionary of results.

### **Function Details**
This code defines a function called `_multivariate_test` that takes in four arguments: `hypotheses`, `exog_names`, `endog_names`, and `fn`. 

The `hypotheses` argument is a list of tuples, where each tuple represents a hypothesis. Each hypothesis tuple can have 2, 3, or 4 elements. The first element is the name of the hypothesis, the second element is the contrast matrix `L`, the third element is the transform matrix `M`, and the fourth element is the constant matrix `C`. 

The `exog_names` argument is a list of names for the exogenous variables, and the `endog_names` argument is a list of names for the endogenous variables. 

The `fn` argument is a function that takes in the contrast matrix `L`, transform matrix `M`, and constant matrix `C`, and returns four values: `E`, `H`, `q`, and `df_resid`. 

Inside the function, the number of exogenous variables `k_xvar` and the number of endogenous variables `k_yvar` are calculated based on the lengths of `exog_names` and `endog_names`, respectively. 

A dictionary called `results` is created to store the results for each hypothesis. 

The function then iterates over each hypothesis in the `hypotheses` list. If the length of the hypothesis tuple is 2, 3, or 4, the name, contrast matrix `L`, transform matrix `M`, and constant matrix `C` are assigned accordingly. If any of the elements in `L` or `M` are strings, they are converted to linear constraints using the `DesignInfo` class. 

Next, the function checks the dimensions of `L`, `M`, and `C` to ensure they are valid. 

The function then calls the `fn` function with `L`, `M`, and `C` as arguments to calculate the values of `E`, `H`, `q`, and `df_resid`. 

The sum of `E` and `H` is calculated and stored in `EH`. The rank of `EH` is calculated and stored in `p`. 

The eigenvalues of the inverse of `EH` multiplied by `H` are calculated and sorted in ascending order. 

A function called `multivariate_stats` is called with the eigenvalues, `p`, `q`, and `df_resid` as arguments to calculate a statistic table. 

The results for the current hypothesis are stored in the `results` dictionary with the name of the hypothesis as the key. 

Finally, the `results` dictionary is returned.

