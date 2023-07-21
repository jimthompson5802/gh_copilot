def _multivariate_ols_fit(endog, exog, method='svd', tolerance=1e-8):
    y = endog
    x = exog
    nobs, k_endog = y.shape
    nobs1, k_exog= x.shape
    if nobs != nobs1:
        raise ValueError('x(n=%d) and y(n=%d) should have the same number of '
                         'rows!' % (nobs1, nobs))

    # Calculate the matrices necessary for hypotheses testing
    df_resid = nobs - k_exog
    if method == 'pinv':
        # Regression coefficients matrix
        pinv_x = pinv(x)
        params = pinv_x.dot(y)

        # inverse of x'x
        inv_cov = pinv_x.dot(pinv_x.T)
        if matrix_rank(inv_cov,tol=tolerance) < k_exog:
            raise ValueError('Covariance of x singular!')

        # Sums of squares and cross-products of residuals
        # Y'Y - (X * params)'B * params
        t = x.dot(params)
        sscpr = np.subtract(y.T.dot(y), t.T.dot(t))
        return (params, df_resid, inv_cov, sscpr)
    elif method == 'svd':
        u, s, v = svd(x, 0)
        if (s > tolerance).sum() < len(s):
            raise ValueError('Covariance of x singular!')
        invs = 1. / s

        params = v.T.dot(np.diag(invs)).dot(u.T).dot(y)
        inv_cov = v.T.dot(np.diag(np.power(invs, 2))).dot(v)
        t = np.diag(s).dot(v).dot(params)
        sscpr = np.subtract(y.T.dot(y), t.T.dot(t))
        return (params, df_resid, inv_cov, sscpr)
    else:
        raise ValueError('%s is not a supported method!' % method)
