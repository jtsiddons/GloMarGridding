import numpy as np
from scipy.stats import chi2

from glomar_gridding.autocorrelation import (
    get_auto_corr_1_vectorize,
    LassoEstimate_AR1,
)


def test_time_series():
    n = 99999
    phi0 = 0.6
    phi1 = 0.4
    innovation0 = 20.0
    innovation1 = 30.0
    # (Opening day of NOCS and UKMO Exeter...)
    seed = 19942003
    rng = np.random.default_rng(seed)
    epsilon0 = rng.normal(loc=0, scale=innovation0, size=n)
    epsilon1 = rng.normal(loc=0, scale=innovation1, size=n)
    x0 = np.zeros(n + 1)
    x1 = np.zeros(n + 1)
    for t in range(n):
        x0[t + 1] = phi0 * x0[t] + epsilon0[t]
        x1[t + 1] = phi1 * x1[t] + epsilon1[t]
    #
    # Other info:
    # sigma(x0) = SQRT( (20 x 20) / (1 - 0.6 x 0.6) ) ~ 25.0
    # sigma(x1) = SQRT( (30 x 30) / (1 - 0.4 x 0.4) ) ~ 32.73
    # var(x0) = 625.0
    # var(x1) = 1071
    x0_variance_theory = (innovation0**2) / (1 - phi0**2)
    x1_variance_theory = (innovation1**2) / (1 - phi1**2)
    #
    # Shape should (100000, 2)
    x_matrix = np.column_stack([x0, x1])
    print(x_matrix.shape)
    ans = get_auto_corr_1_vectorize(x_matrix)
    simulated_ar_coefficients = ans[0]
    # simulated_autocovariance = ans[1]
    simulated_variance = ans[2]
    #
    # Standard error of correlation coefficient is SQRT( (1 - r x r) / (n - 2) )
    # https://stats.stackexchange.com/questions/226380/derivation-of-the-standard-error-for-pearsons-correlation-coefficient
    ar_vec = np.array([phi0, phi1])
    se_ar = np.sqrt((1 - np.square(ar_vec)) / (n + 1 - 2))
    delta_ar = np.abs(simulated_ar_coefficients - ar_vec)
    #
    theory_variance_vec = np.array(
        [
            x0_variance_theory,
            x1_variance_theory,
        ]
    )
    alpha = 0.005
    chi2_test_stat = (n + 1 - 1) * simulated_variance / theory_variance_vec
    chi2_test_stat_lower_bound = chi2.ppf(alpha / 2, n + 1 - 1)
    chi2_test_stat_upper_bound = chi2.ppf(1 - (alpha / 2), n + 1 - 1)
    chi2_test = np.logical_and(
        chi2_test_stat_upper_bound > chi2_test_stat,
        chi2_test_stat > chi2_test_stat_lower_bound,
    )
    print(chi2_test)
    #
    # For the above seed: it is known to pass the below checks
    #
    # Simulated AR should be within 1 standard error
    assert np.all(delta_ar < (1.0 * se_ar))
    # You have to be quite unlucky for the simulated variance to
    # sit outside the 99% two tail test...
    assert np.all(chi2_test)
    #
    # Let make 2 time series correlated...
    corr_x0_x1 = np.array(
        [
            [1.0, 0.7],
            [0.7, 1.0],
        ]
    )
    l_cov = np.linalg.cholesky(corr_x0_x1)
    x_matrix_correlated = l_cov.dot(x_matrix.T).T
    big_x = np.column_stack(
        [
            x_matrix_correlated[:, 0],
            x_matrix_correlated[:, 1],
            x_matrix[:, 0],
            x_matrix[:, 1],
        ]
    )
    print(big_x)
    print(big_x.shape)
    lasso = LassoEstimate_AR1(
        big_x,
        lasso_lambda_hyperparm=0.001,
        out_of_sample_residues=False,
        standardise=True,
        lasso_kws={"random_state": seed},
    )
    lasso.fit()
    #
    # Note: W = cov(x(t+1), x(t)) @ inv(cov(x, x))
    #
    assert np.all(np.abs(lasso.coefficients < 1.0))
    assert np.any(np.isclose(lasso.coefficients, phi0, atol=0.05))
