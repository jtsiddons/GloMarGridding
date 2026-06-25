import numpy as np
from scipy.stats import chi2

from glomar_gridding.autocorrelation import (
    get_auto_corr_vectorize,
    LassoEstimate_AR1,
)


def test_time_series():
    # It has been Monte Carlo/stochastic simulation
    # can be used to test anything!
    n = 99999
    phi0 = 0.6
    phi1 = 0.4
    innovation0 = 20.0
    innovation1 = 30.0
    # Opening year of NOCS and UKMO Exeter...
    seed = 19942003
    #
    # Simulate 4 temporally correlated time series
    # x0 and x1 used for all unit tests
    # x2 and x3 used for the second unit test
    rng = np.random.default_rng(seed)
    epsilon0 = rng.normal(loc=0, scale=innovation0, size=n)
    epsilon1 = rng.normal(loc=0, scale=innovation1, size=n)
    epsilon2 = rng.normal(loc=0, scale=innovation0, size=n)
    epsilon3 = rng.normal(loc=0, scale=innovation1, size=n)
    x0 = np.zeros(n + 1)
    x1 = np.zeros(n + 1)
    x2 = np.zeros(n + 1)
    x3 = np.zeros(n + 1)
    for t in range(n):
        x0[t + 1] = phi0 * x0[t] + epsilon0[t]
        x1[t + 1] = phi1 * x1[t] + epsilon1[t]
        x2[t + 1] = phi0 * x2[t] + epsilon2[t]
        x3[t + 1] = phi1 * x3[t] + epsilon3[t]
    #
    # Other info:
    # sigma(x0) = SQRT( (20 x 20) / (1 - 0.6 x 0.6) ) ~ 25.0
    # sigma(x1) = SQRT( (30 x 30) / (1 - 0.4 x 0.4) ) ~ 32.73
    # var(x0) = 625.0
    # var(x1) = 1071
    x0_variance_theory = (innovation0**2) / (1 - phi0**2)
    x1_variance_theory = (innovation1**2) / (1 - phi1**2)
    #
    x_matrix1 = np.column_stack([x0, x1])
    x_matrix2 = np.column_stack([x2, x3])
    result = get_auto_corr_vectorize(x_matrix1)
    simulated_ar_coefficients = result[0]
    simulated_variance = result[2]
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
    #
    # For the above seed, it is known to pass the below two unit tests
    #
    # Simulated AR should be within 1 standard error
    assert np.all(delta_ar < (1.0 * se_ar))
    # You have to be quite unlucky for the simulated variance to
    # sit outside the 99% two-tail test... and it doesn't for this
    # particular seed
    assert np.all(chi2_test)
    #
    # Make x2 and x3 correlated with each other
    cov_x2_x3 = x_matrix2.T @ x_matrix2 / (n - 1)
    sigma_x2_x3 = np.sqrt(np.diag(cov_x2_x3))
    mean_x0_x1 = np.mean(x_matrix2, axis=0)
    x_matrix_standardised = x_matrix2.copy()
    x_matrix_standardised[:, 0] = (
        x_matrix_standardised[:, 0] - mean_x0_x1[0]
    ) / sigma_x2_x3[0]
    x_matrix_standardised[:, 1] = (
        x_matrix_standardised[:, 1] - mean_x0_x1[1]
    ) / sigma_x2_x3[1]
    #
    # Use Cholesky to make them correlated
    adj_cov_x2_x3 = (
        np.diag(sigma_x2_x3)
        @ np.array(
            [
                [1.0, 0.7],
                [0.7, 1.0],
            ]
        )
        @ np.diag(sigma_x2_x3)
    )
    l_cov = np.linalg.cholesky(adj_cov_x2_x3)
    x_matrix_correlated = l_cov.dot(x_matrix_standardised.T)
    x_matrix_correlated = x_matrix_correlated.T
    #
    big_x = np.column_stack(
        [
            x_matrix_correlated[:, 0],
            x_matrix_correlated[:, 1],
            x_matrix1[:, 0],
            x_matrix1[:, 1],
        ]
    )
    #
    for out_of_sample_residues in [True, False]:
        lasso = LassoEstimate_AR1(
            big_x,
            lam=0.001,
            out_of_sample_residues=out_of_sample_residues,
            standardise=False,
            lasso_kws={"random_state": seed},
        )
        lasso.fit()
        #
        # Note:
        # W = cov(x(t+1), x(t)) @ inv(cov(x, x)) for OLS
        # This is not true for Lasso in general, but should
        # nearly be the case for this particular example
        #
        # For the 4 diagonal values of lasso.coefficients:
        # 2 values should be closed to phi0 (0,0 and 2,2)
        # 2 values should be closed to phi1 (1,1 and 3,3)
        # lasso.coefficients[3, 2] and lasso.coefficients[2, 3] should be
        # close to 0 because x0 and x1 are not correlated
        # However, either lasso.coefficients[0, 1] or lasso.coefficients[1, 0]
        # are not close to 0 because adjusted x2 and x3 are correlated
        #
        assert np.all(np.abs(lasso.coefficients < 1.0))
        assert np.isclose(lasso.coefficients[0, 0], phi0, atol=0.01)
        assert np.isclose(lasso.coefficients[2, 2], phi0, atol=0.01)
        assert np.isclose(lasso.coefficients[1, 1], phi1, atol=0.01)
        assert np.isclose(lasso.coefficients[3, 3], phi1, atol=0.01)
        assert np.isclose(lasso.coefficients[2, 3], 0, atol=0.01)
        assert np.isclose(lasso.coefficients[3, 2], 0, atol=0.01)
        assert np.logical_or(
            ~np.isclose(lasso.coefficients[0, 1], 0, atol=0.01),
            ~np.isclose(lasso.coefficients[1, 0], 0, atol=0.01),
        )
        #
        lasso.estimate_errcov()
        assert np.all(np.isclose(lasso.R - lasso.R.T, 0))
        # Features 2 and 3 are not correlated...
        assert np.abs(lasso.R[2, 3] / lasso.R[2, 2]) < 0.01
        assert np.abs(lasso.R[2, 3] / lasso.R[3, 3]) < 0.01
        # But features 0 and 1 are...
        # given parameters above should be 0.6-0.7ish?
        mini_errcov = lasso.R[:2, :2]
        inv_sigma = np.reciprocal(np.sqrt(np.diag(mini_errcov)))
        mini_errcor = np.diag(inv_sigma) @ mini_errcov @ np.diag(inv_sigma)
        assert np.abs(mini_errcor[0, 1] - 0.7) < 0.1
        # Standard error of the (error) covariance diagonal
        # https://stats.stackexchange.com/questions/156518/what-is-the-standard-error-of-the-sample-standard-deviation
        # SE_variance = SQRT((2 x sigma**4)/(N - 1)) for N(*) for known sigma
        nt = lasso.residues.shape[1]
        se_innovation0 = np.sqrt(2.0 * (innovation0**4) / (nt - 1))
        se_innovation1 = np.sqrt(2.0 * (innovation1**4) / (nt - 1))
        # Due to the way big_x is constructed:
        # R[0, 0], R[2, 2] should have be close to innovation0**2
        # R[3, 3] should have be close to innovation1**2
        # R[1, 1] is different... it gets adjusted by Cholesky to mimic the
        # the correlation between feature 0,1 while keeping feature 0 unchanged
        assert np.abs(lasso.R[0, 0] - innovation0**2) < (se_innovation0 * 3)
        assert np.abs(lasso.R[2, 2] - innovation0**2) < (se_innovation0 * 3)
        assert np.abs(lasso.R[3, 3] - innovation1**2) < (se_innovation1 * 3)
    #
    # Test weight expansion
    D = np.array(
        [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ],
        dtype=np.uint8,
    )
    original_2_2 = lasso.coefficients[2, 2]
    lasso.expand_coefficients(D, fill_value=999)
    assert lasso.coefficients[2, 2] == 999
    assert lasso.expanded
    fake_lats = [-2, -1, 0, 1, 2]
    fake_lons = [180]
    da = lasso.to_xarray_da_coefficients(fake_lats, fake_lons)
    assert np.all(da.values == lasso.coefficients.toarray())
    lasso.unexpand_coefficients()
    assert lasso.coefficients[2, 2] == original_2_2
    assert not lasso.expanded
