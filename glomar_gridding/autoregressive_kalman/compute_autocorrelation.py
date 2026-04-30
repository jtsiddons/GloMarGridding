"""
Docstring for lagged_correlation

Estimation of lag-1 autocorrelation and other metrics
for a single time series

Uses numpy.vectorize to extend the computation to multiple time series,
but each time series are treated seperately (i.e. no correlation
between time series). Function applied along each row.

get_anomalies: n -> n:
vectorize for an array with m rows n columns gives m rows and n columns

get_obs_variance etc: n -> scalar:
vectorize for an array with m rows give m scalars.

The same behavior can be gotten using np.apply_along_axis
"""
import numpy as np


def get_anomalies(vec: np.ndarray):
    """
    Compute anomalies, 0-mean centering data samples
    Samples are NOT standardized by their standard deviation

    Parameters
    ----------
    vec: numpy.ndarray
        Array that needs be 0-meaned

    Returns
    -------
    anomalies: numpy.ndarray
        0-meaned values
    """
    sample_mean = np.nanmean(vec)
    anomalies = vec - sample_mean
    return anomalies


# get_anomalies, operated over many time series
# for a matrix with m rows and n colums
# gives an answer of m rows and n columns;
# anomalies computed along each row
get_anomalies_vectorize = np.vectorize(
    get_anomalies,
    doc='Vectorized `get_anomalies`',
    signature='(n)->(n)',
)


def get_obs_variance(
        vec: np.ndarray,
        dof: int = 1
    ):
    """
    Compute observed variance (standard deviation squared)
    Bessel correction applied by default

    Parameters
    ----------
    vec: numpy.ndarray
        1D vector that needs observed variance
    dof: int
        Degrees of freedom correction, default 1 (Bessel's correction)

    Returns
    -------
    variance: float
        standard deviation squared
    """
    n = np.sum(~np.isnan(vec))  # Non-nan sample size
    squared_anomalies = np.square(get_anomalies(vec))
    variance = np.nansum(squared_anomalies) / (n - dof)
    return variance

# obs_variance, operated over many time series
# for a matrix with m rows and n colums
# gives an answer of m scalars


obs_variance_vectorize = np.vectorize(
    lambda vec: get_obs_variance(vec, dof=1),
    doc='Vectorized `get_obs_variance`',
    signature='(n),()->()',
)


def get_auto_corr(
        vec: np.ndarray,
        n: int = 1,
    ):
    """
    Compute lag-n autocorrelation

    This is the "weight" for spatially uncorrelated autoregressive model

    Degrees of freedom:
    - This should be n minus lag minus 1
    - for lag-1, there are n - 1 samples, with an additional -1
    from Bessel's correction

    Parameters
    ----------
    vec: numpy.ndarray
        1D vector that needs autoregression correlation
    n: int
        order (lag) of AR

    Returns
    -------
    ar_n : float
        the n-th lagged correlation
        this is autocovariance_n divided by (non-lagged) variance
    autocovariance_n: float
        Same as in ar_n but not standardized
    variance: float
        standard deviation squared
    sigma2_eps: float
        Standard error of the autoregressive model associated with
        the same autocorrelation
    """
    anomalies = get_anomalies(vec)
    print(f"Number of valid values = {np.sum(~np.isnan(anomalies))}")
    # See Wilks book Time Domain II chapter Eq 8.22
    # With reasonably large sample size 30 year of daily data,
    # the dof correction is neglieble
    # For lagged correlation, the degrees of freedom is
    # Sample_Size - Lag (aka n) - 1 (Bessel Correction)
    ddof_correction = n + 1
    #
    # Observed variance
    # Sorts out the dof correction here: n minus lag minus 1
    variance = get_obs_variance(vec, dof=ddof_correction)
    cov = np.ma.cov(
        np.ma.masked_invalid(anomalies[:-n]),
        np.ma.masked_invalid(anomalies[n:]),
        ddof=0,   # Or you can sort out the dof here...
    )
    var_t_eq_0 = cov[0, 0]  # variance of ... :-n
    var_t_plus_n = cov[1, 1]  # Variance of n: ...
    autocovariance_n = cov[1, 0]  # Observed autocovariance
    ar_n = autocovariance_n / np.sqrt(var_t_eq_0 * var_t_plus_n)  # autocorr
    sigma2_eps = (1 - ar_n ** 2) * variance  # Uncertainity**2 of the AR model
    return ar_n, autocovariance_n, variance, sigma2_eps

# get_auto_corr_1, operated over many time series
# behaves similarily to obs_variance_vectorize but give
# more scalars (aka for a m-row n-column array, it gives
# 4 different m-lengthed vectors


get_auto_corr_1_vectorize = np.vectorize(
    lambda vec: get_auto_corr(vec, 1),
    doc='Vectorized `get_auto_corr`',
    signature='(n)->(), (), (), ()',
)
