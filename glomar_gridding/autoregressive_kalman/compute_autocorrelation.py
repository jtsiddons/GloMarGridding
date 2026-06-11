"""
Compute lagged (lag-1) correlations

Estimation of lag-1 autocorrelation and other metrics
for a single time series

Outputs can be organised to be used `Autoregressive1ForecastUncorr`
in `forecast_linear_ar1`. It is not the most sophisticated and fancy
computation (in a matter of fact very elementary).

Uses numpy.vectorize to extend the computation to multiple time series,
but each time series are treated separately (i.e. no correlation
between time series). Function applied along each row.

get_anomalies: n -> n:
vectorize for an array with m rows n columns gives m rows and n columns

get_obs_variance etc: n -> scalar:
vectorize for an array with m rows give m scalars.

The same behavior can be gotten using np.apply_along_axis
"""

import logging
import numpy as np
import xarray as xr
import warnings

from functools import partial


def get_anomalies(vec: np.ndarray) -> np.ndarray:
    """
    Compute anomalies, 0-mean centering data samples.

    Samples are NOT standardized by their standard deviation.

    Parameters
    ----------
    vec: numpy.ndarray
        Array that needs be 0-averaged

    Returns
    -------
    anomalies: numpy.ndarray
        0-meaned values
    """
    sample_mean = np.nanmean(vec)
    anomalies = vec - sample_mean
    return anomalies


def get_anomalies_vectorize(arr: np.ndarray):
    """
    Vectorised `get_anomalies`

    Similar precautions with `get_obs_variance_vectorize`:
    this assumes the convention used in climate data,
    `scikit-learn`, multivariate random num generator etc.
    (transpose arr automatically so it works properly with `np.vectorize`).
    See `get_obs_variance_vectorize` docstring

    Parameters
    ----------
    arr: numpy.ndarray
        Array that needs be 0-averaged for each ROW;
        shape `(N, M)` (N samples, M features/independent variable)

    Returns
    -------
    anomalies: numpy.ndarray
        0-averaged values along each ROW;
        shape `(N, M)` (N samples for M features)
    """
    return np.vectorize(get_anomalies, signature="(n)->(n)")(arr.T)


def get_obs_variance(
    vec: np.ndarray,
    ddof: int = 1,
    compute_anomalies=True,
) -> float:
    """
    Compute observed variance (standard deviation squared).
    Bessel correction applied by default.

    Parameters
    ----------
    vec: numpy.ndarray
        1D vector that needs observed variance
    ddof: int
        Degrees of freedom correction, default 1 (Bessel's correction);
        this is the similar to kwargs used in `numpy.cov` (`bias` and `ddof`)
    compute_anomalies: bool
        If `True`, the mean will be removed from vec

    Returns
    -------
    variance: float
        standard deviation squared
    """
    n = np.sum(~np.isnan(vec))  # Non-nan sample size
    if compute_anomalies:
        samples = get_anomalies(vec)
    else:
        samples = vec
    squared_anomalies = np.square(samples)
    variance = np.nansum(squared_anomalies) / (n - ddof)
    return variance


def get_obs_variance_vectorize(
    arr: np.ndarray,
    kwargs_for_get_obs_variance: dict | None = None,
) -> np.ndarray:
    """
    Running get_obs_variance for a multi-dimension array

    Warning: naive `np.vectorize` may result operations on WRONG axis.

    Nearly all climate data, functions like `scipy.stats.multivariate_norm`
    (aka scikit-learn convention) will need to be TRANSPOSED for the function
    to work as normally intended, which is automatically performed here.
    (aka (N, M) -- N samples, M features/independent variable); this function
    (and its companion ones) ASSUMES this will be the shape of `arr`.

    Try this function with 1D vector:

    `vec.reshape(-1, 1)`
    (scikit-learn convention for single feature)
    gives (sample_size, 1) and WORKS

    `vec.reshape(1, -1)`
    (scikit-learn convention for single sample)
    gives (1, sample_size) and DOES NOT WORK

    Parameters
    ----------
    arr: numpy.ndarray
        A multidimensional array that needs a vector of variance for
        each COLUMN;
        shape `(N, M)` (N samples, M features/independent variable)
    kwargs_for_get_obs_variance: dict | None
        `kwargs` to be added to get_obs_variance (i.e. `compute_anomalies` and `ddof`)

    Returns
    -------
    variance: numpy.ndarray
        standard deviation squared;
        shape (M,) (M features)
    """
    if kwargs_for_get_obs_variance is None:
        kwargs_for_get_obs_variance = {}
    operator = partial(get_obs_variance, **kwargs_for_get_obs_variance)
    return np.vectorize(operator, signature="(n)->()")(arr.T)


def get_auto_corr(
    vec: np.ndarray,
    n: int = 1,
) -> tuple[float, float, float, float]:
    """
    Compute lag-n autocorrelation

    This is the "weight" for spatially uncorrelated autoregressive model.

    Degrees of freedom:
    - This should be n minus lag minus 1
    - For lag-1, there are n - 1 samples
    with an additional -1 from Bessel's correction

    Parameters
    ----------
    vec: numpy.ndarray
        1D vector that needs autoregression correlation
    n: int
        order (lag) of AR

    Returns
    -------
    ar_n : float
        The n-th lagged correlation; this is
        autocovariance_n divided by (non-lagged) variance
    autocovariance_n: float
        Same as in `ar_n` but not normalised
    variance: float
        Standard deviation squared
    sigma2_eps: float
        Standard error of the autoregressive model associated with
        the same autocorrelation
    """
    anomalies = get_anomalies(vec)
    logging.debug(f"Number of valid values = {np.sum(~np.isnan(anomalies))}")
    # See Wilks book Time Domain II chapter Eq 8.22
    # With reasonably large sample size 30 year of daily data,
    # the dof correction is negligible
    #
    # Compute observed variance with dof correction
    # Sample size: len(vec)
    # Degree of freedom: len(vec) - 1
    variance = get_obs_variance(anomalies, compute_anomalies=False)
    #
    # Compute observed autocovariance and autocorrelation
    # Sample size: len(vec) - lag
    # Degree of freedom: len(vec) - lag - 1
    cov = np.ma.cov(
        np.ma.masked_invalid(anomalies[:-n]),
        np.ma.masked_invalid(anomalies[n:]),
    )
    var_t_eq_0 = cov[0, 0]  # Observed variance of [... :-n]
    var_t_plus_n = cov[1, 1]  # Observed variance of [n: ...]
    autocovariance_n = cov[1, 0]  # Observed autocovariance
    ar_n = autocovariance_n / np.sqrt(
        var_t_eq_0 * var_t_plus_n
    )  # Normalises autocovariance to autocorrelation
    # Note:
    # Due to finite sample size, var_t_eq_0 != var_t_plus_n != variance
    # even if they are asymptotically the same.
    # For reasonably large len(vec), they are undistinguishable.
    sigma2_eps = (1 - ar_n**2) * variance  # Uncertainty**2 of the AR model
    return ar_n, autocovariance_n, variance, sigma2_eps


def get_auto_corr_1_vectorize(
    arr: np.ndarray,
    n: int = 1,
):
    """
    Vectorised get_auto_corr

    Defaults to lag-1 via `kwargs_for_get_auto_corr`

    Similar precautions with `get_obs_variance_vectorize`:
    this assumes the convention used in climate data,
    scikit-learn, multivariate random num generator etc.
    (transpose arr automatically so it works properly with np.vectorize);
    see `get_obs_variance_vectorize` docstring.

    Parameters
    ----------
    arr: numpy.ndarray
        A multi-dimension array that needs autoregression correlation;
        shape `(N, M)` (N samples, M features/independent variable)
    n: int
        order (lag) of AR

    Returns
    -------
    ar_n : float
        The n-th lagged correlation; this is
        autocovariance_n divided by (non-lagged) variance;
        shape `(M,)` (M features)
    autocovariance_n: float
        Same as in ar_n but not standardised;
        shape `(M,)` (M features)
    variance: float
        Standard deviation squared;
        shape `(M,)` (M features)
    sigma2_eps: float
        Standard error of the autoregressive model associated with
        the same autocorrelation;
        shape `(M,)` (M features)
    """
    operator = partial(get_auto_corr, n=n)
    return np.vectorize(operator, signature="(n)->(), (), (), ()")(arr.T)


def compute_lag_1_ts_metrics_from_da(
    da: xr.DataArray,
    lag: int = 1,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Compute the following (local) time series metrics
    - nth-lagged autocorrelation
    - sample variance

    For n > 1, the autoregressive regression coefficient
    depends on lagged correlation for shorter lags.
    See Yule-Walker equations for details.

    This computation ignores spatial correlation. Time series metrics
    are local.

    Parameters
    ----------
    da: xarray.DataArray
        A 2+-dimension data array that presumes time is on axis-0;
        by climate data conventions, usually only ensemble number/realization
        should have axis number lower than time.
        e.g. Ens, time, z, y, x ... etc.
        Reorder your dimensions first!
    lag: int
        The lagged autocorrelation needed;
        if lag > 1, then your autoregressive model coefficients are
        different, see Yule-Walker equations for details.

    Returns
    -------
    da1: xarray.DataArray
        Data array with lagged autoregression
    da2: xarray.DataArray
        Data array with sample variance
    """
    if da.dims[0] != "time":
        raise ValueError(f"Unexpected dim order: {da.dims}")
    da_varname = da.name
    da_long_varname = getattr(da, "long_name", da_varname)
    if hasattr(da, "units"):
        units = da.units
    else:
        warnings.warn(
            "ds has no units, setting it to 1",
            UserWarning,
        )
        units = "1"
    #
    ar = da[0].copy()
    ar = ar.rename(f"{da_varname}_ar{lag}")
    ar.attrs["units"] = "1"
    ar.attrs["long_name"] = f"{da_long_varname} lag-{lag} correlation"
    logging.debug(f"{ar.shape = }")
    #
    variance = da[0].copy()
    variance = variance.rename(f"{da_varname}_variance")
    variance.attrs["units"] = f"{units}**2"
    variance.attrs["long_name"] = f"{da_long_varname} variance"
    logging.debug(f"{variance.shape = }")
    #
    ar_arr, _, variance_arr, _ = np.apply_along_axis(
        get_auto_corr,
        0,
        da.values,
    )
    #
    variance.values = variance_arr
    ar.values = ar_arr
    #
    return (
        ar,
        variance,
    )
