"""Functions for helping with perturbations/random drawing"""

import numpy as np
from scipy import stats
import logging


def scipy_mv_normal_draw(  # noqa: C901
    loc: np.ndarray,
    cov: np.ndarray,
    ndraws: int = 1,
    eigen_rtol: float = 1e-6,
    eigen_fudge: float = 1e-8,
) -> np.ndarray:
    """
    Do a random multivariate normal draw using
    scipy.stats.multivariate_normal.rvs

    numpy.random.multivariate_normal can also,
    but fixing seeds are more difficult using numpy

    This function has similar API as GP_draw with less kwargs.

    Warning/possible future scipy version may change this:
    It seems if one uses stats.Covariance, you have to have add [0] from rvs
    function. The above behavior applies to scipy v1.14.0

    Parameters
    ----------
    loc : float
        the location for the normal dry
    cov : numpy.ndarray
        not a xarray/iris cube! Some of our covariances are saved in numpy
        format and not netCDF files
    n_draws : int
        number of simulations, this is usually set to 1 except during
    unit testing
    eigen_rtol : float
        relative tolerance to negative eigenvalues
    eigen_fudge : float
        forced minimum value of eigenvalues if negative values are detected

    Returns
    -------
    draw : np.ndarray
        The draw(s) from the multivariate random normal distribution defined
        by the loc and cov parameters. If the cov parameter is not
        positive-definite then a new covariance will be determined by adjusting
        the eigen decomposition such that the modified covariance should be
        positive-definite.
    """

    def any_complex(arr: np.ndarray) -> bool:
        return bool(np.any(np.iscomplex(arr)))

    cov_shape = cov.shape
    if len(cov_shape) != 2:
        raise ValueError("cov should be 2D.")
    if cov_shape[0] != cov_shape[1]:
        raise ValueError("cov is not a square matrix")
    try:
        draw = np.random.multivariate_normal(loc, cov, size=ndraws)
        return draw[0] if ndraws == 1 else draw
    except np.linalg.LinAlgError as e:
        pass
    except Exception as e:
        raise e

    # Try to use eigen decomposition to generate a new covariance matrix that
    # would be positive-definite
    w, v = np.linalg.eigh(cov)
    w = np.real_if_close(w)
    v = np.real_if_close(v)
    if any_complex(w):
        raise ValueError("w is complex")
    if any_complex(v):
        raise ValueError("v is complex")
    if np.any(w < 0):
        raise Exception("STOP")
        most_neg_eigval = np.min(w)
        largest_eig_val = np.max(w)
        rtol_check = np.abs(most_neg_eigval) / largest_eig_val
        logging.warning(
            "Negative eigenvalues detected: largest = "
            + f"{largest_eig_val}; smallest = {most_neg_eigval}; "
            + f"ratio = {rtol_check}"
        )
        if rtol_check >= eigen_rtol:
            raise ValueError("Negative eigenvalues are unexpectedly large.")
        w[w < eigen_fudge] = eigen_fudge

    cov2 = stats.Covariance.from_eigendecomposition((w, v))

    # WARN: Weird/inconsistent behavior warning
    # if size==1 and cov is an instance of stats.Covariance
    # ans has shape of (1, len(loc2),)
    # this behavior is consistent with size > 1 which yields (size, len(loc2))
    # but is INCONSISTENT with behavior when cov is a
    # valid numpy array ---> shape is (len(loc2),)

    draw = stats.multivariate_normal.rvs(mean=loc, cov=cov2, size=ndraws)
    # draw = np.random.multivariate_normal(loc, cov2.covariance, size=ndraws)
    return draw[0] if ndraws == 1 else draw
