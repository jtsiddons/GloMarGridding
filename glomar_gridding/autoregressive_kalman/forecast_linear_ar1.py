"""
Conduct local AR1 forecast of uncertain data
output new uncertainties
"""

import numpy as np


class Autoregressive1Forecast:
    """
    Class to compute AR1 forecast

    Compute Lag-1 autoregressive forecast as a prior for Kalman filter.
    """

    def __init__(
        self,
        independent_var_t: np.ndarray,
        errcov_independent_var_t: np.ndarray,
        lag_1_autocor: np.ndarray,
        climatology_mean: np.ndarray,
        climatology_variance: np.ndarray,
        climatology_variance_is_sdev: bool = False,
    ):
        """
        __init__ for Autoregressive1Forecast class

        Parameters
        ----------
        independent_var_t: numpy.ndarray
            1D vector of data for t=t
        errcov_independent_var_t: numpy.ndarray
            2D error covariance for independent_var_t
        lag_1_autocor_for_t_plus_1: numpy.ndarray
            1D vector of lag-1 autocorrelation
        climatology_mean: numpy.ndarray
            Climatological mean
            Shape should be same as independent_var_t
        climatology_variance: numpy.ndarray
            Climatological variance,
            If n is the number of elements of independent_var_t,
            the shape of this should either be n x n (like ellipse covariance)
            or n (vector of variance at each grid point)
        climatology_variance_is_sdev: bool
            Flag indicating if climatology_variance is
            variance or standard deviation. This only applies if vectored form
            (n x 1 shape) of climatology_variance
            Default False, climatology_variance is variance (like SST**2) or
            full covariance matrix.
            True if standard deviation
        """
        #
        self.independent_var_t = independent_var_t
        self.errcov_independent_var_t = errcov_independent_var_t
        self.lag_1_autocor = lag_1_autocor
        self.climatology_mean = climatology_mean
        if climatology_variance_is_sdev:
            if len(climatology_variance.shape) > 1:
                err_msg = "climatology_variance_is_sdev is True, but "
                err_msg += "climatology_variance is not a vector; "
                err_msg += f"{climatology_variance.shape = }."
                raise ValueError(err_msg)
            self.climatology_variance = np.square(climatology_variance)
        else:
            self.climatology_variance = climatology_variance
        self._check_args()

    def _check_args(self):
        """Check attributes set on init"""
        if len(self.independent_var_t.shape) != 1:
            raise ValueError("independent_var_t should be 1D")
        if len(self.errcov_independent_var_t.shape) != 2:
            raise ValueError("errcov_independent_var_t should be 2D")
        if (
            self.errcov_independent_var_t.shape[0]
            != self.errcov_independent_var_t.shape[1]
        ):  # noqa: E501
            raise ValueError("errcov_independent_var_t is not square")
        if len(self.lag_1_autocor.shape) != 1:
            raise ValueError("lag_1_autocor_for_t_plus_1 should be 1D")
        if len(self.climatology_mean.shape) != 1:
            raise ValueError("climatology_mean should be 1D")
        if len(self.climatology_variance.shape) > 2:
            raise ValueError("climatology_variance should be 1 or 2D")
        #
        if (
            self.independent_var_t.shape[0]
            != self.errcov_independent_var_t.shape[0]
        ):  # noqa: E501
            raise ValueError(
                "independent_var_t shape inconsistent with errcov_independent_var_t"  # noqa: E501
            )
        if self.independent_var_t.shape[0] != self.climatology_mean.shape[0]:
            raise ValueError(
                "independent_var_t shape inconsistent with climatology_mean"
            )  # noqa: E501
        if self.climatology_variance.shape[0] != self.climatology_mean.shape[0]:
            raise ValueError(
                "climatology_variance shape inconsistent with climatology_mean"
            )  # noqa: E501

    def compute_forecast(self):
        """
        Compute AR1 forecast and estimate uncertainties

        Speed:
        https://stackoverflow.com/questions/44388358/python-numpy-matrix-multiplication-with-one-diagonal-matrix

        This version uses *, less np.diag, and should be faster.

        Parameters
        ----------
        independent_var_t: numpy.ndarray
            1D vector of independent variables for t
        errcov_independent_var_t: numpy.ndarray
            2D errcov for independent_var_t
        lag_1_autocor: numpy.ndarray
            1D vector of lag correlation
        climatology_mean: numpy.ndarray
            1D climatological mean for independent_var
        climatology_variance: numpy.ndarray
            climatology_variance: 1D climatological variance for independent_var

        Returns
        -------
        forecast_t_plus_1_anomaly: numpy.ndarray
            AR1 forecast
        errcov: numpy.ndarray
            The error covariance for the forecast
        """
        print("Computing forecast")
        diff_with_climatology = self.independent_var_t - self.climatology_mean
        self.forecast = (self.lag_1_autocor * diff_with_climatology.T).T
        self.forecast += self.climatology_mean
        #
        print("Computing uncertainties")
        lag_1_autocor_squared = self.lag_1_autocor * self.lag_1_autocor
        # Compute error covariances associated with AR1 epsilon
        # that are associated with climatological variance
        if len(self.climatology_variance.shape) == 1:
            # climatology_variance is a vector of variances
            # No off-diagonal/correlated terms
            # Not a usually useful method
            # (1-PHI*PHI) * CLIM_VAR
            climvar_mult = (
                np.ones_like(lag_1_autocor_squared) - lag_1_autocor_squared
            )
            errcov_clim = np.diag(
                (climvar_mult * self.climatology_variance.T).T
            )
        else:
            # climatology_variance is a covariance matrix
            # Off-diagonals/correlated terms included
            # For covariance functions determined using
            # variogram analyses
            # DIAG(SQRT(1-PHI*PHI)) @ CLIM_VAR @ DIAG(SQRT(1-PHI*PHI))
            climvar_mult = np.sqrt(
                np.ones_like(lag_1_autocor_squared) - lag_1_autocor_squared
            )
            errcov_clim = (climvar_mult * self.climatology_variance).T
            errcov_clim = errcov_clim * climvar_mult
        # Compute error covariance associated uncertaintes of the inputs
        # DIAG(PHI) @ OBS_UNCERT @ DIAG(PHI)
        errcov_uncert_ind_var = (
            self.lag_1_autocor * self.errcov_independent_var_t
        ).T
        errcov_uncert_ind_var = errcov_uncert_ind_var * self.lag_1_autocor
        #
        self.errcov = errcov_clim + errcov_uncert_ind_var
