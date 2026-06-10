"""Kalman Filter: Compute inverse variance weighted average"""

import logging
import numpy as np
from numpy import linalg
import scipy as sp
import warnings

from glomar_gridding.autoregressive_kalman import cov_diagonal as cd
from glomar_gridding.utils import (
    check_sq_matrix,
    _vec_or_sq_matrix_check,
    make_square,
    make_diag,
)

EFFECTIVELY_ZERO_VAR_DEFAULT = 1e-6


def matmul(
    a: np.ndarray | sp.sparse.sparray,
    b: np.ndarray | sp.sparse.sparray,
):
    """
    Matrix multiplication
    - np.matmul does not work with sp.sparse.sparray; it will throw an exception
    - "@" always work, but "@" cannot be passed as a function like np.multiply.
    """
    return a @ b


class KalmanOut:
    """
    class to compute blended forecast and observations
    variable names follows compute_inv_variance_wgt_mean_kalman

    Note:
    If only one of the two error covariances matrices is a vector/diagonal
    matrix but the other one is not (square) and is much denser with
    comparable diagonals, KF and variance weighted means
    may strongly favour the vector/diagonal matrix, and give poor
    results (even if results are strictly correct). Check your
    underlying reason why one is doing that to make sure that is what
    one intended.
    Vector autoregression should only match with full Kriging covariance
    (a fully spatially correlated Kalman filter); local time series
    autoregression would normally only match with local uncertainty of the
    Kriging covariance (i.e. the Kriging uncertainty) (i.e. effectively a 1D
    Kalman Filter).

    Parameters
    ----------
    forecast_vector: numpy.ndarray
        1D vector of forecasts
    obs_vector: numpy.ndarray
        1D vector of (gridded) observations
    errcov_forecast: numpy.ndarray
        2D matrix of errcov for forecast_vector
    errcov_obs: numpy.ndarray
        2D matrix of errcov for obs_vector
    cov_forecast_and_obs: numpy.ndarray | None
        covariance between forecast/first-guess & observations
        In NWP-DA, this is usually assumed to be zero
        aka observation and forecast errors are not correlated.
        Set it to None if that is zero
    use_diag_only: bool
        Ignore off-diagonals of errcov_forecast, errcov_obs,
        and cov_forecast_and_obs if set to True, default True
    """

    def __init__(
        self,
        forecast_vector: np.ndarray,
        obs_vector: np.ndarray,
        errcov_forecast: np.ndarray,
        errcov_obs: np.ndarray,
        cov_forecast_and_obs: np.ndarray | None,
        use_diag_only: bool = True,
    ):
        """__init__ for KalmanOut class"""
        self.forecast_vector = forecast_vector
        self.obs_vector = obs_vector
        if use_diag_only:
            self.use_diag_only = True
            self.errcov_forecast = make_diag(errcov_forecast)
            self.errcov_obs = make_diag(errcov_obs)
            if cov_forecast_and_obs is not None:
                self.cov_forecast_and_obs = make_diag(cov_forecast_and_obs)
            else:
                self.cov_forecast_and_obs = None
            self.multiply_operator = np.multiply
            self.one_maker = np.ones
        else:
            self.use_diag_only = False
            # See docstring for further discussion of this check:
            # XOR(
            # _vec_or_sq_matrix_check(errcov_obs),
            # _vec_or_sq_matrix_check(errcov_forecast)
            # ) returns True only if one is True and the other is False
            if _vec_or_sq_matrix_check(errcov_obs) != _vec_or_sq_matrix_check(
                errcov_forecast
            ):
                advisory = "One of the arrays (errcov_obs or errcov_forecast) "
                advisory += "is a vector while the other is square. This could "
                advisory += "produce poor results depending on the reason why "
                advisory += "the shapes of them are like that : "
                advisory += f"{errcov_obs.shape = }  {errcov_forecast.shape = }"
                warnings.warn(advisory, UserWarning)
            self.errcov_forecast = make_square(errcov_forecast)
            self.errcov_obs = make_square(errcov_obs)
            if cov_forecast_and_obs is not None:
                self.cov_forecast_and_obs = make_square(cov_forecast_and_obs)
            else:
                self.cov_forecast_and_obs = None
            self.multiply_operator = matmul
            self.one_maker = np.eye

    def sparse_approx_4_errcov(
        self,
        sparse_threshold: float = EFFECTIVELY_ZERO_VAR_DEFAULT,
        convert2sparse: bool = True,
    ):
        """
        Setting small elements of self.errcov_obs and errcov_forecast to zero
        as defined by sparse_threshold value

        If there are small values in the diagonals

        This reduces memory footprint and may make block-splitting
        by cov_diagonal easier.

        It is not intended to be use for use_diag_only mode (?).
        This is only used to handle situations that the error covariances
        are big.

        Parameters
        ----------
        sparse_threshold: float
            (Absolute) off-diagonal values that are to set to zero
        convert2sparse: bool
            Convert error covariances to scipy sparse array if True
        """
        if self.use_diag_only:
            err_msg = "This method is not intended to be use with "
            err_msg += "use_diag_only."
            raise ValueError(err_msg)
        self.errcov_forecast = self._small_elements_2_zero_and_sparse(
            self.errcov_forecast,
            sparse_threshold=sparse_threshold,
            convert2sparse=convert2sparse,
        )
        self.errcov_obs = self._small_elements_2_zero_and_sparse(
            self.errcov_obs,
            sparse_threshold=sparse_threshold,
            convert2sparse=convert2sparse,
        )

    def _small_elements_2_zero_and_sparse(
        self,
        arr: np.ndarray,
        sparse_threshold: float = EFFECTIVELY_ZERO_VAR_DEFAULT,
        leave_diagonal_alone: bool = True,
        convert2sparse: bool = True,
    ) -> np.ndarray | sp.sparse.sparray:
        """
        Set "small" elements of the array to zero
        Convert of scipy sparse if required

        Parameters
        ----------
        arr: numpy.ndarray
            Array to be manipulated
        sparse_threshold: float
            threshold in which values less than to be to set 0
        leave_diagonal_alone: bool
            Set to True to leave the diagonal unchanged
            True by default
        convert2sparse: bool
            Convert arr to scipy sparse matrix if True
            True by default

        Returns
        -------
        arr: numpy.ndarray | scipy.sparse.sparray
            Modified sparse array
        """
        if len(arr.shape) != 2:
            raise ValueError("This function accepts 2D arrays only.")
        if leave_diagonal_alone:
            gaid = np.diag(arr)
        arr[np.abs(arr) < sparse_threshold] = 0.0
        if leave_diagonal_alone:
            np.fill_diagonal(arr, gaid)
        if convert2sparse:
            arr = sp.sparse.csc_array(arr)
        return arr

    def compute_outputs(self):
        """
        Compute the inverse variance weighted average of
        forecast_vector & obs_vector using provided error covariances
        errcov_forecast & errcov_obs

        This uses a form that requires only ONE matrix inverses
        and reciprocals (good!) and is more commonly seen in
        Kalman Filter guides (including the form uses in Wikipedia)
        https://en.wikipedia.org/wiki/Kalman_filter
        Probably everyone hate matrix inverses... (for good reason)

        1D vector form of Kalman gain:
        gain = forecast_err / (forecast_err + obs_err)

        Matrix form of Kalman gain:
        gain = forecast_err @ inv(forecast_err + obs_err)
        gain @ (forecast_err + obs_err) = forecast_err
        (gain @ (forecast_err + obs_err)).T = forecast_err.T
        (forecast_err + obs_err)).T @ gain.T = forecast_err.T

        but forecast_err and obs_err are symmetric
        Hence solve set of linear equations that:
        (forecast_err + obs_err) @ gain.T = forecast_err

        Attributes
        ----------
        wgt_mean: numpy.ndarray
            Posterior analysis
        errcov: numpy.ndarray
            Posterior error covariance
        kalman_gain_from_new_obs: numpy.ndarray
            Kalman gain (either a 2D matrix or 1D vector)
            aka the weights assigned to the new information
        wgts_from_ar_forecast: numpy.ndarray
            Identity matrix or one vector minus Kalman gain
            aka the weights assigned to the prior/first-guess
        """
        forecast_vector_shape = self.forecast_vector.shape
        if len(forecast_vector_shape) != 1:
            raise ValueError("forecast_vector is not 1D vector")
        if self.errcov_forecast.shape[0] != forecast_vector_shape[0]:
            raise ValueError(
                "forecast_vector shape inconsistent with errcov_forecast"
            )
        #
        obs_vector_shape = self.obs_vector.shape
        if len(obs_vector_shape) != 1:
            raise ValueError("obs_vector is not 1D vector")
        if self.errcov_obs.shape[0] != obs_vector_shape[0]:
            raise ValueError("obs_vector shape inconsistent with errcov_obs")
        #
        if forecast_vector_shape[0] != obs_vector_shape[0]:
            raise ValueError(
                "obs_vector shape inconsistent with forecast_vector"
            )
        #
        # Weights and error covariance if obs and forecast are uncorrelated
        logging.debug("Computing kalman_gain")
        if self.use_diag_only:
            # Vector form
            self.kalman_gain_from_new_obs = self.multiply_operator(
                self.errcov_forecast,
                np.reciprocal(self.errcov_forecast + self.errcov_obs),
            )
        else:
            # Matrix form
            self.kalman_gain_from_new_obs = linalg.solve(
                self.errcov_forecast + self.errcov_obs,
                self.errcov_forecast,
            ).T
        #
        logging.debug("Computing forecast_wgt")
        self.wgts_from_ar_forecast = (
            self.one_maker(self.kalman_gain_from_new_obs.shape[0])
            - self.kalman_gain_from_new_obs
        )
        #
        # Output weighted mean
        logging.debug("Computing weighted mean")
        self.wgt_mean = self.multiply_operator(
            self.kalman_gain_from_new_obs,
            (self.obs_vector - self.forecast_vector),
        )
        self.wgt_mean += self.forecast_vector
        #
        # Output error covariance
        logging.debug("Computing updating uncertainties")
        self.errcov = self.multiply_operator(
            (
                self.one_maker(self.errcov_obs.shape[0])
                - self.kalman_gain_from_new_obs
            ),
            self.errcov_forecast,
        )
        if self.cov_forecast_and_obs is not None:
            w1w2cov = self.multiply_operator(
                self.multiply_operator(
                    self.kalman_gain_from_new_obs,
                    self.wgts_from_ar_forecast,
                ),
                self.cov_forecast_and_obs,
            )
            self.errcov += self.multiply_operator(
                2.0 * self.one_maker(obs_vector_shape[0]), w1w2cov
            )


class KalmanOutUncorrCorrSplit:
    """
    class to compute blended forecast and observations
    This splits the error covariances into diagonal and non-diagonal bits

    It is a wrapper for KalmanOut.

    Parameters
    ----------
    forecast_vector: numpy.ndarray
        1D vector of forecasts
    obs_vector: numpy.ndarray
        1D vector of (gridded) observations
    errcov_forecast: numpy.ndarray
        2D matrix of errcov for forecast_vector
    errcov_obs: numpy.ndarray
        2D matrix of errcov for obs_vector
    cov_forecast_and_obs: numpy.ndarray | None
        covariance between forecast & observations
        Set it to None if that is zero
    arr_2_decide_if_points_are_isolated: numpy.ndarray
        The matrix (usually a covariance) to determine
        if the point is diagonally isolated. This can
        be the prior spatial covariance.
    zero_threshold: float
        The threshold applied to arr_2_decide_if_points_are_isolated
        to decide if the row/column is diagonal.
    """

    def __init__(
        self,
        forecast_vector: np.ndarray,
        obs_vector: np.ndarray,
        errcov_forecast: np.ndarray,
        errcov_obs: np.ndarray,
        cov_forecast_and_obs: np.ndarray | None,
        arr_2_decide_if_points_are_isolated: np.ndarray,
        zero_threshold: float = cd.EFFECTIVELY_ZERO_DEFAULT,
    ):
        """__init__ for KalmanOut class"""
        #
        if not check_sq_matrix(errcov_forecast):
            raise ValueError("errcov_forecast should be 2D and square")
        if not check_sq_matrix(errcov_obs):
            raise ValueError("errcov_obs should be 2D and square")
        if cov_forecast_and_obs is not None:
            if not check_sq_matrix(cov_forecast_and_obs):
                raise ValueError("cov_forecast_and_obs should be 2D and square")
        #
        self.d_off_diagonal, _, self.d_diagonal_only, _ = (
            cd.diag_and_nondiag_rows_subsampler(  # noqa: E501
                # errcov_forecast + errcov_obs,
                arr_2_decide_if_points_are_isolated,
                zero_threshold=zero_threshold,
                return_subsampled_arr=False,
            )
        )
        logging.debug(f"{self.d_off_diagonal = }")
        logging.debug(f"{np.sum(self.d_off_diagonal) = }")
        logging.debug(f"{self.d_diagonal_only = }")
        logging.debug(f"{np.sum(self.d_diagonal_only) = }")
        #
        has_diag_only = self._d_2_bool(self.d_diagonal_only)
        #
        forecast_vector_d = forecast_vector[has_diag_only]
        obs_vector_d = obs_vector[has_diag_only]
        #
        errcov_forecast_d = errcov_forecast[has_diag_only, :][:, has_diag_only]
        errcov_obs_d = errcov_obs[has_diag_only, :][:, has_diag_only]
        #
        cov_forecast_and_obs_d = None
        if cov_forecast_and_obs is not None:
            cov_forecast_and_obs_d = cov_forecast_and_obs[has_diag_only, :][
                :, has_diag_only
            ]
            cov_forecast_and_obs_d = np.diag(cov_forecast_and_obs_d)
        #
        has_off_diagonal = self._d_2_bool(self.d_off_diagonal)
        #
        forecast_vector_c = forecast_vector[has_off_diagonal]
        obs_vector_c = obs_vector[has_off_diagonal]
        #
        errcov_forecast_c = errcov_forecast[has_off_diagonal, :][
            :, has_off_diagonal
        ]
        errcov_obs_c = errcov_obs[has_off_diagonal, :][:, has_off_diagonal]
        #
        cov_forecast_and_obs_c = None
        if cov_forecast_and_obs is not None:
            cov_forecast_and_obs_c = cov_forecast_and_obs[has_off_diagonal, :][
                :, has_off_diagonal
            ]
        #
        self.uncorr_part = KalmanOut(
            forecast_vector_d,
            obs_vector_d,
            np.diag(errcov_forecast_d),
            np.diag(errcov_obs_d),
            cov_forecast_and_obs_d,
            use_diag_only=True,
        )
        self.corr_part = KalmanOut(
            forecast_vector_c,
            obs_vector_c,
            errcov_forecast_c,
            errcov_obs_c,
            cov_forecast_and_obs_c,
            use_diag_only=False,
        )

    def solve_uncorr(self):
        """Alias for instance_name.uncorr_part.compute_outputs()"""
        self.uncorr_part.compute_outputs()

    def solve_corr(self):
        """Alias for instance_name.corr_part.compute_outputs()"""
        self.corr_part.compute_outputs()

    def _d_2_bool(
        self,
        d_matrix: np.ndarray | sp.sparse.sparray,
    ):
        """Convert subsampling matrix D to a vector of bool"""
        if isinstance(d_matrix, np.ndarray):
            return np.any(d_matrix, axis=0)
        elif sp.sparse.issparse(d_matrix):
            return np.any(d_matrix.toarray(), axis=0)
        else:
            raise ValueError("Unknown type d_matrix")

    def combine_results(self):
        """Blend the results of uncorrelated and correlated streams"""
        if not hasattr(self.uncorr_part, "kalman_gain_from_new_obs"):
            err_msg = "Output attributes missing for uncorr_part; "
            err_msg += "do instance_name.uncorr_part.compute_outputs() first."
            raise AttributeError(err_msg)
        if not hasattr(self.corr_part, "kalman_gain_from_new_obs"):
            err_msg = "Output attributes missing for corr_part; "
            err_msg += "do instance_name.corr_part.compute_outputs() first."
            raise AttributeError(err_msg)
        self.wgt_mean = np.array(
            np.matrix(self.uncorr_part.wgt_mean) @ self.d_diagonal_only
            + np.matrix(self.corr_part.wgt_mean) @ self.d_off_diagonal
        )[0]
        #
        self.errcov = cd.restore_diag_only_rows(
            np.diag(self.uncorr_part.errcov),
            self.d_diagonal_only,
            diag_fill_value=0.0,
        ) + cd.restore_diag_only_rows(
            self.corr_part.errcov, self.d_off_diagonal, diag_fill_value=0.0
        )
        #
        self.kalman_gain_from_new_obs = cd.restore_diag_only_rows(
            np.diag(self.uncorr_part.kalman_gain_from_new_obs),
            self.d_diagonal_only,
            diag_fill_value=0.0,
        ) + cd.restore_diag_only_rows(
            self.corr_part.kalman_gain_from_new_obs,
            self.d_off_diagonal,
            diag_fill_value=0.0,
        )
        #
        self.wgts_from_ar_forecast = cd.restore_diag_only_rows(
            np.diag(self.uncorr_part.wgts_from_ar_forecast),
            self.d_diagonal_only,
            diag_fill_value=0.0,
        ) + cd.restore_diag_only_rows(
            self.corr_part.wgts_from_ar_forecast,
            self.d_off_diagonal,
            diag_fill_value=0.0,
        )
