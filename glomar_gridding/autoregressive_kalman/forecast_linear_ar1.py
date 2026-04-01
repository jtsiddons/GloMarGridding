"""
Conduct local AR1 forecast of uncertain data
output new uncertainties
"""

import numpy as np
import scipy as sp
import warnings


class Autoregressive1Forecast:
    """
    Class to compute AR1 forecast

    Local AR only for now, the full vector AR model is not ready yet.
    All predictions are based on local autocorrelation only
    To work with vector form, full autocovariance is needed.

    Compute Lag-1 autoregressive forecast as a prior for Kalman filter.
    """

    def __init__(
        self,
        obs: np.ndarray,
        errcov_obs: np.ndarray,
        lag_1_autocov: np.ndarray,
        clim_mean: np.ndarray,
        clim_covar: np.ndarray,
        errcov_obs_is_sdev: bool = False,
        clim_covar_is_sdev: bool = False,
        predict_local: bool = True,
    ):
        """
        __init__ for Autoregressive1Forecast class

        Parameters
        ----------
        obs: numpy.ndarray
            1D vector of data for t=t
        errcov_obs: numpy.ndarray
            1D error variance or
            2D error covariance for obs
        lag_1_autocov: numpy.ndarray
            1D vector of lag-1 autocorrelation or
            2D full lag-1 autocovariance
            Note: 1D case should be autocorrelation (standardised)!
        clim_mean: numpy.ndarray
            1D vector of climatological mean
            Shape should be same as obs
        clim_covar: numpy.ndarray
            Climatological covariance,
            Can be 1D or 2D,
            The shape of this should either be n x n (like ellipse covariance)
            or n (vector of variance at each grid point)
        errcov_obs_is_sdev: bool
            Flag indicating if errcov_obs is
            variance or standard deviation.
            This only applies to vectored form (n x 1 shape) of errcov_obs
            Default False, errcov_obs is variance (like SST**2) or
            full covariance matrix.
            True if standard deviation
        clim_covar_is_sdev: bool
            Flag indicating if clim_covar is
            variance or standard deviation.
            This only applies to vectored form (n x 1 shape) of clim_covar
            Default False, clim_covar is variance (like SST**2) or
            full covariance matrix.
            True if standard deviation
        predict_local: bool
            True forces local prediction
            It is always True if clim_covar is 1D.
            Not vector autoregression
        """
        #
        self.obs = obs
        print(f"{errcov_obs_is_sdev = }")
        if errcov_obs_is_sdev:
            if len(errcov_obs.shape) > 1:
                err_msg = "errcov_obs_is_sdev is True, but "
                err_msg += "errcov_obs is not a vector; "
                err_msg += f"{errcov_obs.shape = }."
                raise ValueError(err_msg)
            self.errcov_obs = np.square(errcov_obs)
        else:
            self.errcov_obs = errcov_obs
        self.lag_1_autocov = lag_1_autocov
        self.clim_mean = clim_mean
        print(f"{clim_covar_is_sdev = }")
        if clim_covar_is_sdev:
            if len(clim_covar.shape) > 1:
                err_msg = "clim_covar_is_sdev is True, but "
                err_msg += "clim_covar is not a vector; "
                err_msg += f"{clim_covar.shape = }."
                raise ValueError(err_msg)
            self.clim_covar = np.square(clim_covar)
        else:
            self.clim_covar = clim_covar
        #
        print(f"{predict_local = }")
        if predict_local or len(lag_1_autocov.shape) == 1:
            print("Local autoregressive predictions only.")
            self.predict_local = True
            self.compute_forecast = self.compute_forecast_local
            if len(self.clim_covar.shape) == 2:
                self.clim_covar = np.diag(self.clim_covar)
            if len(self.errcov_obs.shape) == 2:
                self.errcov_obs = np.diag(self.errcov_obs)
            if len(self.lag_1_autocov.shape) == 2:
                self.lag_1_autocov = np.diag(self.lag_1_autocov)
        else:
            # raise NotImplementedError("Vector autoregression is WIP")
            self.compute_forecast = self.compute_forecast_vector
            print(f"{self.clim_covar.shape = }")
            print(f"{self.errcov_obs.shape = }")
            print(f"{self.lag_1_autocov.shape = }")
        #
        self._check_args()
        self.bad_model = None

    def _check_args(self):  # noqa: C901
        """Check attributes set on init"""
        # 1D variable check
        if len(self.obs.shape) != 1:
            raise ValueError("independent_var_t should be 1D")
        if len(self.clim_mean.shape) != 1:
            raise ValueError("climatology_mean should be 1D")
        if self.obs.shape[0] != self.clim_mean.shape[0]:
            raise ValueError("obs shape inconsistent with clim_mean")
        #
        # 1 or 2D variable check
        if len(self.lag_1_autocov.shape) == 1:
            advisory = "lag_1_autocov is a vector; "
            advisory += "make sure this is autocorrelation, "
            advisory += "otherwise it would not work!"
            print(advisory)
        else:
            if not self._check_sq_matrix(self.lag_1_autocov):
                err_msg = f"Bad shape lag_1_autocov {self.lag_1_autocov.shape}."
                raise ValueError(err_msg)
        #
        if len(self.clim_covar.shape) == 1:
            print("clim_covar is 1D.")
        else:
            if not self._check_sq_matrix(self.clim_covar):
                err_msg = f"Bad shape clim_covar {self.clim_covar.shape}."
                raise ValueError(err_msg)
        #
        if len(self.errcov_obs.shape) == 1:
            print("errcov_obs is 1D.")
        else:
            if not self._check_sq_matrix(self.errcov_obs):
                err_msg = f"Bad shape errcov_obs {self.errcov_obs.shape}."
                raise ValueError(err_msg)
        print(f"{self.clim_covar.shape = }")
        print(f"{self.errcov_obs.shape = }")
        print(f"{self.lag_1_autocov.shape = }")
        if self.clim_covar.shape != self.errcov_obs.shape:
            raise ValueError("Inconsistent shape detected!")
        if self.errcov_obs.shape != self.lag_1_autocov.shape:
            raise ValueError("Inconsistent shape detected!")

    def _check_sq_matrix(self, arr: np.ndarray):
        check1 = arr.shape[0] == arr.shape[1]
        check2 = len(arr.shape) == 2
        return check1 and check2

    def compute_forecast(self):
        """Call default compute_forecast based on set attributes"""
        if self.predict_local:
            self.compute_forecast_local()
        else:
            self.compute_forecast_vector()

    def compute_forecast_local(self, full_errcov_out: bool = True):
        """
        Compute AR1 forecast and estimate uncertainties

        Speed:
        https://stackoverflow.com/questions/44388358/python-numpy-matrix-multiplication-with-one-diagonal-matrix

        This version uses scalar multiplication, less use of np.diag, and should
        be faster.

        Parameters
        ----------
        obs: numpy.ndarray
            1D vector of independent variables for t
        errcov_obs: numpy.ndarray
            2D errcov for obs
        lag_1_autocov: numpy.ndarray
            1D vector of lag correlation
        clim_mean: numpy.ndarray
            1D climatological mean for obs
        clim_covar: numpy.ndarray
            climatology_variance: 1D (local) climatological variance for obs

        Returns
        -------
        forecast: numpy.ndarray
            AR1 forecast
        errcov: numpy.ndarray
            The error covariance for the forecast
        """
        print("Computing forecast")
        #
        # The simple local case:
        self.weights = np.diag(self.lag_1_autocov)
        diff_with_clim_mean = self.obs - self.clim_mean
        self.forecast = self.lag_1_autocov * diff_with_clim_mean
        self.forecast += self.clim_mean
        #
        print("Computing uncertainties")
        # Compute error covariances associated with AR1 epsilon
        # that are associated with climatological variance
        #
        # This only works 1D/local prediction
        # clim_covar is a vector of climatological variances
        # and errcov_obs is a vector of local uncertainties
        # No off-diagonal/correlated terms
        # Works fast but is crude, and don't use all possible info
        #
        # Equation:
        # sigma_sq_eps = (1 - PHI * PHI) * clim_covar
        # sigma_sq_obs = PHI * PHI * errcov_obs
        autocorr_sq = np.square(self.lag_1_autocov)
        climvar_mult = np.ones_like(autocorr_sq) - autocorr_sq
        sigma_sq_eps = climvar_mult * self.clim_covar
        sigma_sq_obs = autocorr_sq * self.errcov_obs
        self.errcov = sigma_sq_eps + sigma_sq_obs
        self.bad_model = False
        if full_errcov_out:
            self.errcov = np.diag(self.errcov)

    def compute_forecast_vector(
            self,
            check_wgt_stability: bool = True,
            full_errcov_out: bool = True,
            stability_perturbation: float = 0.05 ** 2):
        """
        Parameters
        ----------
        obs: numpy.ndarray
            1D vector of independent variables for t
        errcov_obs: numpy.ndarray
            2D errcov for obs
        lag_1_autocov: numpy.ndarray
            2D matrix of autocovariance
        clim_mean: numpy.ndarray
            1D climatological mean for obs
        clim_covar: numpy.ndarray
            climatology_variance: 2D climatological covariance for obs
        check_wgt_stability: bool
            Check weights (in extension for the auto and climatological
            covariances) are stable; see var_stability_check
        full_errcov_out: bool
            Return full error covariance if True
        stability_perturbation: float
            In some cases, prediction is possible even with weights being
            unstable, but a perturbation may be needed to ensure sensible
            results are obtainable.

        Returns
        -------
        forecast: numpy.ndarray
            AR1 forecast
        errcov: numpy.ndarray
            The error covariance for the forecast
        """
        # raise NotImplementedError("Not finished yet.")
        print("Computing weights")
        # W = <x(t), x(t-1)> @ <x(t), x(t)>**-1
        # W @ <x(t), x(t)> = <x(t), x(t-1)>
        # <x(t), x(t)> @ W.T = <x(t), x(t-1)>
        # (note: <x(t), x(t-1)> and <x(t), x(t)> are symmetric)
        ccp = stability_perturbation + self.clim_covar
        self.weights = np.linalg.solve(
            ccp,
            self.lag_1_autocov,
        ).T
        print(f"{self.weights = }")
        if check_wgt_stability:
            self.var_stability_check()
        else:
            warn_msg = "check_wgt_stability is disabled; "
            warn_msg += "check is recommended as VAR is often unstable. "
            warn_msg += "As precaution, self.bad_model defaults to True."
            warnings.warn(warn_msg, UserWarning)
            self.bad_model = True
        #
        print("Computing forecast")
        diff_with_clim_mean = self.obs - self.clim_mean
        self.forecast = self.weights @ diff_with_clim_mean
        self.forecast += self.clim_mean
        #
        # https://math.stackexchange.com/questions/5004102/covariance-of-a-multivariate-autoregression
        # I didn't cheat
        # I got same equation independently using good ole pen and paper!
        #
        # If weights fail stability check, you will get wrong uncertainty
        # estimates, like off diagonal term being (considerably) larger than
        # diagonal terms; that's an invalid (error) covariance matrix, saying
        # correlation > 1 which is impossible!
        print("Computing uncertainties")
        left = ccp
        right = self.weights @ ccp @ self.weights.T
        print(f"{left = }")
        print(f"{right = }")
        sigma_sq_eps = left - right
        #
        # Add input uncertainty
        sigma_sq_obs = self.weights @ self.errcov_obs @ self.weights.T
        self.errcov = sigma_sq_eps + sigma_sq_obs
        if not full_errcov_out:
            self.errcov = np.diag(self.errcov)

    def var_stability_check(self, warn_instead: bool = True):
        """
        A multivariate AR model is only stable if and only all eigvals
        of the weights have absolute values less than 1.0
        https://kevinkotze.github.io/ts-7-slide/#10
        which is from
        https://www.jstor.org/stable/j.ctv14jx6sm
        Chapter 10 Proposition 10.1
        """
        print("Checking weight stability")
        smallest_eigval = sp.linalg.eigh(
            self.weights,
            subset_by_index=(0, 0),
            eigvals_only=True,
        )[0]
        largest_eigval = sp.linalg.eigh(
            self.weights,
            subset_by_index=(
                self.weights.shape[0] - 1,
                self.weights.shape[0] - 1
            ),
            eigvals_only=True,
        )[0]
        print(f"{smallest_eigval = }")
        print(f"{largest_eigval = }")
        self.bad_model = np.logical_or(
            smallest_eigval < -1.0,
            largest_eigval > 1.0,
        )
        if self.bad_model:
            errmsg = "Eigenvalues of estimated weights have values > 1; "
            errmsg += "autocovariance and contempory "
            errmsg += "covariance do not satisfy stationarity requirements, "
            errmsg == "and error covariances are incorrect."
            if warn_instead:
                warnings.warn(errmsg, UserWarning)
            else:
                raise ValueError(errmsg)
