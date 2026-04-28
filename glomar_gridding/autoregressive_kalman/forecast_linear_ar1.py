"""
Conduct local AR1 forecast of uncertain data
output new uncertainties

for the purpose of docstring
N: number of analysis points, covariates/features
"""

import numpy as np
import scipy as sp
import warnings


class Autoregressive1ForecastUncorr:
    """
    Class to compute AR1 forecast

    Local/uncorrelated only: Lag-1 correlations are the
    weights and are in 1-D vector. All predictions are based on
    local autocorrelation only

    Use Autoregressive1Forecast for full vector case.

    Compute Lag-1 autoregressive forecast as a prior for Kalman filter.
    """

    def __init__(
        self,
        analysis: np.ndarray,
        errcov_analysis: np.ndarray,
        lag_1_autocor: np.ndarray,
        clim_mean: np.ndarray,
        clim_covar: np.ndarray,
        errcov_analysis_is_sdev: bool = False,
        clim_covar_is_sdev: bool = False,
    ):
        """
        __init__ for Autoregressive1Forecast class

        Parameters
        ----------
        analysis: numpy.ndarray
            1D vector of analysis
            (such as Kriging and Kalman filter outputs)
            for t=t
            dtype `float`, shape `(N, )`
        errcov_analysis: numpy.ndarray
            1D error variance or
            2D error covariance for analysis
            if latter, it will only use the diagonal
            dtype `float`, shape `(N, )` | `(N, N)`
        lag_1_autocor: numpy.ndarray
            1D vector of lag-1 autocorrelation
            Shape should be same as analysis
            dtype `float`, shape `(N, )`
        clim_mean: numpy.ndarray
            1D vector of climatological mean
            Shape should be same as analysis
            dtype `float`, shape `(N, )`
        clim_covar: numpy.ndarray
            Climatological covariance,
            Can be 1D or 2D,
            The shape of this should either be n x n (like ellipse covariance)
            or n (vector of variance at each grid point)
            If a 2D matrix is provided, it will take the diagonal
            dtype `float`, shape `(N, )` | `(N, N)`
        errcov_analysis_is_sdev: bool
            Flag indicating if errcov_analysis is
            variance or standard deviation.
            This only applies to vectored form (n x 1 shape) of errcov_analysis
            Default False, errcov_analysis is variance (like SST**2) or
            full covariance matrix.
            True if standard deviation
        clim_covar_is_sdev: bool
            Flag indicating if clim_covar is
            variance or standard deviation.
            This only applies to vectored form (n x 1 shape) of clim_covar
            Default False, clim_covar is variance (like SST**2) or
            full covariance matrix.
            True if standard deviation
        """
        #
        # analysis
        self.analysis = analysis
        #
        # (diagonal) analysis error covariance
        print(f"{errcov_analysis_is_sdev = }")
        if errcov_analysis_is_sdev:
            if len(errcov_analysis.shape) > 1:
                err_msg = "errcov_analysis_is_sdev is True, but "
                err_msg += "errcov_analysis is not a vector; "
                err_msg += f"{errcov_analysis.shape = }."
                raise ValueError(err_msg)
            self.errcov_analysis = np.square(errcov_analysis)
        else:
            self.errcov_analysis = errcov_analysis
        #
        # lag-1 autocorrelation
        self.lag_1_autocor = lag_1_autocor
        #
        # climatological mean
        self.clim_mean = clim_mean
        #
        # (diagonal) climatological variance
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
        # Names of methods to compute prrediction
        self.compute_forecast = self.compute_forecast_local
        self.predict = self.compute_forecast_local  # sklearn standard
        #
        # Use only diagonal if 2D matrices are provided for
        # clim_covar and errcov_analysis
        if len(self.clim_covar.shape) == 2:
            if not self._check_sq_matrix(self.clim_covar):
                raise ValueError(f"Not square {self.clim_covar.shape = }.")
            self.clim_covar = np.diag(self.clim_covar)
        if len(self.errcov_analysis.shape) == 2:
            if not self._check_sq_matrix(self.errcov_analysis):
                raise ValueError(f"Not square {self.errcov_analysis.shape = }.")
            self.errcov_analysis = np.diag(self.errcov_analysis)
        #
        self._check_args()

    def _check_args(self):
        """Check attributes set on init"""
        # 1D variable check
        if len(self.analysis.shape) != 1:
            raise ValueError("analysis should be 1D")
        if len(self.clim_mean.shape) != 1:
            raise ValueError("clim_mean should be 1D")
        if self.analysis.shape[0] != self.clim_mean.shape[0]:
            raise ValueError("analysis shape inconsistent with clim_mean")
        #
        print(f"{self.lag_1_autocor.shape = }")
        if len(self.lag_1_autocor.shape) != 1:
            err_msg = "lag_1_autocor is not a vector. "
            err_msg += "This class only accepts autocorrelations."
            raise ValueError(err_msg)
        self.bad_model = np.any(np.abs(self.lag_1_autocor) >= 1)
        if self.bad_model:
            print('Large abs values (>= 1, <= -1) detected in lag_1_autocor')
            print('They are set to 0.99')
            self.lag_1_autocor[self.lag_1_autocor >= 1.0] = 0.99
            self.lag_1_autocor[self.lag_1_autocor <= -1.0] = -0.99
        #
        if len(self.clim_covar.shape) >= 3:
            raise ValueError(f"Bad shp {self.clim_covar.shape = }.")
        #
        if len(self.errcov_analysis.shape) >= 3:
            err_msg = "Bad shape errcov_analysis; "
            err_msg += f"{self.errcov_analysis.shape = }"
            raise ValueError(err_msg)
        print(f"{self.clim_covar.shape = }")
        print(f"{self.errcov_analysis.shape = }")
        if self.clim_covar.shape != self.errcov_analysis.shape:
            raise ValueError("Inconsistent shape detected!")
        if self.errcov_analysis.shape != self.lag_1_autocor.shape:
            raise ValueError("Inconsistent shape detected!")

    def _check_sq_matrix(self, arr: np.ndarray):
        check1 = arr.shape[0] == arr.shape[1]
        check2 = len(arr.shape) == 2
        return check1 and check2

    def compute_forecast_local(self, full_errcov_out: bool = True):
        """
        Compute AR1 forecast and estimate uncertainties

        Speed:
        https://stackoverflow.com/questions/44388358/python-numpy-matrix-multiplication-with-one-diagonal-matrix

        This version uses scalar multiplication, less use of np.diag, and should
        be faster.

        Parameters
        ----------
        analysis: numpy.ndarray
            1D vector of independent variables for t
        errcov_analysis: numpy.ndarray
            2D errcov for analysis
        lag_1_autocov: numpy.ndarray
            1D vector of lag correlation
        clim_mean: numpy.ndarray
            1D climatological mean for analysis
        clim_covar: numpy.ndarray
            climatology_variance: 1D (local) climatological variance
            for analysis

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
        self.weights = np.diag(self.lag_1_autocor)
        diff_with_clim_mean = self.analysis - self.clim_mean
        self.forecast = self.lag_1_autocor * diff_with_clim_mean
        self.forecast += self.clim_mean
        #
        print("Computing uncertainties")
        # Compute error covariances associated with AR1 epsilon
        # that are associated with climatological variance
        #
        # This only works 1D/local prediction
        # clim_covar is a vector of climatological variances
        # and errcov_analysis is a vector of local uncertainties
        # No off-diagonal/correlated terms
        # Works fast but is crude, and don't use all possible info
        #
        # Equation:
        # sigma_sq_eps = (1 - PHI * PHI) * clim_covar
        # sigma_sq_analysis = PHI * PHI * errcov_analysis
        autocorr_sq = np.square(self.lag_1_autocor)
        climvar_mult = np.ones_like(autocorr_sq) - autocorr_sq
        sigma_sq_eps = climvar_mult * self.clim_covar
        sigma_sq_analysis = autocorr_sq * self.errcov_analysis
        self.errcov = sigma_sq_eps + sigma_sq_analysis
        self.bad_model = False
        if full_errcov_out:
            self.errcov = np.diag(self.errcov)


class Autoregressive1ForecastVector:
    """
    Class to compute AR1 forecast

    The full vector AR model is not ready yet.

    Weights can be provided or computed if lag-0, lag-1 covariances are
    provided.

    Latter is possibly unstable; typical spatial lag-0 covariances
    are computed using variogram techniques, and those don't methods may not
    work with lag-1 and there is no sure way to ensure
    <x(t-1), x(t)> @ <X(t), X(t)>**-1 is stable.

    Weights are only stable if its spectral radius is less than 1. Otherwise
    weights violate weak temporal stationarity assumption.

    Instead usually weights are estimated seperately using regression taking
    advantage of seemingly (un)related regression (SUR) method with additional
    use of regularisation to deal with weight stability, multicollinearity, and
    insufficient sample sizes/too many covariates.

    Regardless, this method is much more expensive. It is in theory more
    complete and deals with spatially correlated component of the analysis.

    Compute Lag-1 autoregressive forecast as a prior for Kalman filter.
    """

    def __init__(
        self,
        analysis: np.ndarray,
        errcov_analysis: np.ndarray,
        lag_1_autocov: np.ndarray | sp.sparse.sparray,
        clim_mean: np.ndarray,
        clim_covar: np.ndarray,
        lag_1_autocov_is_wgts: bool = False,
    ):
        """
        __init__ for Autoregressive1Forecast class

        Parameters
        ----------
        analysis: numpy.ndarray
            1D vector of analysis
            (such as Kriging and Kalman filter outputs)
            for t=t
            dtype `float`, shape `(N, )`
        errcov_analysis: numpy.ndarray
            2D error covariance for analysis
            dtype `float`, shape `(N, N)`
        lag_1_autocov: numpy.ndarray | scipy.sparse.sparray
            2D full lag-1 autocovariance or weights
            dtype `float`, shape `(N, N)`
        clim_mean: numpy.ndarray
            1D vector of climatological mean
            Shape should be same as analysis
            dtype `float`, shape `(N, )`
        clim_covar: numpy.ndarray
            Climatological covariance,
            Can be 1D or 2D,
            The shape of this should either be n x n (like ellipse covariance)
            or n (vector of variance at each grid point)
            dtype `float`, shape `(N, N)`
        lag_1_autocov_is_wgts: bool
            Default False
            Bool for lag_1_autocov_is_wgts is the multivariate/VAR weight.
            If True, lag_1_autocov will be used as weights directly, but
            bool predict_local takes prescendence if that is set True.
            Nevertheless, for diagonal lag_1_autocov (autocorrelation /
            predict_local == True case), the autocorrelations ARE the weights.
            For the general multivariate/VAR case, such prescribed weights are
            usually estimated seperately using regression analysis. Such
            estimate is preferred approach because there are ways to make the
            weights stable and they are storable in sparse format reducing
            memory footprint. Any attempt to solve the weights on the fly using
            some two covariances are more expensive and are possibly unstable
            because there is no prior way to ensure the weights statisfy the
            requirements that the spectral radius of the weights must be < 1.
        """
        #
        self.analysis = analysis
        self.errcov_analysis = errcov_analysis
        self.lag_1_autocov = lag_1_autocov
        self.clim_mean = clim_mean
        self.clim_covar = clim_covar
        #
        # Names of methods to compute prrediction
        self.compute_forecast = self.compute_forecast_vector
        self.predict = self.compute_forecast_vector  # sklearn standard
        #
        self.lag_1_autocov_is_wgts = lag_1_autocov_is_wgts
        print(f"{self.lag_1_autocov_is_wgts = }")
        if lag_1_autocov_is_wgts:
            wgt_advisory = "lag_1_autocov will be treated as weights."
            check_if_wgts_are_sp_sparse = isinstance(
                self.lag_1_autocov,
                sp.sparse.sparray,
            )
            print(f"{check_if_wgts_are_sp_sparse = }")
            if not check_if_wgts_are_sp_sparse:
                wgt_advisory += "\nlag_1_autocov is not a scipy sparse "
                wgt_advisory += "matrix; computation will slow and memory "
                wgt_advisory += "intensive; Lasso weights are sparse, but "
                wgt_advisory += "OLS (unstable) and Ridge weights are dense."
        else:
            wgt_advisory = "Weights will be solved from lag_1_autocov "
            wgt_advisory += "and clim_covar; this is often unstable "
            wgt_advisory += "and is not recommended."
        print(wgt_advisory)
        print(f"{self.clim_covar.shape = }")
        print(f"{self.errcov_analysis.shape = }")
        print(f"{self.lag_1_autocov.shape = }")
    #
        self._check_args()
        self.bad_model = None

    def _check_args(self):  # noqa: C901
        """Check attributes set on init"""
        # 1D variable check
        if len(self.analysis.shape) != 1:
            raise ValueError("independent_var_t should be 1D")
        if len(self.clim_mean.shape) != 1:
            raise ValueError("climatology_mean should be 1D")
        if self.analysis.shape[0] != self.clim_mean.shape[0]:
            raise ValueError("analysis shape inconsistent with clim_mean")
        #
        if not self._check_sq_matrix(self.lag_1_autocov):
            raise ValueError(f"Bad shp {self.lag_1_autocov.shape = }.")
        #
        if not self._check_sq_matrix(self.clim_covar):
            raise ValueError(f"Bad shp {self.clim_covar.shape = }.")
        #
        if not self._check_sq_matrix(self.errcov_analysis):
            err_msg = "Bad shape errcov_analysis "
            err_msg += f"{self.errcov_analysis.shape}."
            raise ValueError(err_msg)
        print(f"{self.clim_covar.shape = }")
        print(f"{self.errcov_analysis.shape = }")
        print(f"{self.lag_1_autocov.shape = }")
        if self.clim_covar.shape != self.errcov_analysis.shape:
            raise ValueError("Inconsistent shape detected!")
        if self.errcov_analysis.shape != self.lag_1_autocov.shape:
            raise ValueError("Inconsistent shape detected!")

    def _check_sq_matrix(self, arr: np.ndarray):
        check1 = arr.shape[0] == arr.shape[1]
        check2 = len(arr.shape) == 2
        return check1 and check2

    def compute_forecast_vector(
        self,
        check_wgt_stability: bool = True,
        full_errcov_out: bool = True,
        stability_perturbation: float | None = None,
    ):
        """
        Compute VAR forecast using weights or covariances
        that are part of class.

        Parameters
        ----------
        analysis: numpy.ndarray
            1D vector of independent variables for t
        errcov_analysis: numpy.ndarray
            2D errcov for analysis
        lag_1_autocov: numpy.ndarray
            2D matrix of autocovariance
        clim_mean: numpy.ndarray
            1D climatological mean for analysis
        clim_covar: numpy.ndarray
            climatology_variance: 2D climatological covariance for analysis
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
        if stability_perturbation is not None:
            ccp = stability_perturbation + self.clim_covar
        else:
            ccp = self.clim_covar
        #
        if self.lag_1_autocov_is_wgts:
            print("Using lag_1_autocov as weights")
            self.weights = self.lag_1_autocov
        else:
            print("Computing weights")
            # W = <x(t), x(t-1)> @ <x(t), x(t)>**-1
            # W @ <x(t), x(t)> = <x(t), x(t-1)>
            # <x(t), x(t)> @ W.T = <x(t), x(t-1)>
            # (note: <x(t), x(t-1)> and <x(t), x(t)> are symmetric)
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
        # It can be shown that the below produces the same
        # result as if applying spatiotemporal Kriging for t+1
        # The weight for spatiotemporal Kriging is:
        # W_Krige(t, t-1) = C(t, t-1) D.T @ INV(C(t, t) + E)
        # That is the same as weights for AR1 multiplying on top of
        # Kriging (t,t):
        # W_AR = C(t, t-1) @ INV(C(t, t))
        # W_Krige(t,t) = C @ D.T @ INV(C(t, t) + E)
        # W_AR @ W_Krige(t,t)
        # = C(t, t-1) @ INV(C) @ C @ D.T @ INV(C(t, t) + E)
        # = C(t, t-1) @ D.T @ INV(C(t, t) + E)
        # = W_Krige(t, t-1)
        print("Computing forecast")
        diff_with_clim_mean = self.analysis - self.clim_mean
        self.forecast = self.weights @ diff_with_clim_mean
        self.forecast += self.clim_mean
        #
        # https://math.stackexchange.com/questions/5004102/covariance-of-a-multivariate-autoregression
        # I didn't cheat
        # I got same equation independently using good ole pen and paper!
        #
        # If weights fail stability check, it implies expected value
        # of anomalies can grow in time.
        # Testing shows, one also gets wrong uncertainty
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
        # It can be shown that if errcov_analysis are Kriging covariance that
        # sigma_sq_eps + sigma_sq_analysis == spatiotemporal Kriging for t+1
        sigma_sq_analysis = self.weights @ self.errcov_analysis @ self.weights.T
        self.errcov = sigma_sq_eps + sigma_sq_analysis
        if not full_errcov_out:
            self.errcov = np.diag(self.errcov)

    def var_stability_check(self, warn_instead: bool = True):
        """
        A multivariate stationary AR model is only stable if and only
        if all eigvals of the weights have absolute values less than 1
        https://kevinkotze.github.io/ts-7-slide/#10
        which is from
        https://www.jstor.org/stable/j.ctv14jx6sm
        Chapter 10 Proposition 10.1
        """
        print("Checking weight stability")
        # Don't use sp.linalg.eigh or np.linalg.eigvalsh etc.,
        # weights are not a symmetric matrix!
        if isinstance(self.weights, sp.sparse.sparray):

            def eigval_solver(arr):
                """Compute largest and smallest real eigval for sparse arr"""
                largest = sp.sparse.linalg.eigs(
                    arr, k=1, which="LR", return_eigenvectors=False
                )[0]
                smallest = sp.sparse.linalg.eigs(
                    arr, k=1, which="SR", return_eigenvectors=False
                )[0]
                return np.array([largest, smallest])
        elif isinstance(self.weights, np.ndarray):
            eigval_solver = np.linalg.eigvals
        else:
            err_msg = f"Unknown type self.weights: {type(self.weights)}"
            raise ValueError(err_msg)
        eigvals = eigval_solver(self.weights)
        smallest_eigval = np.min(np.real(eigvals))
        largest_eigval = np.max(np.real(eigvals))
        print(f"{smallest_eigval = }")
        print(f"{largest_eigval = }")
        self.bad_model = np.logical_or(
            smallest_eigval < -1.0,
            largest_eigval > 1.0,
        )
        if self.bad_model:
            errmsg = "Eigenvalues of estimated weights have values > 1; "
            errmsg += "autocovariance "
            errmsg += "covariance do not satisfy stationarity requirements, "
            errmsg == "and error covariances are incorrect."
            if warn_instead:
                warnings.warn(errmsg, UserWarning)
            else:
                raise ValueError(errmsg)
