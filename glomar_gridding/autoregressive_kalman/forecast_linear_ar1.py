"""
Conduct local AR1 forecast of uncertain data
output new uncertainties

for the purpose of docstring
N: number of analysis points, covariates/features
"""

import numpy as np
import scipy as sp
import warnings

from glomar_gridding.covariance_tools import explained_variance_clip
from glomar_gridding.utils import check_sq_matrix


class Autoregressive1ForecastUncorr:
    """
    Class to compute AR1 forecast

    Local/uncorrelated only: Lag-1 correlations are the
    weights and are in 1-D vector. All predictions are based on
    local autocorrelation only

    Use Autoregressive1Forecast for full vector case.

    Compute Lag-1 autoregressive forecast as a prior for Kalman filter.

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
        """__init__ for Autoregressive1Forecast class"""
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
        self.weights = sp.sparse.csc_array(np.diag(self.lag_1_autocor))
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
        # Names of methods to compute prediction
        self.compute_forecast = self.compute_forecast_local
        self.predict = self.compute_forecast_local  # sklearn standard
        #
        # Use only diagonal if 2D matrices are provided for
        # clim_covar and errcov_analysis
        if len(self.clim_covar.shape) == 2:
            if not check_sq_matrix(self.clim_covar):
                raise ValueError(f"Not square {self.clim_covar.shape = }.")
            self.clim_covar = np.diag(self.clim_covar)
        if len(self.errcov_analysis.shape) == 2:
            if not check_sq_matrix(self.errcov_analysis):
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
            print("Large abs values (>= 1, <= -1) detected in lag_1_autocor")
            print("They are set to 0.99")
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

    def compute_forecast_local(self, full_errcov_out: bool = True):
        """
        Compute AR1 forecast and estimate uncertainties

        Speed:
        https://stackoverflow.com/questions/44388358/python-numpy-matrix-multiplication-with-one-diagonal-matrix

        This version uses scalar multiplication, less use of np.diag, and should
        be faster.

        Parameters
        ----------
        full_errcov_out: bool
            If True, new attribute errcov will be a diagonal matrix

        Attributes
        ----------
        forecast: numpy.ndarray
            AR1 forecast
        errcov: numpy.ndarray
            The error covariance for the forecast
        """
        print("Computing forecast")
        #
        # The simple local case:
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
    Class to compute AR1 forecast in full vectorized form

    Weights can be provided. It can be computed from non-lagged and
    lag-1 covariances, using classmethod `from_autocov`.

    Latter is possibly unstable. Typical spatial 0-lag covariances
    are computed using variogram techniques, and they may not
    work with lag-1 and there is no sure way to ensure
    <x(t-1), x(t)> @ <X(t), X(t)>**-1 is stable. GloMarGridding has no
    functionality to compute lagged autocovariances.

    Autoregressive1ForecastUncorr is a special case of this class with
    the weights being the lag-1 autocorrelation.

    Weights are only stable if its spectral radius is less than 1. Otherwise
    weights violate weak temporal stationarity assumption.

    Instead usually weights are estimated separately using regression taking
    advantage of seemingly (un)related regression (SUR) method with additional
    use of regularization (like Lasso) to deal with weight stability,
    multicollinearity, and insufficient sample sizes/too many covariates.
    Weights estimated this way often meets spectral radius < 1 requirement.

    Regardless, this method is much more expensive. It is in theory more
    complete and deals with spatially correlated component of the analysis.

    Compute Lag-1 autoregressive forecast as a prior for Kalman filter.

    Other notes:

    (1) If `analysis` is a Kriging analysis, results here produces the same
    result as if applying spatiotemporal Kriging for t+1

    The weight for spatiotemporal Kriging is:
    W_Krige(t, t-1) = C(t, t-1) D.T @ INV(C(t, t) + E)

    That is the same as weights for AR1 multiplying on top of Kriging (t,t):
    W_AR = C(t, t-1) @ INV(C(t, t))
    W_Krige(t,t) = C @ D.T @ INV(C(t, t) + E)

    W_AR @ W_Krige(t,t)
    = C(t, t-1) @ INV(C) @ C @ D.T @ INV(C(t, t) + E)
    = C(t, t-1) @ D.T @ INV(C(t, t) + E)
    = W_Krige(t, t-1)

    Uncertainty: It can be shown that if errcov_analysis are Kriging covariance
    that
    sigma_sq_eps + sigma_sq_analysis == spatiotemporal Kriging for t+1

    In practice, `analysis` will be computed recursively using Kalman filter.

    (2) Uncertainty of multivariate autoregression:
    https://math.stackexchange.com/questions/5004102/covariance-of-a-multivariate-autoregression

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
    weights: numpy.ndarray | scipy.sparse.sparray
        2D weights
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
    estimated_ar1_errcov: numpy.ndarray | None
        A precomputed estimate for :math:`C - W C W^{T}`
        Defaults to None in which that is computed from
        `weights` and `clim_covar`
    """

    def __init__(
        self,
        analysis: np.ndarray,
        errcov_analysis: np.ndarray,
        weights: np.ndarray | sp.sparse.sparray,
        clim_mean: np.ndarray,
        clim_covar: np.ndarray,
        estimated_ar1_errcov: np.ndarray | None = None,
    ):
        """__init__ for Autoregressive1Forecast class"""
        #
        self.analysis = analysis
        self.errcov_analysis = errcov_analysis
        self.weights = weights
        self.clim_mean = clim_mean
        self.clim_covar = clim_covar
        if not (
            isinstance(estimated_ar1_errcov, np.ndarray)
            or (estimated_ar1_errcov is None)
        ):
            err_msg = "Unknown type/value for estimated_ar1_errcov; "
            err_msg += f"{estimated_ar1_errcov = }."
            raise ValueError(err_msg)
        self.estimated_ar1_errcov = estimated_ar1_errcov
        if self.estimated_ar1_errcov is None:
            err_msg = "estimated_ar1_errcov is None; module can estimate "
            err_msg += "VAR errcov on the fly, but it is possibly unstable "
            err_msg += "due to (1) insufficient number of obs from "
            err_msg += "sample clim_covar or (2) inconsistencies between "
            err_msg += "variogram-estimated clim_cov and regularised "
            err_msg += "regression weights. Estimated errcov may not be P(S)D "
            err_msg += "(needs clipping). Stability is much higher with errcov "
            err_msg += "being P(S)D if estimated_ar1_errcov is pre-estimated "
            err_msg += "using regression residues or out-of-sample forecast "
            err_msg += "errors."
            warnings.warn(
                err_msg,
                UserWarning,
            )
        #
        # Names of methods to compute prediction
        self.compute_forecast = self.compute_forecast_vector
        self.predict = self.compute_forecast_vector  # sklearn standard
        #
        check_if_wgts_are_sp_sparse = isinstance(
            self.weights,
            sp.sparse.sparray,
        )
        print(f"{check_if_wgts_are_sp_sparse = }")
        if not check_if_wgts_are_sp_sparse:
            wgt_advisory = "weight is not a scipy sparse "
            wgt_advisory += "matrix; computation will slow and memory "
            wgt_advisory += "intensive; Lasso weights and lag1 autocorrelation "
            wgt_advisory += "are sparse, but OLS (unstable) and Ridge weights "
            wgt_advisory += "are dense."
            print(wgt_advisory)
        print(f"{self.clim_covar.shape = }")
        print(f"{self.errcov_analysis.shape = }")
        print(f"{self.weights.shape = }")
        #
        self._check_args()
        self.bad_model = None

    @classmethod
    def from_autocov(
        cls,
        analysis: np.ndarray,
        errcov_analysis: np.ndarray,
        lag_1_autocov: np.ndarray,
        clim_mean: np.ndarray,
        clim_covar: np.ndarray,
        stability_perturbation: float | None = None,
        estimated_ar1_errcov: np.ndarray | None = None,
    ):
        """
        With provided lag_1_autocov (lag-1 autocovariance),
        estimate weights using W = lag1_autocov @ inv(clim_covar)

        W = <x(t), x(t-1)> @ <x(t), x(t)>**-1
        W @ <x(t), x(t)> = <x(t), x(t-1)>
        <x(t), x(t)> @ W.T = <x(t), x(t-1)>  <<< computed using np.linalg.solve

        Both <x(t), x(t)> and <x(t), x(t-1)> are symmetric

        This approach is often unstable as lag1_autocov and clim_covar
        are estimated separately, and there is no guarantee that the computed
        weights satisfy weak stationary requirement that the spectral radius
        of weights have to be less than 1.

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
        lag_1_autocov: numpy.ndarray
            2D lag-1 autocovariance
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
        stability_perturbation: float | None
            A diagonal perturbation to be added clim_covar for
            the purpose of computing the weights. This may avoid
            problems that clim_covar being (close to) invertible.
        """
        wgt_advisory = "Weights will be solved from lag1_autocov "
        wgt_advisory += "and clim_covar; this is often unstable "
        wgt_advisory += "and is not recommended."
        print(wgt_advisory)
        #
        if stability_perturbation is not None:
            new_diag = np.diag(clim_covar) + stability_perturbation
            ccp = clim_covar.copy()
            np.fill_diagonal(ccp, new_diag)
        else:
            ccp = clim_covar
        print("Computing weights")
        weights = np.linalg.solve(
            ccp,
            lag_1_autocov,
        ).T
        return cls(
            analysis,
            errcov_analysis,
            weights,
            clim_mean,
            clim_covar,
            estimated_ar1_errcov=estimated_ar1_errcov,
        )

    def _check_args(self):
        """Check attributes set on init"""
        # 1D variable check
        if len(self.analysis.shape) != 1:
            raise ValueError("independent_var_t should be 1D")
        if len(self.clim_mean.shape) != 1:
            raise ValueError("climatology_mean should be 1D")
        if self.analysis.shape[0] != self.clim_mean.shape[0]:
            raise ValueError("analysis shape inconsistent with clim_mean")
        #
        if not check_sq_matrix(self.weights):
            raise ValueError(f"Bad shp {self.weights.shape = }.")
        #
        if not check_sq_matrix(self.clim_covar):
            raise ValueError(f"Bad shp {self.clim_covar.shape = }.")
        #
        if not check_sq_matrix(self.errcov_analysis):
            err_msg = "Bad shape errcov_analysis "
            err_msg += f"{self.errcov_analysis.shape}."
            raise ValueError(err_msg)
        if self.clim_covar.shape != self.errcov_analysis.shape:
            raise ValueError("Inconsistent shape detected!")
        if self.errcov_analysis.shape != self.weights.shape:
            raise ValueError("Inconsistent shape detected!")

    def compute_forecast_vector(
        self,
        check_wgt_stability: bool = True,
        full_errcov_out: bool = True,
        check_errcov_psd: bool = True,
        **clip_kwargs,
    ):
        """
        Compute VAR forecast using weights or covariances
        that are part of class.

        Parameters
        ----------
        check_wgt_stability: bool
            Check weights (in extension for the auto and climatological
            covariances) are stable; see var_stability_check
        full_errcov_out: bool
            Return full error covariance if True
            Defaults to True
        check_errcov_psd: bool
            Check if error covariance is positive semi-definite, fix if needed
        clip_kwargs: dict | None
            kwargs for check_and_fix_psd_errcov_by_clipping

        Attributes
        ----------
        forecast: numpy.ndarray
            AR1 forecast
        errcov: numpy.ndarray
            The error covariance for the forecast
        bad_model: bool
            Boolean advisory that the weights being unchecked
        """
        print(f"{self.weights = }")
        if check_wgt_stability:
            self.var_stability_check()
        else:
            warn_msg = "check_wgt_stability is disabled; "
            warn_msg += "check is recommended as VAR may be unstable. "
            warn_msg += "As precaution, self.bad_model defaults to True."
            warnings.warn(warn_msg, UserWarning)
            self.bad_model = True
        #
        print("Computing forecast")
        diff_with_clim_mean = self.analysis - self.clim_mean
        self.forecast = self.weights @ diff_with_clim_mean
        self.forecast += self.clim_mean
        #
        print("Computing uncertainties")
        if self.estimated_ar1_errcov is None:
            print("Computing C - W C W.T")
            sigma_sq_eps = (
                self.clim_covar
                - self.weights @ self.clim_covar @ self.weights.T
            )
        else:
            print("Using pre-computed C - W C W.T")
            print(f"{self.estimated_ar1_errcov = }")
            sigma_sq_eps = self.estimated_ar1_errcov
        #
        # Warning:
        # Error covariances may not be (semi-)positive definite
        sigma_sq_analysis = self.weights @ self.errcov_analysis @ self.weights.T
        print(f"{sigma_sq_eps = }")
        print(f"{sigma_sq_analysis = }")
        self.errcov = sigma_sq_eps + sigma_sq_analysis
        #
        if check_errcov_psd:
            self.check_and_fix_psd_errcov_by_clipping(**clip_kwargs)
        #
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

        If weights fail stability check, it implies expected value
        of anomalies can grow in time.
        Testing shows, one also gets wrong uncertainty
        estimates, like off diagonal term being (considerably) larger than
        diagonal terms; that's an invalid (error) covariance matrix, saying
        correlation > 1 which is impossible!

        Parameters
        ----------
        warn_instead: bool
            By default, this method only generates a warning attribute
            If this is set to True, method will raise exception

        Attributes
        ----------
        bad_model: bool
            Boolean advisory that the weights being not weakly stationary
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
        self.bad_model = (smallest_eigval < -1.0) or (largest_eigval > 1.0)
        if self.bad_model:
            errmsg = "Eigenvalues of estimated weights have values > 1; "
            errmsg += "autocovariance "
            errmsg += "covariance do not satisfy stationarity requirements, "
            errmsg == "and error covariances are incorrect."
            if warn_instead:
                warnings.warn(errmsg, UserWarning)
            else:
                raise ValueError(errmsg)

    def check_and_fix_psd_errcov_by_clipping(
        self,
        target_variance_fraction=0.95,
    ):
        """
        Check if self.errcov positive semi-definite
        If not, use explained_variance_clip to patch it.
        """
        if not hasattr(self, "errcov"):
            raise ValueError("errcov has not been estimated yet!")
        print("Checking validity of error covariance")
        errcov_eigvals = np.linalg.eigvalsh(self.errcov)
        if np.min(errcov_eigvals) < 0:
            self.errcov = explained_variance_clip(
                self.errcov,
                target_variance_fraction=target_variance_fraction,
            )
        else:
            print("Error covariance clipping is unnecessary.")
