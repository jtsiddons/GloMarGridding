# Copyright 2025 National Oceanography Centre
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Spline Methods"""

from abc import abstractmethod
from collections.abc import Callable
from math import exp, factorial
from types import NoneType
from typing import Protocol

import numpy as np
import scipy as sp
from scipy.optimize import OptimizeResult, minimize_scalar

from sklearn.metrics.pairwise import haversine_distances, euclidean_distances


def _haversine_degrees(X1: np.ndarray, X2: np.ndarray | None = None):
    """Compute haversine distances from degree inputs, lat, lon."""
    if X2 is None:
        return haversine_distances(np.radians(X1))
    return haversine_distances(np.radians(X1), np.radians(X2))


class _Interpolator(Protocol):
    """
    Protocol for Spline-based interpolators. Defines the structure required for
    the interpolation, including the required attributes. Primary use of this
    class is for type annotation and this should not be called directly.
    """

    X: np.ndarray
    y: np.ndarray
    K: np.ndarray
    n_t: int
    n_pts: int
    fit: Callable
    predict: Callable
    fitted: bool = False

    @property
    def trend_basis(self) -> np.ndarray: ...
    @property
    def error_cov(self) -> np.ndarray: ...

    def get_K(self, positions: np.ndarray | None) -> np.ndarray: ...  # noqa: N802

    def dist_func(self, positions: np.ndarray | None) -> np.ndarray: ...

    def kernel(self, distances: np.ndarray) -> np.ndarray: ...

    def get_trend_basis(self, positions: np.ndarray) -> np.ndarray: ...


class SplineResult:
    """
    Result class for a Spline, holds result attributes from the spline fitting.

    Parameters
    ----------
    lam : float
        The lambda value used in fitting, controls the smoothing.
    weights : numpy.ndarray
        The weights from fitting a Spline.
    trend_coefs : numpy.ndarray
        The trend coefficients from fitting a Spline.
    M_inv : numpy.ndarray
        The inverse of the fitting matrix, used to compute some statistics.
    y_fit : numpy.ndarray
        The fitted result of the Spline interpolation at the input coordinates.
    parent : Spline
        The Spline instance used to generate the result. Requires 'covariance',
        'trend_basis' and 'y' attributes to be set.
    compute_stats : bool
        Compute fitting statistics. Sets statistical attributes to the instance.
    """

    def __init__(
        self,
        lam: float,
        weights: np.ndarray,
        trend_coefs: np.ndarray,
        M_inv: np.ndarray,
        y_fit: np.ndarray,
        parent: _Interpolator,  # Should be a Spline
        compute_stats: bool = True,
    ) -> NoneType:
        self.lam = lam
        self.weights = weights
        self.trend_coefs = trend_coefs
        self.M_inv = M_inv
        self.y_fit = y_fit
        self.parent = parent

        if compute_stats:
            self.fit_stats()

        return None

    def fit_stats(self) -> NoneType:
        """
        Get statistics from a Spline fit. Requires the 'parent' attribute to
        be set and to be a Spline with 'covariance', 'trend_basis', and 'y'
        attributes to be set.

        Attributes
        ----------
        signal : float
            Signal of the fit.
        error : float
            Error.
        gcv : float
            Generalised Cross-Validation.
        msr : float
            Mean square error.
        var : float
            Variance.
        """
        n_pts = len(self.y_fit)
        residual = self.parent.y - self.y_fit
        sumsq = np.sum(np.power(residual, 2))

        H = (
            np.hstack([self.parent.K, self.parent.trend_basis])
            @ self.M_inv[:, :n_pts]
        )

        self.signal = np.trace(H)
        self.error = n_pts - self.signal
        self.gcv = (sumsq * n_pts) / (self.error**2)
        self.msr = sumsq / n_pts
        self.var = sumsq / self.error

        return None


class Spline(_Interpolator):
    """
    Generic class for Spline Interpolation and Smoothing approaches.

    **Do not use this default class.**

    Parameters
    ----------
    X : numpy.ndarray
        The training data positions. Expected shape `[num points, num coords]`.
        Must be unique positions.
    y : numpy.ndarray
        The training data values. Expected to have the same `num points` values.
    error_cov : numpy.ndarray | None
        An optional error covariance. If unset then the default error covariance
        matrix is used - in most cases this will be the identity matrix.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        error_cov: np.ndarray | NoneType = None,
    ) -> NoneType:
        if not hasattr(self, "method"):
            raise NotImplementedError("Do not use the generic class directly.")

        if len(X) != len(np.unique(X, axis=0)):
            raise ValueError("Positions must be unique")

        self.n_pts = len(y)
        if self.n_pts != len(X):
            raise ValueError(
                f"Mismatch between number of positions {len(X) = } "
                + f"and value {len(y) = }"
            )

        self.X = X
        self.y = y
        self.error_cov = error_cov
        self.K = self.get_K()

        return None

    @property
    def trend_basis(self) -> np.ndarray:
        """Trend Basis of Training Data"""
        return self.get_trend_basis(self.X)

    @property
    def default_error_cov(self) -> np.ndarray:
        """Default error covariance"""
        raise NotImplementedError(
            "Not implemented for the generic Spline class"
        )

    @property
    def error_cov(self) -> np.ndarray:
        """Error Covariance Matrix"""
        if not hasattr(self, "_error_cov"):
            self._error_cov = self.default_error_cov
        return self._error_cov

    @error_cov.setter
    def error_cov(self, value: np.ndarray | NoneType) -> NoneType:
        if value is not None and value.shape != (
            self.n_pts,
            self.n_pts,
        ):
            raise ValueError("Mismatch in size of error covariance")
        self._error_cov = self.error_cov if value is None else value

    @property
    def result(self) -> SplineResult:
        """Interpolation result, generated from 'fit'"""
        if not hasattr(self, "_result"):
            raise AttributeError(
                "'result' attribute has not been set, use 'fit' method."
            )
        return self._result

    @result.setter
    def result(self, value: SplineResult) -> NoneType:
        self._result = value

    @abstractmethod
    def dist_func(
        self,
        positions: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Distances between each pair of points.

        Parameters
        ----------
        positions : numpy.ndarray | None
            Optionally compute the distance between each position in this value
            to the training positions. Otherwise compute the distance between
            training positions.

        Returns
        -------
        dist : numpy.ndarray
            The distances.
        """
        raise NotImplementedError(
            "Not implemented for the generic Spline class."
        )

    def get_K(self, positions: np.ndarray | None = None) -> np.ndarray:  # noqa: N802
        """
        Get the spatial structure / covariance matrix.

        Parameters
        ----------
        positions : numpy.ndarray | None
            Compute the kernel between the training positions and the values
            provided. If not set this will return the value from the kernel
            between the training positions with themselves.

        Returns
        -------
        result : numpy.ndarray
        """
        return self.kernel(self.dist_func(positions=positions))

    @abstractmethod
    def get_trend_basis(self, positions: np.ndarray) -> np.ndarray:
        """Trend Basis"""
        raise NotImplementedError(
            "Not implemented for the generic Spline class."
        )

    @abstractmethod
    def kernel(self, distances: np.ndarray) -> np.ndarray:
        """Spline Kernel"""
        raise NotImplementedError(
            "Not implemented for the generic Spline class."
        )

    def _compute_result(self, covariance, w, trend_basis, c) -> np.ndarray:
        if self.n_t > 0:
            return covariance @ w + trend_basis @ c
        return covariance @ w

    def _get_weights(
        self,
        lam: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fit a spline model with an optional smoothing parameter."""
        K_reg = np.copy(self.K)
        if lam > 0:
            K_reg += lam * self.error_cov

        if self.n_t > 0:
            K_reg = np.block(
                [
                    [K_reg, self.trend_basis],
                    [self.trend_basis.T, np.zeros((self.n_t, self.n_t))],
                ]
            )
        # NOTE: require inverse for other calculations
        M_inv = sp.linalg.inv(K_reg, assume_a="sym")

        b = self.y
        if self.n_t > 0:
            b = np.concatenate([b, np.zeros(self.n_t)])

        sol = M_inv @ b
        # Weights
        weights = sol[: self.n_pts]
        # Trend Coefficients
        trend_coefs = sol[self.n_pts :]

        return weights, trend_coefs, M_inv

    def fit(
        self,
        lam: float | None = None,
        set_results: bool = True,
        compute_stats: bool = True,
        **kwargs,
    ) -> SplineResult:
        """
        Fit the Spline. With optional smoothing. Optionally save the results.

        Parameters
        ----------
        lam : float | None
            Optional smoothing parameter. If set to 0 the interpolation will be
            an **exact** interpolator. If unset, the value will be estimated
            using a GCV method.
        set_results : bool
            Assign the results to the `result` attribute and set the `fitted`
            attribute to True. This is generally advised.
        compute_stats : bool
            Add the statistics to the results attribute.
        **kwargs
            Keyword parameters to pass to minimize_scalar if estimating the
            smoothing parameter using the GCV approach. Note, if 'lam' is not
            None then keyword arguments will not be used.

        Returns
        -------
        result : SplineResult
            The result of fitting the Spline class using the training data used
            to initialise the Spline instance. This value is also set to the
            'result' attribute if `set_results` is True.
        """
        if lam is None:
            print("Estimating Lambda Using GCV")
            lam = self.estimate_lambda_gcv(**kwargs)

        weights, trend_coefs, M_inv = self._get_weights(lam=lam)

        y_fit = self._compute_result(
            self.K, weights, self.trend_basis, trend_coefs
        )

        result = SplineResult(
            lam=lam,
            weights=weights,
            trend_coefs=trend_coefs,
            M_inv=M_inv,
            y_fit=y_fit,
            parent=self,
            compute_stats=compute_stats,
        )

        if set_results:
            self.result = result
            self.fitted = True

        return result

    def get_stats(self) -> NoneType:
        """Get the statistics of the fitted result."""
        if not hasattr(self, "result"):
            raise AttributeError("'result' attribute not set, run 'fit' method")

        self.result.fit_stats()

    def _predict(
        self,
        X_test: np.ndarray,
        compute_se: bool = True,
        result: SplineResult | NoneType = None,
    ) -> tuple[np.ndarray, np.ndarray | NoneType]:
        """
        Predict for new positions.

        Predicts using the weights and trend coefficients from the set `result`
        attribute, or from a provided SplineResult instance.

        Parameters
        ----------
        X_test : numpy.ndarray
            New positions use to predict using the SplineResult. Must be unique
            positions.
        compute_se : bool
            Optionally compute standard error of the prediction.
        result : SplineResult | None
            Optional SplineResult instance to use for prediction (instead of the
            set `result` attribute). If not set, the `result` attribute must be
            set and the `fitted` attribute must be True.

        Returns
        -------
        prediction : numpy.ndarray
            The predicted values at the test positions.
        standard_error : numpy.ndarray | None
            Standard error of the fit if `compute_se` is True, otherwise None.
        """
        if result is None:
            if not self.fitted or not hasattr(self, "result"):
                raise ValueError(
                    "Must first fit the model if not providing a result."
                )

            result = self.result

        n_pts = len(X_test)
        if n_pts != len(np.unique(X_test, axis=0)):
            raise ValueError("Have duplicate positions")

        if not hasattr(result, "var"):
            print("Computing variance of the fit.")
            result.fit_stats()

        P = self.get_trend_basis(X_test)
        # Num grid pts, num obs pts
        D = self.dist_func(X_test)
        K = self.kernel(D)

        prediction = self._compute_result(
            K, result.weights, P, result.trend_coefs
        )

        if not compute_se:
            return prediction, None

        U = np.hstack([K, P]) if self.n_t > 0 else K
        Usolved = np.abs(np.sum((U @ result.M_inv) * U, axis=1))
        standard_error = np.sqrt(Usolved * result.var)

        return prediction, standard_error

    def estimate_lambda_gcv(
        self,
        **kwargs,
    ) -> float:
        """
        Use Generalised Cross-Validation to estimate an optimal smoothing
        parameter.

        Parameters
        ----------
        **kwargs
            Keyword arguments to pass to minimize_scalar.

        Returns
        -------
        lam : float
            The result of an optimisation for the smoothing parameter.
        """
        if "bounds" in kwargs:
            kwargs["bounds"] = np.log(kwargs["bounds"])

        def gcv_loss(log_lam: float) -> float:
            lam = exp(log_lam)
            result = self.fit(lam, set_results=False, compute_stats=True)

            return result.gcv

        result = minimize_scalar(gcv_loss, **kwargs)

        if not isinstance(result, OptimizeResult):
            raise TypeError(
                "result of estimation is not an instance of "
                + "'scipy.optimize.OptimizeResult'."
            )
        return np.exp(result.x)

    def predict(
        self,
        X_test: np.ndarray,
        compute_se: bool = True,
        result: SplineResult | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Predict for new positions.

        Predicts using the weights and trend coefficients from the set `result`
        attribute, or from a provided SplineResult instance.

        Parameters
        ----------
        X_test : numpy.ndarray
            New positions use to predict using the SplineResult. Must be unique
            positions.
        compute_se : bool
            Optionally compute standard error of the prediction.
        result : SplineResult | None
            Optional SplineResult instance to use for prediction (instead of the
            set `result` attribute). If not set, the `result` attribute must be
            set and the `fitted` attribute must be True.

        Returns
        -------
        prediction : numpy.ndarray
            The predicted values at the test positions.
        standard_error : numpy.ndarray | None
            Standard error of the fit if `compute_se` is True, otherwise None.
        """
        return self._predict(
            X_test,
            compute_se=compute_se,
            result=result,
        )


class ThinPlateSpline(Spline):
    r"""
    Thin Plate Spline Interpolation and Smoothing. This uses the standard kernel
    for thin-plate-spline interpolation:

    .. math::
        r^2 \times log(r).

    Where r is the Euclidean distance between a pair of positions. Note that
    this is undefined for 0 distance. The kernel is instead defined to be 0 at 0
    distance, which is the limit.

    The methodology use the Euclidean distance metric, applied directly to the
    input position values.

    The trend basis for this method is the positions and intercept. Used as
    additional predictors for the interpolation. Note, this method may not be
    appropriate on spherical geometry.

    The methodology follows [Hutchinson]_.

    Parameters
    ----------
    X : numpy.ndarray
        The training data positions. Expected shape `[num points, num coords]`.
        Must be unique positions. It is generally recommended that the input
        positions should be scaled. There are numerous ways to approach this
        scaling, such as by the range, or the mean and standard deviation.
        Typically this depends on the input data and domain. It is assumed that
        the positions have been scaled appropriately.
    y : numpy.ndarray
        The training data values. Expected to have the same `num points` values.
    error_cov : numpy.ndarray | None
        An optional error covariance. If unset then the default error covariance
        matrix is used - in most cases this will be the identity matrix.

    References
    ----------
    [Hutchinson]_
    """

    method: str = "thin_plate_spline"

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        error_cov: np.ndarray | NoneType = None,
    ) -> NoneType:
        self.n_t = X.shape[1] + 1
        super().__init__(X, y, error_cov)

    @property
    def default_error_cov(self) -> np.ndarray:
        """
        Default Error Covariance, for thin plate spline this is the identity
        matrix.
        """
        return np.eye(self.n_pts)

    def dist_func(
        self,
        positions: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Pairwise Euclidean Distances between each pair of points.

        Parameters
        ----------
        positions : numpy.ndarray | None
            Optionally compute the distance between each position in this value
            to the training positions. Otherwise compute the distance between
            training positions.

        Returns
        -------
        dist : numpy.ndarray
            The distances.
        """
        if positions is None:
            return euclidean_distances(self.X)
        return euclidean_distances(positions, self.X)

    def get_trend_basis(self, positions: np.ndarray) -> np.ndarray:
        """Linear trend basis for universal kriging: [1, x, y, ...].

        Parameters
        ----------
        positions : numpy.ndarray | None
            Compute the trend basis of these positions, otherwise the training
            positions will be used.

        Returns
        -------
        P : numpy.ndarray
            The trend-basis. An (n x m + 1) matrix of ones where n is the
            number of positions, and m is the number of coordinates.
        """
        n_pts = positions.shape[0]
        res = np.hstack((np.ones((n_pts, 1)), positions))
        return res

    def kernel(self, distances: np.ndarray) -> np.ndarray:
        r"""
        Thin Plate Spline RBF kernel. This is

        .. math::
           r^2 \times \log(r)

        or 0 if r == 0. Where r is the pairwise distance.

        Parameters
        ----------
        distances : numpy.ndarray
            The pairwise haversine distances.

        Returns
        -------
        numpy.ndarray
            The result of the ThinPlateSpline kernel.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(
                distances == 0, 0.0, np.power(distances, 2) * np.log(distances)
            )


class SphericalThinPlateSpline(Spline):
    r"""
    Spherical form of Thin Plate Spline Interpolation and Smoothing.
    Adapted from [Wahba_Sphere]_.

    This approach is specifically designed to operate using spherical geometry,
    such as the earth. The results will be continuous over the full domain
    (including at boundaries such as 0, 2pi).

    The Kernel used in this Spline interpolation is the solution to equation 3.3
    and table 1 from [Wahba_Sphere]_ for :math:`m=2` (:math:`q_2`).

    .. math::
        q_m(z) = \int_0^1 (1-h)^m (1-2hz+h^2)^{-1/2} dh.

    Where :math:`z` is the cosine of the Haversine distance between two points.

    To guarantee continuity of the result on spherical geometry the Haversine
    distance is used, and the trend basis is a vector of ones. This means that
    the positions are not used (directly) as predictors in this method (unlike
    the traditional thin-plate spline method). Consequently, this method is
    closer to ordinary kriging.

    The use of the Kernel can be overridden by specifying an input covariance
    matrix.

    Parameters
    ----------
    X : numpy.ndarray
        The training data positions, expected to be lat, lon in degrees. The
        input positions must be unique. Expected shape `[num points, 2]`.
    y : numpy.ndarray
        The training data values. Expected to have the same `num points` values.
    error_cov : numpy.ndarray | None
        An optional error covariance. If unset then the default error covariance
        matrix is used - in this case n * identity matrix, where n is the number
        of positions in `X`.

    References
    ----------
    [Wahba_Sphere]_
    """

    method: str = "spherical_thin_plate_spline"
    n_t: int = 1

    @property
    def default_error_cov(self) -> np.ndarray:
        """Default Error Covariance, n times identity matrix."""
        return self.n_pts * np.eye(self.n_pts)

    def dist_func(
        self,
        positions: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Pairwise Haversine Distances between each pair of points.

        Parameters
        ----------
        positions : numpy.ndarray | None
            Optionally compute the distance between each position in this value
            to the training positions. Otherwise compute the distance between
            training positions.

        Returns
        -------
        dist : numpy.ndarray
            The distances.
        """
        if positions is None:
            return _haversine_degrees(self.X)
        if positions.shape[1] != 2:
            raise ValueError(
                "Coordinates must be 2 dimensional lat and lon in degrees."
            )
        return _haversine_degrees(positions, self.X)

    def get_trend_basis(self, positions: np.ndarray) -> np.ndarray:
        """
        Trend basis for ordinary kriging: [1]. Note that this does not include
        the position components, since this could break continuity of the result
        over the sphere (e.g. at 0, 360 degrees boundary).

        Parameters
        ----------
        positions : numpy.ndarray | None
            Compute the trend basis of these positions, otherwise the training
            positions will be used.

        Returns
        -------
        P : numpy.ndarray
            The trend-basis. An (n x 1) matrix of ones where n is the number of
            positions.
        """
        n_pts = positions.shape[0]
        return np.ones((n_pts, 1))

    def kernel(self, distances: np.ndarray) -> np.ndarray:
        """
        Spherical Thin Plate Spline RBF kernel, this is q[2] from
        [Wahba_Sphere]_.

        Parameters
        ----------
        distances : numpy.ndarray
            The pairwise haversine distances.

        Returns
        -------
        numpy.ndarray
            The result of the SphericalThinPlateSpline kernel.
        """
        m = 2
        m2_1 = 2 * m - 1
        fac_m2_1 = factorial(m2_1)
        pi_frac = 1 / (2 * np.pi)
        frac = pi_frac / fac_m2_1
        var_val = frac * (m2_1 * 0.5 - 1)

        with np.errstate(divide="ignore", invalid="ignore"):
            q = np.where(distances == 0, 0.5, _q2(distances))

        q[np.isnan(q)] = 0.5

        res = frac * (m2_1 * q - 1)
        # INFO: Convert to correct form for standard error
        res -= var_val
        res[np.isclose(res, 0.0)] = 0.0
        return res


def _zwca(distances: np.ndarray) -> tuple[np.ndarray, ...]:
    """Get the z, W, C, and A components for q2"""
    z = np.cos(distances)
    W = (1 - z) / 2
    sqrtW = np.sqrt(W)
    C = 2 * sqrtW
    A = np.log(1 + 1 / sqrtW)
    return z, W, C, A


def _q2(distances: np.ndarray) -> np.ndarray:
    """R with q2 from [Wahba_Sphere]_"""
    _, W, C, A = _zwca(distances)
    return (A * (12 * W**2 - 4 * W) - 6 * C * W + 6 * W + 1) / 2
