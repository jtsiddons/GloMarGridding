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

from abc import ABC, abstractmethod
from math import exp, factorial
from types import NoneType
from typing import Literal

import numpy as np
import scipy as sp
from scipy.optimize import OptimizeResult, minimize_scalar
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import haversine_distances

from glomar_gridding.variogram import Variogram, variogram_to_covariance


class SplineResult:
    """
    Result class for a Spline, holds result attributes from the spline fitting.

    Parameters
    ----------
    lam : float
        The lambda value used in fitting, controls the smoothing.
    weights : numpy.ndarray
    trend_coefs : numpy.ndarray
    M_inv : numpy.ndarray
    y_fit : numpy.ndarray
        The fitted result of the Spline interpolation at the input coordinates.
    parent : Spline
        The Spline instance used to generate the result.
    compute_stats : bool
        Compute fitting statistics
    """

    def __init__(
        self,
        lam: float,
        weights: np.ndarray,
        trend_coefs: np.ndarray,
        M_inv: np.ndarray,
        y_fit: np.ndarray,
        parent,  # Should be a Spline
        compute_stats: bool = True,
    ) -> NoneType:
        if not hasattr(parent, "_spline"):
            raise TypeError("parent must be a Spline")

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
        """Get statistics from the fit"""
        n_pts = len(self.y_fit)
        residual = self.parent.y - self.y_fit
        sumsq = np.sum(np.power(residual, 2))

        H = (
            np.hstack([self.parent.covariance, self.parent.trend_basis])
            @ self.M_inv[:, :n_pts]
        )

        signal = np.trace(H)
        error = n_pts - signal
        gcv = (sumsq * n_pts) / (error**2)
        msr = sumsq / n_pts
        var = sumsq / error

        self.signal = signal
        self.error = error
        self.gcv = gcv
        self.msr = msr
        self.var = var

        return None


class Spline(ABC):
    """
    Generic class for Spline Interpolatation and Smoothing approaches.

    Do not use this default class.

    Parameters
    ----------
    X : numpy.ndarray
        The training data positions.
    y : numpy.ndarray
        The training data values.
    covariance : numpy.ndarray | None
        Optional covariance matrix, covariance between observations positions.
        If not supplied then the Spline's kernel will be used (if Implemented)
    error_cov : numpy.ndarray | None
        An optional error covariance. If unset then the default error covariance
        matrix is used - in most cases this will be the identity matrix.
    """

    _spline = True
    fitted = False

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        covariance: np.ndarray | NoneType = None,
        error_cov: np.ndarray | NoneType = None,
    ) -> NoneType:
        if not hasattr(self, "method"):
            raise NotImplementedError("Do not use the generic class directly.")

        self.X = X
        self.y = y
        self.n_pts = len(self.y)
        self.trend_basis, self.n_t = self.get_trend_basis(self.X)

        if covariance is None:
            distances = self.dist_func(self.X)
            self.covariance = self.kernel(distances)
            del distances
        elif covariance.shape == (self.n_pts, self.n_pts):
            self.covariance = covariance
        else:
            msg = "Covariance, observations size mismatch. "
            msg += f"Got {covariance.shape = }, {len(y) = }"
            raise ValueError(msg)

        if error_cov is not None and error_cov.shape != (
            self.n_pts,
            self.n_pts,
        ):
            raise ValueError("Mismatch in size of error covariance")

        self.error_cov = (
            error_cov if error_cov is not None else self.default_error_cov
        )

        return None

    def clear(self) -> NoneType:
        """
        Clear the set variables.

        Clears the "covariance", "trend_basis", "result". Unsets "fitted".
        """
        if hasattr(self, "result"):
            delattr(self, "result")
        self.fitted = False
        return None

    @property
    def default_error_cov(self) -> np.ndarray:
        """Default error covariance"""
        raise NotImplementedError(
            "Not implemented for the generic Spline class"
        )

    def _compute_result(self, covariance, w, trend_basis, c) -> np.ndarray:
        if self.n_t > 0:
            return covariance @ w + trend_basis @ c
        return covariance @ w

    @abstractmethod
    def dist_func(
        self,
        positions: np.ndarray,
        X2: np.ndarray | NoneType = None,
    ) -> np.ndarray:
        """Distances between each pair of points"""
        raise NotImplementedError(
            "Not implemented for the generic Spline class"
        )

    @abstractmethod
    def get_trend_basis(self, positions: np.ndarray) -> tuple[np.ndarray, int]:
        """Trend Basis"""
        raise NotImplementedError(
            "Not implemented for the generic Spline class"
        )

    @abstractmethod
    def kernel(self, distances: np.ndarray) -> np.ndarray:
        """Spline Kernel"""
        raise NotImplementedError(
            "Not implemented for the generic Spline class"
        )

    def _get_weights(
        self,
        lam: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fit a spline model with an optional smoothing parameter"""
        if not hasattr(self, "trend_basis"):
            self.trend_basis, self.n_t = self.get_trend_basis(self.X)
        if not hasattr(self, "covariance"):
            distances = self.dist_func(self.X)
            self.covariance = self.kernel(distances)
            del distances

        K_reg = np.copy(self.covariance)
        if lam > 0:
            K_reg += lam * self.error_cov

        if self.n_t == 0:
            M = K_reg
        else:
            M = np.block(
                [
                    [K_reg, self.trend_basis],
                    [self.trend_basis.T, np.zeros((self.n_t, self.n_t))],
                ]
            )
        M_inv = sp.linalg.inv(M, assume_a="sym")

        if self.n_t == 0:
            b = self.y
        else:
            b = np.concatenate([self.y, np.zeros(self.n_t)])

        sol = M_inv @ b
        # Weights
        w = sol[: self.n_pts]
        # Trend Coefficients
        c = sol[self.n_pts :]

        return w, c, M_inv

    def fit(
        self,
        lam: float | None = None,
        set_results: bool = True,
        compute_stats: bool = True,
        **kwargs,
    ) -> SplineResult:
        """Solve"""
        if lam is None:
            print("Estimating Lambda Using GCV")
            lam = self.estimate_lambda_gcv(**kwargs)

        weights, trend_coefs, M_inv = self._get_weights(lam=lam)

        y_fit = self._compute_result(
            self.covariance, weights, self.trend_basis, trend_coefs
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

    def predict(
        self,
        X_test: np.ndarray,
        covariance: np.ndarray | None = None,
        compute_se: bool = True,
        result: SplineResult | NoneType = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Predict for new positions"""
        if result is not None:
            self.result = result
            self.fitted = True

        if not self.fitted:
            raise ValueError("Must first fit")

        if not hasattr(self, "result"):
            raise ValueError(
                "self.fitted is True, but 'result' attribute is missing"
            )
        if result is None:
            if not self.fitted or hasattr(self, "result"):
                raise ValueError(
                    "Must first fit the model if not providing a result."
                )

            result = self.result

        if not hasattr(result, "var"):
            print("Computing variance of the fit.")
            result.fit_stats()

        P, n_t = self.get_trend_basis(X_test)
        n_pts = len(X_test)
        if covariance is None:
            D = self.dist_func(X_test, self.X)
            covariance = self.kernel(D)
            del D
        elif covariance.shape != (n_pts, self.n_pts):
            raise ValueError(
                f"Incorrect covariance shape, must be {(n_pts, self.n_pts)}."
            )

        prediction = self._compute_result(
            covariance, result.weights, P, result.trend_coefs
        )

        if not compute_se:
            return prediction, None

        U = np.hstack([covariance, P]) if n_t > 0 else covariance
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
                + "'scipy.optimize.OptimizeResult'"
            )
        return np.exp(result.x)


class ThinPlateSpline(Spline):
    """Thin Plate Spline Interpolation and Smoothing"""

    method: str = "thin_plate_spline"

    @property
    def default_error_cov(self) -> np.ndarray:
        """Default Error Covariance"""
        return np.eye(self.n_pts)

    def dist_func(
        self,
        positions: np.ndarray,
        X2: np.ndarray | NoneType = None,
    ) -> np.ndarray:
        """Pairwise Euclidean distances between positions"""
        return (
            cdist(positions, X2)
            if X2 is not None
            else cdist(positions, positions)
        )

    def get_trend_basis(self, positions: np.ndarray) -> tuple[np.ndarray, int]:
        """Linear trend basis for universal kriging: [1, x, y, ...]"""
        n_pts = positions.shape[0]
        return np.hstack((np.ones((n_pts, 1)), positions)), 3

    def kernel(self, distances: np.ndarray) -> np.ndarray:
        """Thin Plate Spline RBF kernel"""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(
                distances == 0, 0.0, np.power(distances, 2) * np.log(distances)
            )


class SphericalThinPlateSpline(Spline):
    """
    Spherical form of Thin Plate Spline Interpolation and Smoothing.
    Adapted from [Wahba_Sphere]_.
    """

    method: str = "spherical_thin_plate_spline"

    @property
    def default_error_cov(self) -> np.ndarray:
        """Default Error Covariance"""
        return self.n_pts * np.eye(self.n_pts)

    def dist_func(
        self,
        positions: np.ndarray,
        X2: np.ndarray | NoneType = None,
    ) -> np.ndarray:
        """Pairwise Haversine distances between positions"""
        if positions.shape[1] != 2:
            raise ValueError(
                "Coordinates must be 2 dimensional lat and lon in radians."
            )
        return (
            haversine_distances(positions, X2)
            if X2 is not None
            else haversine_distances(positions)
        )

    def get_trend_basis(self, positions: np.ndarray) -> tuple[np.ndarray, int]:
        """Linear trend basis for ordinary kriging: [1]"""
        n_pts = positions.shape[0]
        return np.ones((n_pts, 1)), 1

    def kernel(self, distances: np.ndarray) -> np.ndarray:
        """Spherical Thin Plate Spline RBF kernel"""
        m = 2
        m2_1 = 2 * m - 1
        fac_m2_1 = factorial(m2_1)
        pi_frac = 1 / (2 * np.pi)
        frac = pi_frac / fac_m2_1

        with np.errstate(divide="ignore", invalid="ignore"):
            q = np.where(distances == 0, 0.5, _q2(distances))

        q[np.isnan(q)] = 0.5

        return frac * (m2_1 * q - 1)


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


class Kriging(Spline):
    """Kriging using the Spline approach"""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        covariance: np.ndarray,
        error_cov: np.ndarray | NoneType = None,
    ) -> NoneType:
        if covariance is None:
            raise NotImplementedError(
                "'covariance' must be set for Kriging methods"
            )
        super().__init__(X, y, covariance, error_cov)

    @property
    def default_error_cov(self) -> np.ndarray:
        """Default error covariance"""
        return np.zeros((self.n_pts, self.n_pts))

    def predict(
        self,
        X_test: np.ndarray,
        covariance: np.ndarray | None = None,
        compute_se: bool = True,
        result: SplineResult | NoneType = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Predict"""
        if covariance is None:
            raise NotImplementedError(
                "'covariance' must be set for Kriging methods"
            )
        return super().predict(X_test, covariance, compute_se, result)


class OrdinaryKriging(Kriging):
    """Ordinary Kriging using the Spline method"""

    method = "ordinary_kriging"

    def get_trend_basis(self, positions: np.ndarray) -> tuple[np.ndarray, int]:
        """Linear trend basis for ordinary kriging: [1]"""
        n_pts = positions.shape[0]
        return np.ones((n_pts, 1)), 1


class UniversalKriging(Kriging):
    """Universal Kriging using the Spline method"""

    method = "universal_kriging"

    def get_trend_basis(self, positions: np.ndarray) -> tuple[np.ndarray, int]:
        """Linear trend basis for universal kriging: [1, x, y, ...]"""
        n_pts = positions.shape[0]
        return np.hstack((np.ones((n_pts, 1)), positions)), 3


class VariogramKriging(Spline):
    """Use a variogram"""

    method = "variogram_kriging"

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        variogram: Variogram,
        error_cov: np.ndarray | NoneType = None,
        kriging_method: Literal["ordinary", "universal"] = "ordinary",
        distance_method: Literal["haversine", "euclidean"] = "haversine",
        distance_scale: float = 1.0,
    ) -> NoneType:
        self.variogram = variogram
        self.kriging_method = kriging_method
        self.distance_method = distance_method
        self.distance_scale = distance_scale
        super().__init__(X, y, None, error_cov)

    @property
    def default_error_cov(self) -> np.ndarray:
        """Default error covariance"""
        return np.zeros((self.n_pts, self.n_pts))

    def get_trend_basis(self, positions: np.ndarray) -> tuple[np.ndarray, int]:
        """Trend basis for kriging"""
        n_pts = positions.shape[0]
        match self.kriging_method:
            case "ordinary":
                return np.ones((n_pts, 1)), 1
            case "universal":
                return np.hstack((np.ones((n_pts, 1)), positions)), 3
            case _:
                raise ValueError(
                    f"Unexpected 'kriging_method'. Got {self.kriging_method}"
                    + "Expected one of 'ordinary' or 'universal'."
                )

    def dist_func(
        self,
        positions: np.ndarray,
        X2: np.ndarray | NoneType = None,
    ) -> np.ndarray:
        """Pairwise distances between positions"""
        match self.distance_method:
            case "haversine":
                if positions.shape[1] != 2:
                    raise ValueError(
                        "Coordinates must be 2 dimensional lat and lon in "
                        + "radians for haversine distances."
                    )
                return self.distance_scale * (
                    haversine_distances(positions, X2)
                    if X2 is not None
                    else haversine_distances(positions)
                )
            case "euclidean":
                return self.distance_scale * (
                    cdist(positions, X2)
                    if X2 is not None
                    else cdist(positions, positions)
                )
            case _:
                raise ValueError(
                    f"Unexpected 'distance_method'. Got {self.distance_method}"
                    + "Expected one of 'haversine' or 'euclidean'."
                )

    def kernel(self, distances: np.ndarray) -> np.ndarray:
        """Use Variogram"""
        if not hasattr(self, "variogram"):
            raise KeyError("Variogram is not set")

        psill = getattr(self.variogram, "psill", None)
        if psill is None:
            raise KeyError("Variogram is missing 'psill'")

        vario_mat = self.variogram.fit(distances)

        return variogram_to_covariance(vario_mat, psill)  # type: ignore (array)
