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
from math import factorial
from types import NoneType

import numpy as np
from scipy.optimize import OptimizeResult, minimize_scalar
from scipy.spatial.distance import cdist
from scipy.stats import norm
from sklearn.metrics.pairwise import haversine_distances


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

        self.X = X
        self.y = y
        self.n_pts = len(self.y)
        distances = self.distances(self.X)
        self.P = self.trend_basis(self.X)
        self.K = self.kernel(distances)
        del distances

        self.error_cov = (
            error_cov if error_cov is not None else self.default_error_cov
        )

        if error_cov is not None and self.error_cov.shape[0] != self.n_pts:
            raise ValueError("Mismatch in size of error covariance")

        return None

    def clear(self) -> NoneType:
        """
        Clear the set variables.

        Clears the "D", "K", "P", "w", "c", and "lam" attributes.
        """
        to_delete = ["D", "K", "P", "w", "c", "lam"]
        for var in to_delete:
            if hasattr(self, var):
                delattr(self, var)
        return None

    @property
    def default_error_cov(self) -> np.ndarray:
        """Default error covariance"""
        raise NotImplementedError(
            "Not implemented for the generic Spline class"
        )

    def fit(
        self,
        lam: float | None = None,
        set_results: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit a spline model with an optional smoothing parameter"""
        if not hasattr(self, "P"):
            self.P = self.trend_basis(self.X)
        if not hasattr(self, "K"):
            distances = self.distances(self.X)
            self.K = self.kernel(distances)
            del distances

        if lam is None:
            lam = getattr(self, "lam", 0.0)

        K_reg = self.K + lam * self.error_cov

        A = np.block(
            [
                [K_reg, self.P],
                [self.P.T, np.zeros((self.P.shape[1], self.P.shape[1]))],
            ]
        )

        b = np.concatenate([self.y, np.zeros(self.P.shape[1])])

        sol = np.linalg.solve(A, b)
        w = sol[: self.n_pts]
        c = sol[self.n_pts :]

        if set_results:
            self.w = w
            self.c = c
            self.lam = lam
            self.K_reg = K_reg

        return w, c

    @abstractmethod
    def distances(
        self,
        positions: np.ndarray,
        X2: np.ndarray | NoneType = None,
    ) -> np.ndarray:
        """Distances between each pair of points"""
        raise NotImplementedError(
            "Not implemented for the generic Spline class"
        )

    @abstractmethod
    def trend_basis(self, positions: np.ndarray) -> np.ndarray:
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

    def solve(
        self,
        lam: float | None = None,
        w: np.ndarray | NoneType = None,
        c: np.ndarray | NoneType = None,
    ) -> np.ndarray:
        """Solve"""
        # "^" is XOR (Exclusive OR)
        if (w is None) ^ (c is None):
            raise ValueError()

        if lam is None:
            lam = getattr(self, "lam", 0.0)

        if w is None:
            if not hasattr(self, "w") or lam != self.lam:
                w, c = self.fit(lam=lam, set_results=True)  # type: ignore
            else:
                w = self.w
                c = self.c

        return self.K @ w + self.P @ c

    def fit_variance(
        self,
        y_fit: np.ndarray,
        alpha: float = 0.05,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get variance"""
        residual = self.y - y_fit

        tmp = np.linalg.inv(self.K_reg)

        df: float = self.n_pts - np.trace(self.K_reg @ tmp)
        sigma2: float = np.sum(residual**2) / df
        H_diag: np.ndarray = np.sum(self.K_reg * tmp, axis=1)

        var_y_fit: np.ndarray = sigma2 * H_diag

        z = norm.ppf(1 - alpha / 2)
        sigma = z * np.sqrt(var_y_fit)

        return sigma, var_y_fit

    def predict(
        self,
        X_test: np.ndarray,
        lam: float | None = None,
        w: np.ndarray | NoneType = None,
        c: np.ndarray | NoneType = None,
    ) -> np.ndarray:
        """Predict for new positions"""
        if (w is None) ^ (c is None):
            raise ValueError()

        if lam is None:
            lam = getattr(self, "lam", 0.0)

        if w is None:
            if not hasattr(self, "w") or lam != self.lam:
                w, c = self.fit(lam=lam, set_results=False)
            else:
                w = self.w
                c = self.c

        D = self.distances(X_test, self.X)
        P = self.trend_basis(X_test)
        K = self.kernel(D)
        del D

        return K @ w + P @ c

    def estimate_lambda_gcv(
        self,
        set_result: bool = True,
        **kwargs,
    ) -> float:
        """
        Use Generalised Cross-Validation to estimate an optimal smoothing
        parameter.

        Parameters
        ----------
        lambda_bounds : tuple[float, float]
            Bounds on the optimisation for lambda.
        set_result : bool
            Store the result as the 'lam' attribute.

        Returns
        -------
        lam : float
            The result of an optimisation for the smoothing parameter.
        """

        def gcv_loss(lam: float) -> float:
            w, c = self.fit(lam, set_results=False)
            y_fit = self.solve(lam, w=w, c=c)
            residual = self.y - y_fit

            # Approximate hat matrix trace (for GCV)
            K_reg = self.K + lam * self.error_cov
            H = self.K @ np.linalg.solve(K_reg, np.eye(self.n_pts))
            tr_H = np.trace(H)
            return np.sum(np.power(residual, 2)) / np.power(
                self.n_pts - tr_H, 2
            )

        result = minimize_scalar(gcv_loss, **kwargs)

        if not isinstance(result, OptimizeResult):
            raise TypeError(
                "result of estimation is not an instance of "
                + "'scipy.optimize.OptimizeResult'"
            )
        lam = result.x
        if set_result:
            self.lam = lam
        return lam


class ThinPlateSpline(Spline):
    """Thin Plate Spline Interpolation and Smoothing"""

    method: str = "thin_plate_spline"

    @property
    def default_error_cov(self) -> np.ndarray:
        """Default Error Covariance"""
        return np.eye(self.n_pts)

    def distances(
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

    def trend_basis(self, positions: np.ndarray) -> np.ndarray:
        """Linear trend basis for universal kriging: [1, x, y, ...]"""
        n_pts = positions.shape[0]
        return np.hstack((np.ones((n_pts, 1)), positions))

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

    def distances(
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

    def trend_basis(self, positions: np.ndarray) -> np.ndarray:
        """Linear trend basis for universal kriging: [1, x, y, ...]"""
        n_pts = positions.shape[0]
        # return np.hstack((np.ones((n_pts, 1)), positions))
        return np.ones((n_pts, 1))

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
    z = np.cos(distances)
    W = (1 - z) / 2
    sqrtW = np.sqrt(W)
    C = 2 * sqrtW
    A = np.log(1 + 1 / sqrtW)
    return z, W, C, A


def _q2(distances: np.ndarray) -> np.ndarray:
    """R with q2 from Wabha"""
    _, W, C, A = _zwca(distances)
    return (A * (12 * W**2 - 4 * W) - 6 * C * W + 6 * W + 1) / 2
