"""Spline Methods"""

from abc import ABC, abstractmethod
from types import NoneType
from scipy.optimize import OptimizeResult, minimize_scalar
from scipy.spatial.distance import cdist

import numpy as np


class Spline(ABC):
    """
    Class for Spline Interpolatation and Smoothing approaches.
    ...
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

        self.error_cov = (
            error_cov if error_cov is not None else self.default_error_cov
        )

        return None

    @property
    def default_error_cov(self) -> np.ndarray:
        """Default error covariance"""
        raise NotImplementedError()

    def fit(
        self,
        lam: float = 0,
        set_results: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit a spline model with an optional smoothing parameter"""
        if not hasattr(self, "D"):
            self.D = self.distances(self.X)
        if not hasattr(self, "P"):
            self.P = self.trend_basis(self.X)
        if not hasattr(self, "K"):
            self.K = self.kernel(self.D)

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

        return w, c

    @abstractmethod
    def distances(
        self,
        positions: np.ndarray,
        X2: np.ndarray | None = None,
    ) -> np.ndarray:
        """Distances between each pair of points"""
        raise NotImplementedError()

    @abstractmethod
    def trend_basis(self, positions: np.ndarray) -> np.ndarray:
        """Trend Basis"""
        raise NotImplementedError()

    @abstractmethod
    def kernel(self, distances: np.ndarray) -> np.ndarray:
        """Spline Kernel"""
        raise NotImplementedError()

    def solve(
        self,
        lam: float = 0,
        w: np.ndarray | None = None,
        c: np.ndarray | None = None,
    ) -> np.ndarray:
        """Solve"""
        if (w is None) ^ (c is None):
            raise ValueError()

        if w is None:
            if not hasattr(self, "w") or lam != self.lam:
                w, c = self.fit(lam=lam, set_results=False)
            else:
                w = self.w
                c = self.c

        return self.K @ w + self.P @ c

    def predict(
        self,
        X_test: np.ndarray,
        lam: float = 0,
        w: np.ndarray | None = None,
        c: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict for new positions"""
        if (w is None) ^ (c is None):
            raise ValueError()

        if w is None:
            if not hasattr(self, "w") or lam != self.lam:
                w, c = self.fit(lam=lam, set_results=False)
            else:
                w = self.w
                c = self.c

        D = self.distances(X_test, self.X)
        P = self.trend_basis(X_test)
        K = self.kernel(D)

        return K @ w + P @ c

    def estimate_lambda_gcv(
        self,
        lambda_bounds: tuple[float, float] = (1e-6, 1e2),
    ) -> float:
        """Something"""

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

        result = minimize_scalar(
            gcv_loss, bounds=lambda_bounds, method="bounded"
        )
        if not isinstance(result, OptimizeResult):
            raise TypeError(
                "result of estimation is not an instance of "
                + "'scipy.optimize.OptimizeResult'"
            )
        return result.x


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
        X2: np.ndarray | None = None,
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
