"""
Variograms
----------

Varigram classes for construction of spatial covariance structure from distance
matrices.
"""

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Literal
import numpy as np
import xarray as xr

from scipy.special import gamma, kv


@dataclass()
class Variogram(ABC):
    """Generic Variogram Class - defines the abstract class"""

    @abstractmethod
    def fit(
        self, distance_matrix: np.ndarray | xr.DataArray
    ) -> np.ndarray | xr.DataArray:
        """Fit the Variogram model to a distance matrix"""
        raise NotImplementedError("Not implemented for base Variogram class")


@dataclass()
class LinearVariogram(Variogram):
    """
    Linear model

    Parameters
    ----------
    slope : float | np.ndarray
    nugget : float | np.ndarray
    """

    slope: float | np.ndarray
    nugget: float | np.ndarray

    def fit(
        self, distance_matrix: np.ndarray | xr.DataArray
    ) -> np.ndarray | xr.DataArray:
        """Fit the LinearVariogram model to a distance matrix"""
        out = self.slope * distance_matrix + self.nugget
        if isinstance(out, xr.DataArray):
            out.name = "variogram"
        return out


@dataclass()
class PowerVariogram(Variogram):
    """
    Power model

    Parameters
    ----------
    scale : float | np.ndarray
    exponent : float | np.ndarray
    nugget : float | np.ndarray
    """

    scale: float | np.ndarray
    exponent: float | np.ndarray
    nugget: float | np.ndarray

    def fit(
        self, distance_matrix: np.ndarray | xr.DataArray
    ) -> np.ndarray | xr.DataArray:
        """Fit the PowerVariogram model to a distance matrix"""
        return (
            self.scale * np.power(distance_matrix, self.exponent) + self.nugget
        )


@dataclass()
class GaussianVariogram(Variogram):
    """
    Gaussian Model

    Parameters
    ----------
    psill : float | np.ndarray
        The variance of the variogram.
    nugget : float | np.ndarray
    effective_range : float | np.ndarray | None
    range : float | np.ndarray | None
    """

    psill: float | np.ndarray
    nugget: float | np.ndarray
    effective_range: float | np.ndarray | None = None
    range: float | np.ndarray | None = None

    def __post_init__(self):
        if self.range is None and self.effective_range is None:
            raise ValueError(
                "One of range and effective_range must be specified"
            )
        if self.range is None and self.effective_range is not None:
            self.range = self.effective_range / 3
        return None

    def fit(
        self, distance_matrix: np.ndarray | xr.DataArray
    ) -> np.ndarray | xr.DataArray:
        """Fit the GaussianVariogram model to a distance matrix"""
        if self.range is None:
            raise ValueError(
                "range parameter must not be None, "
                + "it wasn't set by the __post_init__ method."
            )
        out = (
            self.psill
            * (
                1.0
                - np.exp(
                    -(
                        np.power(distance_matrix, 2.0)
                        / np.power(self.range, 2.0)
                    )
                )
            )
            + self.nugget
        )
        if isinstance(out, xr.DataArray):
            out.name = "variogram"
        return out


@dataclass()
class ExponentialVariogram(Variogram):
    """
    Exponential Model

    Parameters
    ----------
    psill : float | numpy.ndarray
        The variance of the variogram.
    nugget : float | numpy.ndarray
    effective_range : float | numpy.ndarray | None
    range : float | numpy.ndarray | None
    """

    psill: float | np.ndarray
    nugget: float | np.ndarray
    range: float | np.ndarray | None = None
    effective_range: float | np.ndarray | None = None

    def __post_init__(self):
        if self.range is None and self.effective_range is None:
            raise ValueError(
                "One of range and effective_range must be specified"
            )
        if self.range is None and self.effective_range is not None:
            self.range = self.effective_range / 3
        return None

    def fit(
        self, distance_matrix: np.ndarray | xr.DataArray
    ) -> np.ndarray | xr.DataArray:
        """Fit the ExponentialVariogram model to a distance matrix"""
        if self.range is None:
            raise ValueError(
                "range parameter must not be None, "
                + "it wasn't set by the __post_init__ method."
            )
        out = (
            self.psill * (1.0 - np.exp(-(distance_matrix / self.range)))
            + self.nugget
        )
        if isinstance(out, xr.DataArray):
            out.name = "variogram"
        return out


MaternModel = Literal["sklearn", "gstat", "karspeck"]


@dataclass()
class MaternVariogram(Variogram):
    """
    Matern Models

    Same args as the Variogram classes with additional nu, method parameters.

    Sklearn:

    1) This is called "sklearn" because if d/range = 1.0 and nu=0.5, it gives
       1/e correlation...
    2) This is NOT the same formulation as in GSTAT nor in papers about
       non-stationary anistropic covariance models (aka Karspeck paper).
    3) It is perhaps the most intitutive (because of (1)) and is used in sklearn
       GP and HadCRUT5 and other UKMO dataset.
    4) nu defaults to 0.5 (exponential; used in HADSST4 and our kriging).
       HadCRUT5 uses 1.5.
    5) The "2" is inside the square root for middle and right.

    Reference; see chapter 4.2 of:
    Rasmussen, C. E., & Williams, C. K. I. (2005).
    Gaussian Processes for Machine Learning. The MIT Press.
    https://doi.org/10.7551/mitpress/3206.001.0001

    GeoStatic:

    Similar to Sklearn MaternVariogram model but uses the range scaling in
    gstat.
    Note: there are no square root 2 or nu in middle and right

    Yields the same answer to sklearn MaternVariogram if nu==0.5
    but are otherwise different.

    Karspeck:

    Similar to Sklearn MaternVariogram model but uses the form in Karspeck paper
    Note: Note the 2 is outside the square root for middle and right
    e-folding distance is now at d/SQRT(2) for nu=0.5

    Parameters
    ----------
    psill : float | np.ndarray
        Sill of the variogram where it will flatten out. Values in the variogram
        will not exceed psill + nugget. This value is the variance.
    nugget : float | np.ndarray
        The value of the independent variable at distance 0
    effective_range : float | np.ndarray | None
        Effective Range, this is the lag where 95% of the sill are exceeded.
        This is not the range parameter, which is defined as r/3 if nu < 0.5 or
        nu > 10, otherwise r/2 (where r is the effective range). One of
        effective_range and range must be set.
    range : float | ndarray | None
        The range parameter. One of range and effective_range must be set. If
        range is not set, it will be computed from effective_range.
    nu : float | np.ndarray
        Smoothing parameter, shapes to a smooth or rough variogram function
    method : MaternModel
        One of "sklearn", "gstat", or "karspeck"
        sklearn: https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Matern.html#sklearn.gaussian_process.kernels.Matern
        gstat: https://scikit-gstat.readthedocs.io/en/latest/reference/models.html#matern-model
        karspeck: https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.900
    """

    psill: float | np.ndarray
    nugget: float | np.ndarray
    effective_range: float | np.ndarray | None = None
    range: float | np.ndarray | None = None
    nu: float | np.ndarray = 0.5
    method: MaternModel = "sklearn"

    def __post_init__(self) -> None:
        if self.effective_range is None and self.range is None:
            raise ValueError(
                "One of range and effective_range must be specified"
            )
        if self.range is None and self.effective_range is not None:
            self.range = (
                self.effective_range / 2
                if 0.5 <= self.nu <= 10
                else self.effective_range / 3
            )
        elif self.effective_range is None and self.range is not None:
            self.effective_range = (
                self.range * 2 if 0.5 <= self.nu <= 10 else self.range * 3
            )
        return None

    @property
    def _left(self):
        return 1.0 / (gamma(self.nu) * np.power(2.0, self.nu - 1.0))

    def _middle(
        self, dist_over_range: np.ndarray | xr.DataArray
    ) -> np.ndarray | xr.DataArray:
        match self.method.lower():
            case "sklearn":
                return np.power(
                    np.sqrt(2.0 * self.nu) * dist_over_range,
                    self.nu,
                )
            case "gstat":
                return np.power(dist_over_range, self.nu)
            case "karspeck":
                return np.power(
                    2.0 * np.sqrt(self.nu) * dist_over_range,
                    self.nu,
                )
            case _:
                raise ValueError("Unexpected 'method' value")

    def _right(
        self, dist_over_range: np.ndarray | xr.DataArray
    ) -> np.ndarray | xr.DataArray:
        match self.method.lower():
            case "sklearn":
                return kv(
                    self.nu,
                    np.sqrt(2.0 * self.nu) * dist_over_range,
                )
            case "gstat":
                return kv(self.nu, dist_over_range)
            case "karspeck":
                return kv(
                    self.nu,
                    2.0 * np.sqrt(self.nu) * dist_over_range,
                )
            case _:
                raise ValueError("Unexpected 'method' value")

    def fit(
        self, distance_matrix: np.ndarray | xr.DataArray
    ) -> np.ndarray | xr.DataArray:
        """Fit the MaternVariogram model to a distance matrix"""
        if self.range is None:
            raise ValueError(
                "range parameter must not be None, "
                + "it wasn't set by the __post_init__ method."
            )
        dist_over_range = distance_matrix / self.range
        out = (
            self.psill
            * (
                1
                - (
                    self._left
                    * self._middle(dist_over_range)
                    * self._right(dist_over_range)
                )
            )
            + self.nugget
        )
        # Matern is undefined at 0 distance, so replace nan on diagnonal
        if isinstance(out, xr.DataArray):
            np.fill_diagonal(out.values, self.nugget)
            out.name = "variogram"
        else:
            np.fill_diagonal(out, self.nugget)
        return out


def variogram_to_covariance(
    variogram: np.ndarray | xr.DataArray,
    variance: np.ndarray | float,
) -> np.ndarray | xr.DataArray:
    """
    Convert a variogram matrix to a covariance matrix.

    This is given by:
        covariance = variance - variogram

    Parameters
    ----------
    variogram : numpy.ndarray | xarray.DataArray
        The variogram matrix, output of Variogram.fit.
    variance : numpy.ndarray | float
        The variance

    Returns
    -------
    cov : numpy.ndarray | xarray.DataArray
        The covariance matrix
    """
    cov = variance - variogram
    if isinstance(cov, xr.DataArray):
        cov.name = "covariance"
    return cov
