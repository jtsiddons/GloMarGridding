"""
Varigram classes for construction of spatial covariance structure from distance
matrices.
"""

from dataclasses import dataclass
from typing import Literal
import numpy as np
import xarray as xr

from scipy.special import gamma, kv


@dataclass(frozen=True)
class Variogram:
    """Place holder"""

    def fit(
        self, distance_matrix: np.ndarray | xr.DataArray
    ) -> np.ndarray | xr.DataArray:
        """Fit the Variogram model to a distance matrix"""
        raise NotImplementedError("Not implemented for base Variogram class")


@dataclass(frozen=True)
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


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class GaussianVariogram(Variogram):
    """
    Gaussian Model

    Parameters
    ----------
    psill : float | np.ndarray
    effective_range : float | np.ndarray
    nugget : float | np.ndarray
    """

    psill: float | np.ndarray
    effective_range: float | np.ndarray
    nugget: float | np.ndarray

    @property
    def range(self):
        """The range parameter"""
        return self.effective_range / 3
        # return self.effective_range * (4 / 7)

    def fit(
        self, distance_matrix: np.ndarray | xr.DataArray
    ) -> np.ndarray | xr.DataArray:
        """Fit the GaussianVariogram model to a distance matrix"""
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


@dataclass(frozen=True)
class ExponentialVariogram(Variogram):
    """
    Exponential Model

    Parameters
    ----------
    psill : float | np.ndarray
    effective_range : float | np.ndarray
    nugget : float | np.ndarray
    """

    psill: float | np.ndarray
    effective_range: float | np.ndarray
    nugget: float | np.ndarray

    @property
    def range(self):
        """The range paramter"""
        return self.effective_range / 3

    def fit(
        self, distance_matrix: np.ndarray | xr.DataArray
    ) -> np.ndarray | xr.DataArray:
        """Fit the ExponentialVariogram model to a distance matrix"""
        out = (
            self.psill * (1.0 - np.exp(-(distance_matrix / self.range)))
            + self.nugget
        )
        if isinstance(out, xr.DataArray):
            out.name = "variogram"
        return out


MaternModel = Literal["sklearn", "gstat", "karspeck"]


@dataclass(frozen=True)
class MaternVariogram(Variogram):
    """
    Matern Models

    Same args as the Variogram classes with additional nu, method parameters.

    Sklearn
    -------
    1) This is called ``sklearn'' because if d/range_ = 1.0 and nu=0.5, it gives
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

    GeoStatic
    ---------
    Similar to Classic MaternVariogram model but uses the range scaling in
    gstat.
    Note: there are no square root 2 or nu in middle and right

    Yields the same answer to Classic MaternVariogram if nu==0.5
    but are otherwise different.

    Karspeck
    --------
    Similar to Classic MaternVariogram model but uses the form in Karspeck paper
    Note: Note the 2 is outside the square root for middle and right
    e-folding distance is now at d/SQRT(2) for nu=0.5

    Parameters
    ----------
    psill : float | np.ndarray
        Sill of the variogram where it will flatten out. Values in the variogram
        will not exceed psill + nugget
    effective_range : float | np.ndarray
        Effective Range, this is the lag where 95% of ths sill are exceeded.
        This is not the range parameter, which is defined as r/3 if nu < 0.5 or
        nu > 10, otherwise r/2 (where r is the effective range).
    nugget : float | np.ndarray
        The value of the independent variable at distance 0
    nu : float | np.ndarray
        Smoothing parameter, shapes to a smooth or rough variogram function
    method : MaternModel
        One of "sklearn", "gstat", or "karspeck"
        sklearn:
            https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Matern.html#sklearn.gaussian_process.kernels.Matern
        gstat:
            https://scikit-gstat.readthedocs.io/en/latest/reference/models.html#matern-model
        karspeck:
            https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.900
    """

    psill: float | np.ndarray
    effective_range: float | np.ndarray
    nugget: float | np.ndarray
    nu: float | np.ndarray = 0.5
    method: MaternModel = "sklearn"

    @property
    def range(self):
        """The range parameter"""
        if 0.5 <= self.nu <= 10:
            return self.effective_range / 2
        return self.effective_range / 3

    @property
    def _left(self):
        return 1.0 / (gamma(self.nu) * np.power(2.0, self.nu - 1.0))

    def _middle(
        self, distance_matrix: np.ndarray | xr.DataArray
    ) -> np.ndarray | xr.DataArray:
        match self.method.lower():
            case "sklearn":
                return np.power(
                    np.sqrt(2.0 * self.nu) * distance_matrix / self.range,
                    self.nu,
                )
            case "gstat":
                return np.power(distance_matrix / self.range, self.nu)
            case "karspeck":
                return np.power(
                    2.0 * np.sqrt(self.nu) * distance_matrix / self.range,
                    self.nu,
                )
            case _:
                raise ValueError("Unexpected 'method' value")

    def _right(
        self, distance_matrix: np.ndarray | xr.DataArray
    ) -> np.ndarray | xr.DataArray:
        match self.method.lower():
            case "sklearn":
                return kv(
                    self.nu,
                    np.sqrt(2.0 * self.nu) * distance_matrix / self.range,
                )
            case "gstat":
                return kv(self.nu, distance_matrix / self.range)
            case "karspeck":
                return kv(
                    self.nu,
                    2.0 * np.sqrt(self.nu) * distance_matrix / self.range,
                )
            case _:
                raise ValueError("Unexpected 'method' value")

    def fit(
        self, distance_matrix: np.ndarray | xr.DataArray
    ) -> np.ndarray | xr.DataArray:
        """Fit the MaternVariogram model to a distance matrix"""
        out = (
            self.psill
            * (
                1
                - (
                    self._left
                    * self._middle(distance_matrix)
                    * self._right(distance_matrix)
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
