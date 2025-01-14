"""
Functions for calculating distances or distance-based covariance components.

Some functions can be used for computing pairwise-distances, for example via
squareform. Some functions can be used as a distance function for
glomar_gridding.error_covariance.dist_weights, accounting for the distance
component to an error covariance matrix.
"""

import numpy as np
import polars as pl
from sklearn.metrics.pairwise import haversine_distances

from .utils import check_cols
from .matern_and_tm import tau_dist  # noqa: F401


# NOTE: This is a Variogram result
def haversine_gaussian(
    df: pl.DataFrame,
    R: float = 6371.0,
    r: float = 40,
    s: float = 0.6,
) -> np.ndarray:
    """
    Gaussian Haversine Model

    Parameters
    ----------
    df : polars.DataFrame
        Observations, required columns are "lat" and "lon" representing
        latitude and longitude respectively.
    R : float
        Radius of the sphere on which Haversine distance is computed. Defaults
        to radius of earth in km.
    r : float
        Gaussian model range parameter
    s : float
        Gaussian model scale parameter

    Returns
    -------
    C : np.ndarray
        Distance matrix for the input positions. Result has been modified using
        the Gaussian model.
    """
    check_cols(df, ["lat", "lon"])
    pos = np.radians(df.select(["lat", "lon"]).to_numpy())
    C = haversine_distances(pos) * R
    C = np.exp(-(np.pow(C, 2)) / np.pow(r, 2))
    return s / 2 * C


def radial_dist(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """
     Computes a distance matrix of the coordinates using a spherical metric.

    Parameters
    ----------
    lat1 : float
        latitude of point A
    lon1 : float
        longitude of point A
    lat2 : float
        latitude of point B
    lon2 : float
        longitude of point B

    Returns
    -------
    Radial distance between point A and point B
    """
    # approximate radius of earth in km
    R = 6371.0
    lat1r = np.radians(lat1)
    # lon1r = math.radians(lon1)
    lat2r = np.radians(lat2)
    # lon2r = math.radians(lon2)

    dlon = np.radians(lon2 - lon1)
    dlat = lat2r - lat1r

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c
