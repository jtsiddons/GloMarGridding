"""
Functions for calculating distances or distance-based covariance components.

Some functions can be used for computing pairwise-distances, for example via
squareform. Some functions can be used as a distance function for
glomar_gridding.error_covariance.dist_weights, accounting for the distance
component to an error covariance matrix.
"""

from collections.abc import Callable
import numpy as np
import polars as pl
from sklearn.metrics.pairwise import haversine_distances
from scipy.spatial.distance import pdist, squareform

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


def euclidean_distance(
    loc1: tuple[float, float],
    loc2: tuple[float, float],
    radius: float = 6371.0,
) -> float:
    """
    Calculate the Euclidean distance in kilometers between two positions on the
    earth (specified in decimal degrees).

    See:
    https://math.stackexchange.com/questions/29157/how-do-i-convert-the-distance-between-two-lat-long-points-into-feet-meters
    https://cesar.esa.int/upload/201709/Earth_Coordinates_Booklet.pdf

    d = SQRT((x_2-x_1)**2 + (y_2-y_1)**2 + (z_2-z_1)**2)

    where

    (x_n y_n z_n) = ( Rcos(lat)cos(lon) Rcos(lat)sin(lon) Rsin(lat) )

    Parameters
    ----------
    loc1 : tuple[float, float]
        The first position
    loc2 : tuple[float, float]
        The second position
    radius : float
        The radius of the sphere used for the calculation. Defaults to the
        radius of the earth in km (6371.0 km).

    Returns
    -------
    dist : float
        The direct distance between the two input points through the sphere
        defined by the radius parameter.
    """
    lat1, lon1 = map(np.radians, loc1)
    lat2, lon2 = map(np.radians, loc2)

    x1 = np.cos(lat1) * np.cos(lon1)
    x2 = np.cos(lat2) * np.cos(lon2)
    y1 = np.cos(lat1) * np.sin(lon1)
    y2 = np.cos(lat2) * np.sin(lon2)
    z1 = np.sin(lat1)
    z2 = np.sin(lat2)
    dist = radius * np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    return dist


def haversine_distance(
    loc1: tuple[float, float],
    loc2: tuple[float, float],
    radius: float = 6371,
) -> float:
    """
    Calculate the great circle distance in kilometers between two points
    on the earth (specified in decimal degrees)

    Parameters
    ----------
    loc1 : tuple[float, float]
        The first position
    loc2 : tuple[float, float]
        The second position
    radius : float
        The radius of the sphere used for the calculation. Defaults to the
        radius of the earth in km (6371.0 km).

    Returns
    -------
    dist : float
        The haversine distance between the two input points on the sphere
        defined by the radius parameter.
    """
    lat1, lon1 = map(np.radians, loc1)
    lat2, lon2 = map(np.radians, loc2)

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )
    dist = radius * 2 * np.arcsin(np.sqrt(a))
    return dist


# OPTIM: Improve this! Much slower than haversine_distances
def calculate_distance_matrix(
    df: pl.DataFrame,
    dist_func: Callable = haversine_distance,
    lat_col: str = "lat",
    lon_col: str = "lon",
) -> np.ndarray:
    """
    Create a distance matrix from a DataFrame containing positional information,
    typically latitude and longitude, using a distance function.

    Available functions are `haversine_distance`, `euclidean_distance`. A
    custom function can be used, requiring that the function takes the form:
        (tuple[float, float], tuple[float, float]) -> float

    Parameters
    ----------
    df : polars.DataFrame
        DataFrame containing latitude and longitude columns indicating the
        positions between which distances are computed to form the distance
        matrix
    dist_func : Callable
        The function used to calculate the pairwise distances. Functions
        available for this function are `haversine_distance` and
        `euclidean_distance`.
        A custom function can be based, that takes as input two tuples of
        positions (computing a single distance value between the pair of
        positions). (tuple[float, float], tuple[float, float]) -> float
    lat_col : str
        Name of the column in the input DataFrame containing latitude values.
    lon_col : str
        Name of the column in the input DataFrame containing longitude values.

    Returns
    -------
    dist : np.ndarray[float]
        A matrix of pairwise distances.
    """
    dist = squareform(
        pdist(df.select([lat_col, lon_col]), lambda u, v: dist_func(u, v))
    )
    return dist
