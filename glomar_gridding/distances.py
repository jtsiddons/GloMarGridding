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

"""
Functions for calculating distances or distance-based covariance components.

Some functions can be used for computing pairwise-distances, for example via
squareform. Some functions can be used as a distance function for
glomar_gridding.error_covariance.dist_weights, accounting for the distance
component to an error covariance matrix.

Functions for computing covariance using Matern Tau by Steven Chan (@stchan).
"""

from collections.abc import Callable
from math import cos, sin
from typing import get_args

import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
from shapely.geometry import Point
from sklearn.metrics.pairwise import euclidean_distances, haversine_distances

from glomar_gridding.types import DeltaXMethod
from glomar_gridding.utils import check_cols


def rot_mat(angle: float) -> np.ndarray:
    """
    Compute a 2d rotation matrix from an angle.

    The input angle must be in radians
    """
    c_ang = cos(angle)
    s_ang = sin(angle)
    return np.array([[c_ang, -s_ang], [s_ang, c_ang]])


def inv_2d(mat: np.ndarray) -> np.ndarray:
    """Compute the inverse of a 2 x 2 matrix"""
    det_denom = mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0]
    if det_denom == 0:
        raise ValueError("Denominator is 0")
    inv = np.array([[mat[1, 1], -mat[0, 1]], [-mat[1, 0], mat[0, 0]]])
    return inv / det_denom


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
    C = np.exp(-(np.power(C, 2)) / np.pow(r, 2))
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
    df: pl.DataFrame,
    lat_coord: str = "lat",
    lon_coord: str = "lon",
    to_radians: bool = True,
    radius: float = 6371.0,
) -> np.ndarray:
    r"""
    Calculate the Euclidean distance in kilometers between pairs of lat, lon
    points on the earth (specified in decimal degrees).

    .. math::
        d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2 + (z_2 - z_1)^2}

    where

    .. math::
        (x_n, y_n, z_n) = (R\cos(lat)\cos(lon), R\cos(lat)\sin(lon), R\sin(lat))

    Parameters
    ----------
    df : polars.DataFrame
        DataFrame containing latitude and longitude columns indicating the
        positions between which distances are computed to form the distance
        matrix
    lat_coord : str
        Name of the column in the input DataFrame containing latitude values.
    lon_coord : str
        Name of the column in the input DataFrame containing longitude values.
    to_radians : bool
        Convert inputs from degrees to radians. The distance methodology assumes
        that the input latitude and longitudes are in radians so must be
        converted if they are in degrees.
    radius : float
        The radius of the sphere used for the calculation. Defaults to the
        radius of the earth in km (6371.0 km).

    Returns
    -------
    dist : float
        The direct pairwise distance between the positions in the input
        DataFrame through the sphere defined by the radius parameter.

    References
    ----------
    https://math.stackexchange.com/questions/29157/how-do-i-convert-the-distance-between-two-lat-long-points-into-feet-meters
    https://cesar.esa.int/upload/201709/Earth_Coordinates_Booklet.pdf
    """
    df = df.select([lat_coord, lon_coord])
    if to_radians:
        df = df.select(pl.all().radians())

    df = df.select(
        [
            (pl.col(lat_coord).cos() * pl.col(lon_coord).cos()).alias("x"),
            (pl.col(lat_coord).cos() * pl.col(lon_coord).sin()).alias("y"),
            pl.col(lat_coord).sin().alias("z"),
        ]
    )

    return euclidean_distances(df) * radius


def haversine_distance_from_frame(
    df: pl.DataFrame,
    lat_coord: str = "lat",
    lon_coord: str = "lon",
    to_radians: bool = True,
    radius: float = 6371,
) -> np.ndarray:
    """
    Calculate the great circle distance in kilometers between pairs of lat, lon
    points on the earth (specified in decimal degrees).

    Parameters
    ----------
    df : polars.DataFrame
        DataFrame containing latitude and longitude columns indicating the
        positions between which distances are computed to form the distance
        matrix
    lat_coord : str
        Name of the column in the input DataFrame containing latitude values.
    lon_coord : str
        Name of the column in the input DataFrame containing longitude values.
    to_radians : bool
        Convert inputs from degrees to radians. The distance methodology assumes
        that the input latitude and longitudes are in radians so must be
        converted if they are in degrees.
    radius : float
        The radius of the sphere used for the calculation. Defaults to the
        radius of the earth in km (6371.0 km).

    Returns
    -------
    dist : numpy.ndarray
        The pairwise haversine distances between the inputs in the DataFrame,
        on the sphere defined by the radius parameter.
    """
    df = df.select([lat_coord, lon_coord])
    if to_radians:
        df = df.select(pl.all().radians())
    return haversine_distances(df) * radius


def calculate_distance_matrix(
    df: pl.DataFrame,
    dist_func: Callable = haversine_distance_from_frame,
    **dist_kwargs,
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
    **dist_kwargs
        Keyword arguments to pass to the distance function.

    Returns
    -------
    dist : np.ndarray[float]
        A matrix of pairwise distances.
    """
    return dist_func(
        df,
        **dist_kwargs,
    )


def _latlon2ne(
    latlons: np.ndarray,
    latlons_in_rads: bool = False,
    latlon0: tuple[float, float] = (0.0, 180.0),
) -> np.ndarray:
    """
    Compute Northing and Easting from Latitude and Longitude

    latlons -- a (N, 2) (numpy) array of latlons
    By GIS and netcdf as well as sklearn convention
    [X, 0] = lat
    [X, 1] = lon
    aka [LAT, LON] [Y,X] NOT [X,Y]!!!!!

    latlons_in_rads -- boolean stating if latlons are in radians
    (default False -- input are in degrees)

    latlon0 - a (lat, lon) in degree tuple stating
    the central point of Transverse Mercator for reprojecting to
    Northing East

    returns a (N, 2) numpy array of Northing Easting [km]
    """
    if latlons_in_rads:
        latlons2 = np.rad2deg(latlons)
    else:
        latlons2 = latlons.copy()
    df0 = pd.DataFrame({"lat": latlons2[:, 0], "lon": latlons2[:, 1]})
    df0["geometry"] = df0.apply(lambda row: Point([row.lon, row.lat]), axis=1)
    df0 = gpd.GeoDataFrame(df0, geometry="geometry", crs="EPSG:4326")
    #
    # Transverse Mercator projection
    # Recommended to be centered on the central point
    # of the grid box
    # Large distortions will occur if you use a single value for
    # latlon0 for the entire globe
    proj4 = "+proj=tmerc +lat_0=" + str(latlon0[0])
    proj4 += " +lon_0=" + str(latlon0[1])
    proj4 += " +k=0.9996 +x_0=0 +y_0=0 +units=km"
    df1: gpd.GeoDataFrame = gpd.GeoDataFrame(
        df0,
        crs="EPSG:4326",
        geometry=gpd.points_from_xy(df0["lon"], df0["lat"]),
    )
    df1.to_crs(proj4, inplace=True)
    df1["easting"] = df1.geometry.x
    df1["northing"] = df1.geometry.y
    pos = df1[["northing", "easting"]].to_numpy()
    return pos


def _paired_vector_dist(yx: np.ndarray) -> np.ndarray:
    """
    Input:
    (N, 2) array
    [X, 0] = lat or northing
    [X, 1] = lon or easting
    """
    return yx[:, None, :] - yx


def sigma_rot_func(
    Lx: float,
    Ly: float,
    theta: float | None,
) -> np.ndarray:
    """
    Equation 15 in Karspeck el al 2011 and Equation 6
    in Paciorek and Schervish 2006,
    assuming Sigma(Lx, Ly, theta) locally/moving-window invariant or
    we have already taken the mean (Sigma overbar, PP06 3.1.1)

    Lx, Ly - anistropic variogram length scales
    theta - angle relative to lines of constant latitude
    theta should be radians, and the fitting code outputs radians by default

    Returns
    -------
    sigma : np.ndarray
        2 x 2 matrix
    """
    L = np.diag([Lx**2.0, Ly**2.0])
    if theta is None:
        return L
    R = rot_mat(theta)
    sigma = R @ L @ R.T
    return sigma


def tau_dist(
    dE: float,
    dN: float,
    sigma: np.ndarray,
) -> np.ndarray:
    """
    Eq.15 in Karspeck paper
    but it is standard formulation to the
    Mahalanobis distance
    https://en.wikipedia.org/wiki/Mahalanobis_distance
    10.1002/qj.900
    """
    dx_vec = np.array([dE, dN])
    return np.sqrt(dx_vec.T @ inv_2d(sigma) @ dx_vec)


def _compute_tau_wrapper(dyx: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Wrapper function for computing tau"""
    DE = dyx[:, :, 1]
    DN = dyx[:, :, 0]

    def compute_tau2(dE, dN):
        return tau_dist(dE, dN, sigma)

    compute_tau_vectorised = np.vectorize(compute_tau2)
    return compute_tau_vectorised(DE, DN)


def tau_dist_from_frame(df: pl.DataFrame) -> np.ndarray:
    """
    Compute the tau/Mahalanobis matrix for all records within a gridbox

    Can be used as an input function for observations.dist_weight.

    Eq.15 in Karspeck paper
    but it is standard formulation to the
    Mahalanobis distance
    https://en.wikipedia.org/wiki/Mahalanobis_distance
    10.1002/qj.900

    By Steven Chan - @stchan

    Parameters
    ----------
    df : polars.DataFrame
        The observational DataFrame, containing positional information for each
        observation ("lat", "lon"), gridbox specific positional information
        ("grid_lat", "grid_lon"), and ellipse length-scale parameters used for
        computation of `sigma` ("grid_lx", "grid_ly", "grid_theta").

    Returns
    -------
    tau : numpy.matrix
        A matrix of dimension n x n where n is the number of rows in `df` and
        is the tau/Mahalanobis distance.
    """
    required_cols = [
        "grid_lon",
        "grid_lat",
        "grid_lx",
        "grid_ly",
        "grid_theta",
        "lat",
        "lon",
    ]
    check_cols(df, required_cols)
    # Get northing and easting
    lat0, lon0 = df.select(["grid_lat", "grid_lon"]).row(0)
    latlons = np.asarray(df.select(["lat", "lon"]).to_numpy())
    ne = _latlon2ne(latlons, latlons_in_rads=False, latlon0=(lat0, lon0))
    paired_dist = _paired_vector_dist(ne)

    # Get sigma
    Lx, Ly, theta = df.select(["grid_lx", "grid_ly", "grid_theta"]).row(0)
    sigma = sigma_rot_func(Lx, Ly, theta)

    tau = _compute_tau_wrapper(paired_dist, sigma)
    return np.exp(-tau)


def mahal_dist_func(
    delta_x: np.ndarray,
    delta_y: np.ndarray,
    Lx: float,
    Ly: float,
    theta: float | None = None,
) -> np.ndarray:
    """
    Calculate tau from displacements, Lx, Ly, and theta (if it is known). For
    an array of displacements, for a set of scalar ellipse parameters, Lx, Ly,
    and theta.

    Parameters
    ----------
    delta_x, delta_y : numpy.ndarray
        displacement to remote point as in: (delta_x) i + (delta_y) j in old
        school vector notation
    Lx, Ly : float
        Lx, Ly scale (km or degrees)
    theta : float | None
        rotation angle in radians

    Returns
    -------
    tau : float
        Mahalanobis distance
    """
    # sigma is 2x2 matrix
    if theta is not None:
        sigma = sigma_rot_func(Lx, Ly, theta)
    else:
        sigma = np.diag(np.array([Lx**2.0, Ly**2.0]))

    sigma_inv = inv_2d(sigma)
    # Direct computation of result
    return np.sqrt(
        delta_x * (delta_x * sigma_inv[0, 0] + delta_y * sigma_inv[0, 1])
        + delta_y * (delta_x * sigma_inv[1, 0] + delta_y * sigma_inv[1, 1])
    )


def displacements(
    lats: np.ndarray,
    lons: np.ndarray,
    lats2: np.ndarray | None = None,
    lons2: np.ndarray | None = None,
    delta_x_method: DeltaXMethod | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate east-west and north-south displacement matrices for all pairs
    of input positions.

    The results are not scaled by any radius, this should be performed outside
    of this function.

    Parameters
    ----------
    lats : numpy.ndarray
        The latitudes of the positions, should be provided in degrees.
    lons : numpy.ndarray
        The longitudes of the positions, should be provided in degrees.
    lats2 : numpy.ndarray
        The latitudes of the optional second positions, should be provided in
        degrees.
    lons2 : numpy.ndarray
        The longitudes of the optional second positions, should be provided in
        degrees.
    delta_x_method : str | None
        One of "Met_Office" or "Modified_Met_Office". If set to None, the
        displacements will be returned in degrees, rather than actual distance
        values. Set to "Met_Office" to use a cylindrical approximation, set
        to "Modified_Met_Office" to use an approximation that uses the average
        of the latitudes to set the horizontal displacement scale.

    Returns
    -------
    disp_y : numpy.ndarray
        The north-south displacements.
    disp_x : numpy.ndarray
        The east-west displacements.
    """
    if delta_x_method is not None and delta_x_method not in get_args(
        DeltaXMethod
    ):
        raise ValueError(
            f"Unknown 'delta_x_method' value, got '{delta_x_method}'"
        )
    _l2none = lats2 is None
    lats2 = lats2 if lats2 is not None else lats
    lons2 = lons2 if lons2 is not None else lons

    disp_y = np.subtract.outer(lats, lats2)
    disp_x = np.subtract.outer(lons, lons2)
    disp_x[disp_x > 180.0] -= 360.0
    disp_x[disp_x < -180.0] += 360.0

    if delta_x_method is None:
        return disp_y, disp_x

    disp_y = np.deg2rad(disp_y)
    disp_x = np.deg2rad(disp_x)

    if delta_x_method == "Modified_Met_Office":
        lats = np.radians(lats)
        cos_lats = np.cos(lats)
        if _l2none:
            y_cos_mean = 0.5 * np.add.outer(cos_lats, cos_lats)
        else:
            cos_lats2 = np.cos(np.radians(lats2))
            y_cos_mean = 0.5 * np.add.outer(cos_lats, cos_lats2)

        disp_x = disp_x * y_cos_mean

    return disp_y, disp_x
