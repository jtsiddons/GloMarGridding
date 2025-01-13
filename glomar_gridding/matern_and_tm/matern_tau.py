"""
Functions for computing covariance using Matern Tau.

This requires geopandas, which may not be avail
by default

Author: Steven Chan (@stchan)
"""

import pandas as pd
import polars as pl
import geopandas as gpd
import numpy as np
from scipy.spatial.transform import Rotation
from shapely.geometry import Point

from ..utils import check_cols


def latlon2ne(
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
    pt0 = df0.apply(lambda row: Point([row.lon, row.lat]), axis=1)
    df0 = gpd.GeoDataFrame(df0, geometry=pt0, crs="EPSG:4326")
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


def paired_vector_dist(yx: np.ndarray) -> np.ndarray:
    """
    Input:
    (N, 2) array
    [X, 0] = lat or northing
    [X, 1] = lon or easting
    """
    return yx[:, None, :] - yx


def Ls2sigma(Lx: float, Ly: float, theta: float) -> np.ndarray:  # noqa: N802
    """
    Lx, Ly - anistropic variogram length scales
    theta - angle relative to lines of constant latitude
    theta should be radians, and the fitting code outputs radians by default
    """
    R = Rotation.from_rotvec([0, 0, theta])
    R = R.as_matrix()[:2, :2]
    L = np.diag([Lx**2.0, Ly**2.0])
    sigma = R @ L @ R.T
    return sigma


def compute_tau(
    dE: np.ndarray,
    dN: np.ndarray,
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
    return np.sqrt(dx_vec.T @ np.linalg.inv(sigma) @ dx_vec)


def compute_tau_wrapper(dyx: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Wrapper function for computing tau"""
    DE = dyx[:, :, 1]
    DN = dyx[:, :, 0]

    def compute_tau2(dE, dN):
        return compute_tau(dE, dN, sigma)

    compute_tau_vectorised = np.vectorize(compute_tau2)
    return compute_tau_vectorised(DE, DN)


def tau_dist(df: pl.DataFrame) -> np.ndarray:
    """
    Compute the tau/Mahalanobis matrix for all records within a gridbox

    Can be used as an input function for observations.dist_weight.

    Eq.15 in Karspeck paper
    but it is standard formulation to the
    Mahalanobis distance
    https://en.wikipedia.org/wiki/Mahalanobis_distance
    10.1002/qj.900

    Parameters
    ----------
    df : polars.DataFrame
        The observational DataFrame, containing positional information for each
        observation ("lat", "lon"), gridbox specific positional information
        ("gridcell_lat", "gridcell_lon"), and ellipse length-scale parameters
        used for computation of `sigma` ("gridcell_lx", "gridcell_ly",
        "gridcell_theta").

    Returns
    -------
    tau : numpy.matrix
        A matrix of dimension n x n where n is the number of rows in `df` and
        is the tau/Mahalanobis distance.
    """
    required_cols = [
        "gridcell_lat",
        "gridcell_lon",
        "gridcell_lx",
        "gridcell_ly",
        "gridcell_theta",
        "lat",
        "lon",
    ]
    check_cols(df, required_cols)
    # Get northing and easting
    lat0, lon0 = df.select(["gridcell_lat", "gridcell_lon"]).row(0)
    latlons = np.asarray(df.select(["lat", "lon"]).to_numpy())
    ne = latlon2ne(latlons, latlons_in_rads=False, latlon0=(lat0, lon0))
    paired_dist = paired_vector_dist(ne)

    # Get sigma
    Lx, Ly, theta = df.select(
        ["gridcell_lx", "gridcell_ly", "gridcell_theta"]
    ).row(0)
    sigma = Ls2sigma(Lx, Ly, theta)

    tau = compute_tau_wrapper(paired_dist, sigma)
    return np.exp(-tau)


def _tau_unit_test():
    Lx = 1000.0
    Ly = 250.0
    theta = np.pi / 4
    sigma = Ls2sigma(Lx, Ly, theta)
    print(sigma)
    latlon0 = (10.0, -35.0)
    # df = pd.DataFrame({'lat': [7.0, 12.0, -1.0, 20.0],
    #                    'lon': [-32.0, -24.5, -27.0, -40.0]})
    lats = np.linspace(latlon0[0] - 10, latlon0[0] + 10, 21)
    lons = np.linspace(latlon0[1] - 10, latlon0[1] + 10, 21)
    lons2, lats2 = np.meshgrid(lons, lats)
    df = pd.DataFrame(
        {"lat": lats2.flatten().tolist(), "lon": lons2.flatten().tolist()}
    )
    print(df)
    pos = np.asarray(df[["lat", "lon"]].values)
    print(pos)
    pos2 = latlon2ne(pos, latlons_in_rads=False, latlon0=latlon0)
    print(pos2)
    df["northing"] = pos2[:, 0]
    df["easting"] = pos2[:, 1]
    paired_dis_mat = paired_vector_dist(pos2)
    print("dis_Y:")
    print(paired_dis_mat[:, :, 0])
    print("dis_X:")
    print(paired_dis_mat[:, :, 1])
    tau_mat = compute_tau_wrapper(paired_dis_mat, sigma)
    print("tau:")
    print(tau_mat)
    return {"tau": tau_mat, "sigma": sigma, "grid": df}
