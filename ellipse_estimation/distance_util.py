"""
Functions copied from cube_covariance_nonstationary_stich
Will also be used in future fit testing and ellipse simulation code
"""

import iris
import numpy as np
import logging
from sklearn.metrics.pairwise import haversine_distances

from glomar_gridding.constants import RADIUS_OF_EARTH_KM
from glomar_gridding.types import DELTA_X_METHOD


def haversine_single(
    lon1,
    lat1,
    lon2,
    lat2,
):
    """
    Calculate the great circle distance between two points
    (specified in decimal degrees)

    All args must be of equal length.

    https://stackoverflow.com/questions/29545704/fast-haversine-approximation-python-pandas

    This is an alternative to sklearn version, and can run faster under some
    circumstances

    If the inputs are vectors, the output will also be a vector the same length
    as the inputs. This function will not generate a full distance matrix, only
    row-wise distances.

    Note: the result has not been multipled by the radius of Earth yet, need to
    do that to convert that to great circle distance on Earth

    Parameters
    ----------
    lon1, lat1 : float or np.ndarray
        lon and lat of point 1 in degrees
    lon2, lat2 : float or np.ndarray
        lon and lat of point 2 in degrees

    Returns
    -------
    c : float or np.ndarray
        Great circle distance distance between 1 and 2
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    def hf(rad):
        return (np.sin(rad / 2)) ** 2

    a = hf(dlat) + np.cos(lat1) * np.cos(lat2) * hf(dlon)
    c = 2 * np.arcsin(np.sqrt(a))
    return c


def scalar_cube_great_circle_distance(
    lat_i: float,
    lon_i: float,
    lat_j: float,
    lon_j: float,
    degree_dist: bool = False,
    delta_x_method: DELTA_X_METHOD = "Modified_Met_Office",
    use_sklearn_haversine=False,
) -> tuple[float, float, float]:
    """
    A number of ways to approximate the vectored displacement in
    physical distances on the surface of Earth.

    Coordinates of the two points
    lat_i, lon_i
    lat_j, lon_j

    if degree_dist == True, it just returns the delta degree in latlon
    including Euclidean distance

    delta_x_method = "Met_Office":
        Cylindrical earth: delta_x = 6400km x delta_lon (in radians)

    delta_x_method = "Modified_Met_Office":
        A bit better, but uses the average zonal dist at different lat

    from i to j : i.e. dist = j minus i

    About dist_j, dist_i:

    This is the hard part...

    We prefer distances in physical distance.

    Met Office paper uses a "cylinder" approximation:
    https://doi.org/10.1029/2018JD029867

    For lat, follow lines of constant lon meaning delta_y = delta_lat x R_earth
    For lon, same as lat WITHOUT lat correction: delta_x = delta_lon x R_earth
    which is questionable for high latitudes (but would be okay in the tropics.

    Another approximation based on the Met Office approach is to
    actually allow delta_x to depend somewhat on lat. However, there are
    two ways you can move on lon - either on higher lat (shorter) or the
    equatorward side (longer).
    So you can take the average of the start and end lat

    delta_x = 0.5 x R_Earth x (COS(LAT_0)+COS(LAT_1)) x delta_lon

    Parameters
    ----------
    lat_i, lon_i : float
        Lat Lon of point 1

    lat_j, lon_j :
        Lat Lon of point 2

    degree_dist : bool (default False)
        Compute distances by degrees (no more Earth being round problem duh)

    delta_x_method : str (default="Modified_Met_Office")
        Met_Office :
            tin can (cylindrical) Earth, used in HadSST4
            Distance following down line constant lon (y-th component)
            Distance following across line constant lat,
            assuming NO variation to it by latitude (x-th component)

        Modified_Met_Office :
            squished tin can (Sinusoidal projection) Earth
            Distance following down line constant lon (y-th component)
            Average distance following down line constant lat between
            the starting and ending latitude with sine lat
            correction (x-th component)
            This differs with "Met Office" only in how
            the x-th component is computed

    use_sklearn_haversine : bool (default False)
        use sklearn.metrics.pairwise.haversine_distances or haversine2
        (False means use haversine2)
        This is very sensentive to platform.
        In NOC machines, it does not matter.
        JASMIN installation of sklearn appears to be much slower
        The actual sklearn src is not very complicated,
        but it enforces tough type
        checks and assertations... which somehow JASMIN does not like

    Returns
    -------
    dist : float
        Haversine distance between the two points
        (non directional non-vectored)
    disp_j :
        The y/j-th component of the displacement
    disp_i :
        The x/i-th component of the displacement
    """
    lon_i = lon_i - 360.0 if lon_i > 180.0 else lon_i
    lon_j = lon_j - 360.0 if lon_j > 180.0 else lon_j
    delta_lat = lat_j - lat_i
    delta_lon = lon_j - lon_i
    delta_lon = delta_lon - 360.0 if delta_lon > 180.0 else delta_lon
    delta_lon = delta_lon + 360.0 if delta_lon < -180.0 else delta_lon
    # dx_sign = np.sign(delta_lon)
    if degree_dist:
        # returns Euclidean and delta_lat and lon if degree_dist == True
        return np.sqrt(delta_lon**2.0 + delta_lat**2.0), delta_lat, delta_lon

    ys = np.array([np.deg2rad(lat_i), np.deg2rad(lat_j)])
    xs = np.array([np.deg2rad(lon_i), np.deg2rad(lon_j)])
    yxs = np.column_stack([ys, xs])
    logging.debug(f"From: {np.rad2deg(yxs[0, :])}")
    logging.debug(f"To  : {np.rad2deg(yxs[1, :])}")

    # Bottleneck parm_check sklearn
    if use_sklearn_haversine:
        # Slower - sklearn has more overhead in checks
        dist = haversine_distances(yxs)[0, 1]
    else:
        # 40% faster for 261 pairs; does not fix joblib speed issues
        dist = haversine_single(lon_i, lat_i, lon_j, lat_j)
    disp_j = yxs[1, 0] - yxs[0, 0]
    logging.debug(f"Haversine = {dist} [Radians]")
    logging.debug(f"Haversine = {np.rad2deg(dist)} [Degrees]")
    logging.debug(f"Delta Lat = {disp_j} [Radians]")
    logging.debug(f"Delta Lat = {np.rad2deg(disp_j)} [Degrees]")

    if delta_x_method == "Met_Office":
        disp_i = _get_disp_i(yxs)
    elif delta_x_method == "Modified_Met_Office":
        disp_i = _get_disp_i(yxs)
        average_cos = 0.5 * (np.cos(yxs[1, 0]) + np.cos(yxs[0, 0]))
        disp_i = disp_i * average_cos
    else:
        raise ValueError("Unknown delta_x_method")

    dist = dist * RADIUS_OF_EARTH_KM
    disp_j = disp_j * RADIUS_OF_EARTH_KM
    disp_i = disp_i * RADIUS_OF_EARTH_KM
    # (great circle dist, lat displacement, zonal displacement)
    return dist, disp_j, disp_i


def _get_disp_i(yxs):
    disp_i = yxs[1, 1] - yxs[0, 1]
    if disp_i > np.pi:
        disp_i = disp_i - 2.0 * np.pi
    elif disp_i < -np.pi:
        disp_i = disp_i + 2.0 * np.pi
    return disp_i


def scalar_cube_great_circle_distance_cube(
    scalar_cube_i: iris.cube.Cube,
    scalar_cube_j: iris.cube.Cube,
    degree_dist: bool = False,
    delta_x_method: DELTA_X_METHOD = "Modified_Met_Office",
) -> tuple[float, float, float]:
    """
    Wrapper for scalar_cube_great_circle_distance but allows input
    as iris scalar cube with latlons

    Takes iris cubes, xarray doesn't have coord so will need wrapper function
    but xarray cubes can be converted to iris cubes
    from i to j : i.e. dist = j minus i

    Parameters
    ----------
    scalar_cube_i : iris.cube.Cube
        scalar cube at point 1
    scalar_cube_j : iris.cube.Cube
        scalar cube at point 2
    degree_dist : bool
        See scalar_cube_great_circle_distance
    delta_x_method : str
        See scalar_cube_great_circle_distance

    Returns
    -------
    dist : float
        Haversine distance between the two points
        (non directional non-vectored)
    disp_j : float
        The y/j-th component of the displacement
    disp_i : float
        The x/i-th component of the displacement
    """
    if (len(scalar_cube_i.coord("latitude").points) != 1) or (
        len(scalar_cube_i.coord("longitude").points) != 1
    ):  # noqa: E501
        raise ValueError("Scalar cubes only (i)")
    if (len(scalar_cube_j.coord("latitude").points) != 1) or (
        len(scalar_cube_j.coord("longitude").points) != 1
    ):  # noqa: E501
        raise ValueError("Scalar cubes only (j)")
    lat_i = float(scalar_cube_i.coord("latitude").points)
    lon_i = float(scalar_cube_i.coord("longitude").points)
    lat_j = float(scalar_cube_j.coord("latitude").points)
    lon_j = float(scalar_cube_j.coord("longitude").points)
    return scalar_cube_great_circle_distance(
        lat_i,
        lon_i,
        lat_j,
        lon_j,
        degree_dist=degree_dist,
        delta_x_method=delta_x_method,
    )
