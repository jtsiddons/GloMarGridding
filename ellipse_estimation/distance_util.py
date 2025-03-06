'''
Functions copied from cube_covariance_nonstationary_stich
Will also be used in future fit testing and ellipse simulation code
'''

import numpy as np
from sklearn import metrics as skl_metrics

from ellipse_estimation import cube_covariance as cube_cov

def haversine2(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    (specified in decimal degrees)

    All args must be of equal length.

    https://stackoverflow.com/questions/29545704/fast-haversine-approximation-python-pandas

    This is an alternative to sklearn version, and can run faster under some circumstances

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

    Note: It HAS NOT multipled the radius of Earth yet, need to do that
    to convert that to great circle distance on Earth
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    def hf(rad):
        return (np.sin(rad/2))**2
    a = hf(dlat) + np.cos(lat1) * np.cos(lat2) * hf(dlon)
    c = 2 * np.arcsin(np.sqrt(a))
    return c


def scalar_cube_great_circle_distance(lat_i, lon_i,
                                      lat_j, lon_j,
                                      degree_dist=False,
                                      delta_x_method="Modified_Met_Office",
                                      use_sklearn_haversine=False,
                                      verbose=False):
    '''
    A number of ways to approximate the vectored displacement in 
    physical distances on the surface of Earth, noting that Earth
    is round (!)

    Coordinates of the two points
    lat_i, lon_i
    lat_j, lon_j

    if degree_dist == True, it just returns the delta degree in latlon including Euclidean distance

    delta_x_method = "Spherical_COS_Law": uses COS(C) = COS(A)COS(B)
    delta_x_method = "Met_Office": Suspected flat (cylindrical) Earthers in Exeter! delta_x = 6400km x delta_lon (in radians)
    delta_x_method = "Modified_Met_Office": A bit better, but uses the average zonal dist at different lat

    from i to j : i.e. dist = j minus i

    About dist_j, dist_i:

    This is the hard part...

    We prefer distances in physical distance.

    Spherical law of cosines pose a lot of challenge in use of a Earth coordinate system
    that is based on lat lon, you can say you walk in some direction for delta_lat x R_earth and
    turn 90 degree towards to the other point, but you are actually not following lines
    of constant longitude. A 90 degree directly toward one of the points
    are NOT the same as turning from lines of constant lat and constant lon.

    Pythograean version of spherical law of cosines:
    COS(GREAT_CIRCLE_DIST_IN_RADIANS) = COS(A) COS(B) provided angle between A and B = pi/2

    https://en.wikipedia.org/wiki/Spherical_law_of_cosines

    Met Office paper uses a "cylinder" approximation:
    https://doi.org/10.1029/2018JD029867

    For lat, follow lines of constant lon meaning delta_y = delta_lat x R_earth
    For lon, same as lat WITHOUT lat correction: delta_x = delta_lon x R_earth
    which is questionable for high latitudes (but would be okay in the tropics.

    A less "bad" approximation based on the Met Office approach is to actually allow delta_x to depend
    somewhat on lat. However, there are two ways you can move on lon - either on higher lat (shorter) or the
    equatorward side (longer). So you can take the average of the start and end lat?

    delta_x = 0.5 x R_Earth x (COS(LAT_0)+COS(LAT_1)) x delta_lon

    Spherical coordinates, geosedics, and haversines are yuck!

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
            the starting and ending latitude with sine lat correction (x-th component)
            This differs with "Met Office" only in how the x-th component is computed

    use_sklearn_haversine : bool (default False)
        use sklearn.metrics.pairwise.haversine_distances or haversine2 
        (False means use haversine2)
        This is very sensentive to platform. 
        In NOC machines, it does not matter.
        JASMIN installation of sklearn appears to be much slower
        The actual sklearn src is not very complicated, but it enforces tough type
        checks and assertations... which somehow JASMIN does not like

    verbose : bool (default False)
        More stuff get printed out on stdout

    Returns
    -------
        dist : float
            Haversine distance between the two points (non directional non-vectored)
        dist_j : 
            The y/j-th component of the displacement 
        dist_i :
            The x/i-th component of the displacement 
    '''
    lon_i = lon_i - 360.0 if lon_i > 180.0 else lon_i
    lon_j = lon_j - 360.0 if lon_j > 180.0 else lon_j
    delta_lat = lat_j - lat_i
    delta_lon = lon_j - lon_i
    delta_lon = delta_lon - 360.0 if delta_lon > 180.0 else delta_lon
    delta_lon = delta_lon + 360.0 if delta_lon < -180.0 else delta_lon
    # dx_sign = np.sign(delta_lon)
    if degree_dist:
        # returns Euclidean and delta_lat and lon if degree_dist == True
        return (np.sqrt(delta_lon**2.0 + delta_lat**2.0), delta_lat, delta_lon)
    ##
    ys = np.array([np.deg2rad(lat_i), np.deg2rad(lat_j)])
    xs = np.array([np.deg2rad(lon_i), np.deg2rad(lon_j)])
    yxs = np.column_stack([ys, xs])
    if verbose:
        print('From: ', np.rad2deg(yxs[0, :]))
        print('To  : ', np.rad2deg(yxs[1, :]))
    #
    # Bottleneck parm_check sklearn
    if use_sklearn_haversine:
        # Slower - sklearn has more overhead in checks
        dist0 = skl_metrics.pairwise.haversine_distances(yxs)[0, 1]
    else:
        # 40% faster for 261 pairs; does not fix joblib speed issues
        dist0 = haversine2(lon_i, lat_i, lon_j, lat_j)
    dy0 = yxs[1, 0] - yxs[0, 0]
    if verbose:
        print('Haversine = ', dist0, '[Radians]')
        print('Haversine = ', np.rad2deg(dist0), '[Degrees]')
        print('Delta Lat = ', dy0, '[Radians]')
        print('Delta Lat = ', np.rad2deg(dy0), '[Degrees]')

    if delta_x_method == "Met_Office":
        dx0 = yxs[1, 1] - yxs[0, 1]
        if dx0 > np.pi:
            dx0 = dx0 - 2.0*np.pi
        elif dx0 < -np.pi:
            dx0 = dx0 + 2.0*np.pi
        else:
            pass
    elif delta_x_method == "Modified_Met_Office":
        dx0 = yxs[1, 1] - yxs[0, 1]
        if dx0 > np.pi:
            dx0 = dx0 - 2.0*np.pi
        elif dx0 < -np.pi:
            dx0 = dx0 + 2.0*np.pi
        else:
            pass
        average_cos = 0.5 * (np.cos(yxs[1, 0]) + np.cos(yxs[0, 0]))
        dx0 = dx0 * average_cos
    else:
        raise ValueError('Unknown delta_x_method')
    ##
    dist = dist0 * cube_cov._RADIUS_OF_EARTH / cube_cov._KM2M
    dist_j = dy0 * cube_cov._RADIUS_OF_EARTH / cube_cov._KM2M
    dist_i = dx0 * cube_cov._RADIUS_OF_EARTH / cube_cov._KM2M
    return (dist, dist_j, dist_i)  # (great circle dist, lat displacement, zonal displacement)


def scalar_cube_great_circle_distance_cube(scalar_cube_i,
                                           scalar_cube_j,
                                           degree_dist=False,
                                           delta_x_method="Modified_Met_Office"):
    '''
    Wrapper for scalar_cube_great_circle_distance but allows input as iris scalar cube with latlons
    
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
            Haversine distance between the two points (non directional non-vectored)
        dist_j : 
            The y/j-th component of the displacement 
        dist_i :
            The x/i-th component of the displacement 

    Same as in scalar_cube_great_circle_distance
    '''
    if (len(scalar_cube_i.coord('latitude').points) != 1) or (len(scalar_cube_i.coord('longitude').points) != 1):
        raise ValueError('Scalar cubes only (i)')
    if (len(scalar_cube_j.coord('latitude').points) != 1) or (len(scalar_cube_j.coord('longitude').points) != 1):
        raise ValueError('Scalar cubes only (j)')
    lat_i = float(scalar_cube_i.coord('latitude').points)
    lon_i = float(scalar_cube_i.coord('longitude').points)
    lat_j = float(scalar_cube_j.coord('latitude').points)
    lon_j = float(scalar_cube_j.coord('longitude').points)
    ans = scalar_cube_great_circle_distance(lat_i, lon_i, lat_j, lon_j, degree_dist=degree_dist, delta_x_method=delta_x_method)
    return ans  # (great circle dist, lat displacement, zonal displacement)


def main():
    ''' === Main === '''
    print('=== Main ===')


if __name__ == "__main__":
    main()
