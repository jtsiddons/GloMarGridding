from collections.abc import Callable
from dataclasses import dataclass
import numpy as np
import pandas as pd
from warnings import warn

from scipy.spatial.distance import pdist, squareform

from scipy.special import gamma, kv

from math import radians


@dataclass(frozen=True)
class Variogram:
    """Place holder"""

    def fit(self, distance_matrix: np.ndarray) -> np.ndarray:
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

    def fit(self, distance_matrix: np.ndarray) -> np.ndarray:
        """Fit the LinearVariogram model to a distance matrix"""
        return self.slope * distance_matrix + self.nugget


def linear_variogram_model(m, d):
    """
    Linear model
    m is a list containing [slope, nugget]
    d is an array of the distance values at which to calculate the variogram model
    from: Code by Benjamin S. Murphy and the PyKrige Developers
    bscott.murphy@gmail.com
    """
    warn(
        "'linear_variogram_model' is deprecated. Use "
        + "LinearVariogram(slope, nugget).fit(distance_matrix)",
        DeprecationWarning,
    )
    slope = float(m[0])
    nugget = float(m[1])
    return slope * d + nugget


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

    def fit(self, distance_matrix: np.ndarray) -> np.ndarray:
        """Fit the PowerVariogram model to a distance matrix"""
        return (
            self.scale * np.power(distance_matrix, self.exponent) + self.nugget
        )


def power_variogram_model(m, d):
    """
    Power model
    m is a list containing [scale, exponent, nugget]
    d is an array of the distance values at which to calculate the variogram model
    from: Code by Benjamin S. Murphy and the PyKrige Developers
    bscott.murphy@gmail.com
    """
    warn(
        "'power_variogram_model' is deprecated. Use "
        + "PowerVariogram(scale, exponent, nugget).fit(distance_matrix)",
        DeprecationWarning,
    )
    scale = float(m[0])
    exponent = float(m[1])
    nugget = float(m[2])
    return scale * d**exponent + nugget


@dataclass(frozen=True)
class GaussianVariogram(Variogram):
    """
    Gaussian Model

    Parameters
    ----------
    psill : float | np.ndarray
    range : float | np.ndarray
    nugget : float | np.ndarray
    """

    psill: float | np.ndarray
    range: float | np.ndarray
    nugget: float | np.ndarray

    def fit(self, distance_matrix: np.ndarray) -> np.ndarray:
        """Fit the GaussianVariogram model to a distance matrix"""
        return (
            self.psill
            * (
                1.0
                - np.exp(
                    -(
                        np.power(distance_matrix, 2.0)
                        / np.power(self.range * (4 / 7), 2.0)
                    )
                )
            )
            + self.nugget
        )


def gaussian_variogram_model(m, d):
    """
    Gaussian model
    m is a list containing [psill, range, nugget]
    d is an array of the distance values at which to calculate the variogram model
    from: Code by Benjamin S. Murphy and the PyKrige Developers
    bscott.murphy@gmail.com
    """
    warn(
        "'gaussian_variogram_model' is deprecated. Use "
        + "GaussianVariogram(psill, range, nugget).fit(distance_matrix)",
        DeprecationWarning,
    )
    psill = float(m[0])
    range_ = float(m[1])
    nugget = float(m[2])
    return (
        psill * (1.0 - np.exp(-(d**2.0) / (range_ * 4.0 / 7.0) ** 2.0)) + nugget
    )


@dataclass(frozen=True)
class ExponentialVariogram(Variogram):
    """
    Exponential Model

    Parameters
    ----------
    psill : float | np.ndarray
    range : float | np.ndarray
    nugget : float | np.ndarray
    """

    psill: float | np.ndarray
    range: float | np.ndarray
    nugget: float | np.ndarray

    def fit(self, distance_matrix: np.ndarray) -> np.ndarray:
        """Fit the ExponentialVariogram model to a distance matrix"""
        return (
            self.psill * (1.0 - np.exp(-(distance_matrix / (self.range / 3.0))))
            + self.nugget
        )


def exponential_variogram_model(m, d):
    """
    Exponential model
    m is a list containing [psill, range, nugget]
    d is an array of the distance values at which to calculate the variogram model
    from: Code by Benjamin S. Murphy and the PyKrige Developers
    bscott.murphy@gmail.com
    """
    warn(
        "'exponential_variogram_model' is deprecated. Use "
        + "ExponentialVariogram(psill, range, nugget).fit(distance_matrix)",
        DeprecationWarning,
    )
    psill = float(m[0])
    range_ = float(m[1])
    nugget = float(m[2])
    # C(h) = (sill - range) exp (-3 |h|/r), if |h| > 0 #nugget missing here?
    return (
        psill * (1.0 - np.exp(-d / (range_ / 3.0))) + nugget
    )  # this should be the correct version


@dataclass(frozen=True)
class MaternVariogram(Variogram):
    """
    Matern Models

    Parameters
    ----------
    psill : float | np.ndarray
    range : float | np.ndarray
    nugget : float | np.ndarray
    nu : float | np.ndarray
    method : str
        One of "classic", "gstat", or "karspeck"
    """

    psill: float | np.ndarray
    range: float | np.ndarray
    nugget: float | np.ndarray
    nu: float | np.ndarray = 0.5
    method: str = "classic"

    @property
    def _left(self):
        return 1.0 / (gamma(self.nu) * np.power(2.0, self.nu - 1.0))

    def _middle(self, distance_matrix: np.ndarray) -> np.ndarray:
        match self.method.lower():
            case "classic":
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

    def _right(self, distance_matrix: np.ndarray) -> np.ndarray:
        match self.method.lower():
            case "classic":
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

    def fit(self, distance_matrix: np.ndarray) -> np.ndarray:
        """Fit the MaternVariogram model to a distance matrix"""
        return (
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


def matern_variogram_model_classic(m, d, nu=0.5):
    """
    Same args as the *_variogram_model functions
    with additional kwarg for nu/v parameter.

    One can set up lambda or def functions to use different nu values
    that can be used as part of the kwargs of "variogram"; e.g.

    def matern_nu_eq_1p5(m, d):
        # Prefered style for pep/Python style guidelines
        return matern_variogram_model_classic(m, d, nu=1.5)
    or
    # The lazy way, but not recommended by pep style guidelines (does the same thing)
    matern_nu_eq_1p5 = lambda m, d: matern_variogram_model_classic(m, d, nu=1.5)

    1) This is called ``classic'' because if d/range_ = 1.0 and nu=0.5, it gives 1/e correlation...
    2) This is NOT the same formulation as in GSTAT nor in papers about non-stationary anistropic
    covariance models (aka Karspeck paper).
    3) It is perhaps the most intitutive (because of (1)) and is used in sklearn GP and HadCRUT5 and other UKMO dataset.
    4) nu defaults to 0.5 (exponential; used in HADSST4 and our kriging). HadCRUT5 uses 1.5.
    5) The "2" is inside the square root for middle and right.

    Reference; see chapter 4.2 of:
    Rasmussen, C. E., & Williams, C. K. I. (2005).
    Gaussian Processes for Machine Learning. The MIT Press.
    https://doi.org/10.7551/mitpress/3206.001.0001
    """
    warn(
        "'matern_variogram_model_classic' is deprecated. Use "
        + "MaternVariogram(psill, range, nugget, nu, method='classic')"
        + ".fit(distance_matrix)",
        DeprecationWarning,
    )
    psill = float(m[0])
    range_ = float(m[1])
    nugget = float(m[2])
    left = 1.0 / (gamma(nu) * (2.0 ** (nu - 1)))
    middle = (np.sqrt(2.0 * nu) * d / range_) ** nu
    right = kv(nu, np.sqrt(2.0 * nu) * d / range_)
    return psill * (1.0 - left * middle * right) + nugget


def matern_variogram_model_gstat(m, d, nu=0.5):
    """
    Similar to matern_variogram_model_classic
    but uses the range scaling in gstat.
    Note: there are no square root 2 or nu in middle and right

    Yields the same answer to matern_variogram_model_classic if nu==0.5
    but are otherwise different.
    """
    warn(
        "'matern_variogram_model_gstat' is deprecated. Use "
        + "MaternVariogram(psill, range, nugget, nu, method='gstat')"
        + ".fit(distance_matrix)",
        DeprecationWarning,
    )
    psill = float(m[0])
    range_ = float(m[1])
    nugget = float(m[2])
    left = 1.0 / (gamma(nu) * (2.0 ** (nu - 1)))
    middle = (d / range_) ** nu
    right = kv(nu, d / range_)
    return psill * (1.0 - left * middle * right) + nugget


def matern_variogram_model_karspeck(m, d, nu=0.5):
    """
    Similar to matern_variogram_model_classic
    but uses the form in Karspeck paper
    Note: Note the 2 is outside the square root for middle and right
    e-folding distance is now at d/SQRT(2) for nu=0.5
    """
    warn(
        "'matern_variogram_model_karspeck' is deprecated. Use "
        + "MaternVariogram(psill, range, nugget, nu, method='karspeck')"
        + ".fit(distance_matrix)",
        DeprecationWarning,
    )
    psill = float(m[0])
    range_ = float(m[1])
    nugget = float(m[2])
    left = 1.0 / (gamma(nu) * (2.0 ** (nu - 1)))
    middle = (2.0 * np.sqrt(nu) * d / range_) ** nu
    right = kv(nu, 2.0 * np.sqrt(nu) * d / range_)
    return psill * (1.0 - left * middle * right) + nugget


def watermask(ds_masked_xr):
    water_mask = np.copy(ds_masked_xr.values[:, :])
    water_mask[~np.isnan(water_mask)] = 1
    water_mask[np.isnan(water_mask)] = 0
    print(np.shape(water_mask))
    water_idx = np.asarray(
        np.where(water_mask.flatten() == 1)
    )  # this idx is returned as a row-major
    print(water_idx)
    return water_idx


def find_values(ds_masked_xr, lat, lon, timestep):
    """
    Parameters
    ----------
    cci (array) - array of cci  vaules for each point in the whole domain
    lat (array) - array of latitudes of the observations
    lon (array) - array of longitudes of the observations
    df (dataframe) - dataframe containing all information for a given day (such as location, measurement values)

    Returns
    -------
    Dataframe with added anomalies for each observation point
    """
    cci_lat_idx = find_nearest(ds_masked_xr.lat, lat)
    print(ds_masked_xr.lat, cci_lat_idx)
    # problem with this right here (see email to Liz and Richard) - for both lat and lon
    # see: https://stackoverflow.com/questions/40592630/get-coordinates-of-non-nan-values-of-xarray-dataset
    cci_lon_idx = find_nearest(ds_masked_xr.lon, lon)
    print(ds_masked_xr.lon, cci_lon_idx)
    cci_vals = []  # fake ship obs
    for i in range(len(cci_lat_idx)):
        # cci_ = cci[str(ds_var)].values[timestep,cci_lat_idx[i],cci_lon_idx[i]] #was cci.sst_anomaly
        cci_ = ds_masked_xr.values[
            cci_lat_idx[i], cci_lon_idx[i], timestep
        ]  # was cci.sst_anomaly
        cci_vals.append(cci_)
    cci_vals = np.hstack(cci_vals)
    to_remove = np.argwhere(np.isnan(cci_vals))
    # cci_vals = np.delete(cci_vals, to_remove)
    cci_lat_idx = np.delete(cci_lat_idx, to_remove)
    cci_lon_idx = np.delete(cci_lon_idx, to_remove)
    return cci_vals, cci_lat_idx, cci_lon_idx


def find_nearest(array, values):
    array = np.asarray(array)
    idx_list = [(np.abs(array - value)).argmin() for value in values]
    return idx_list


def getDistanceByEuclidean(
    loc1: tuple[float, float],
    loc2: tuple[float, float],
    to_radians=True,
    earth_radius=6371,
) -> float:
    """
    https://math.stackexchange.com/questions/29157/how-do-i-convert-the-distance-between-two-lat-long-points-into-feet-meters
    https://cesar.esa.int/upload/201709/Earth_Coordinates_Booklet.pdf

    d = SQRT((x_2-x_1)**2+(y_2-y_1)**2+(z_2-z_1)**2)
    (x_n y_n z_n) = ( Rcos(lat)cos(lon) Rcos(lat)sin(lon) Rsin(lat) )

    Calculate the Euclidean distance in kilometers between two points
    on the earth (specified in decimal degrees)

    Input: latitude and longitude columns from Pandas dataframe (arrays of values)
    Output: a 2D numpy array of (lat_decimal,lon_decimal) pairs
    """
    lat1 = loc1[0]
    lon1 = loc1[1]
    lat2 = loc2[0]
    lon2 = loc2[1]
    if to_radians:
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    x1 = np.cos(lat1) * np.cos(lon1)
    x2 = np.cos(lat2) * np.cos(lon2)
    y1 = np.cos(lat1) * np.sin(lon1)
    y2 = np.cos(lat2) * np.sin(lon2)
    z1 = np.sin(lat1)
    z2 = np.sin(lat2)
    km = earth_radius * np.sqrt(
        (x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2
    )
    return km


def getDistanceByHaversine(
    loc1: tuple[float, float],
    loc2: tuple[float, float],
    to_radians=True,
    earth_radius=6371,
) -> float:
    """
    Calculate the great circle distance in kilometers between two points
    on the earth (specified in decimal degrees)

    Input: latitude and longitude columns from Pandas dataframe (arrays of values)
    Output: a 2D numpy array of (lat_decimal,lon_decimal) pairs
    """
    # "unpack" our numpy array, this extracts column wise arrays
    lat1 = loc1[0]
    lon1 = loc1[1]
    lat2 = loc2[0]
    lon2 = loc2[1]
    if to_radians:
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )
    km = earth_radius * 2 * np.arcsin(np.sqrt(a))
    return km


def calculate_distance_matrix(df, dist_func=getDistanceByHaversine):
    distance_matrix = pd.DataFrame(
        squareform(pdist(df, lambda u, v: dist_func(u, v))),
        index=df.index,
        columns=df.index,
    )
    return distance_matrix


def variogram(
    distance_matrix,
    variance,
    nugget_=0.0,
    range_=350.0,
    variogram_model=exponential_variogram_model,
):
    # range from Dave's presentation on space length scales (in km)
    # range_ = 350
    # from researchgate - Sill of the semivariogram is equal to the variance of the random variable, at large distances, when variables become uncorrelated
    sill_ = variance
    # nugget for now can be set to zero, it will change once we quantify the obs uncertainty better
    # nugget_ = 0

    # create m - a list containing [psill, range, nugget]
    m = [sill_, range_, nugget_]

    # create d - an array of the distance values at which to calculate the variogram model
    d = distance_matrix

    # call variogram function
    # this calculates the covariance between the observations only (equivalemnt in Simon's code is "S" martix)
    obs_covariance = variogram_model(m, d)
    return obs_covariance


def variogram_hadcrut5(
    distance_matrix, terrain="lnd", nugget_=0.0, method: str = "classic"
):
    if terrain not in ["lnd", "sea"]:
        raise ValueError("terrain must be lnd or sea.")
    hadcrut5_covariance_parms = {
        "lnd": {"sigma": 1.2, "r": 1300.0, "v": 1.5},
        "sea": {"sigma": 0.6, "r": 1300.0, "v": 1.5},
    }
    variogram_parms = hadcrut5_covariance_parms[terrain]

    return MaternVariogram(
        psill=np.power(variogram_parms["sigma"], 2.0),
        range=variogram_parms["r"],
        nugget=nugget_,
        nu=variogram_parms["v"],
        method=method,
    ).fit(distance_matrix)


def variogram_to_covariance(variogram, variance):
    return variance - variogram


"""
######################
# read in the observational data
######################
filename = pd.read_csv(sys.argv[1]) # netcdf file #/noc/users/agfaul/data_nc/MAT_1820_7.nc
lats = filename['lat'].values #np.asarray(nc.Dataset(filename).variables['lat']) #it's a 1-D array based on the psv file
lons = filename['lon'].values #np.asarray(nc.Dataset(filename).variables['lon']) #it's a 1-D array based on the psv file
#coords = list(zip(list(lats),list(lons)))
vals = filename['anomaly'].values #np.asarray(nc.Dataset(filename).variables['anomaly']) #it's a 1-D array based on the psv file

print('LATS', lats, len(lats))
print('LONS', lons, len(lons))
#print('COORDS', coords)
print('VALS', vals, len(vals))



##################
# compute pairwise distances
##################

# distance matrix using pandas
data = {'lat': lats, 'lon': lons}
df = pd.DataFrame(data, columns=['lat', 'lon'])

distance_matrix = pd.DataFrame(squareform(pdist(df, lambda u, v: getDistanceByHaversine(u,v))), index=df.index, columns=df.index)

print('Distance matrix \n', distance_matrix)


#####################
# compute the raw variogram values
#####################
dissim = np.abs(vals[:, None] - vals[None, :])**2 / 2
dissim_ = squareform(dissim)

variance = np.var(vals)
print('VARIANCE', variance)


#####################
# set nugget, sill and range
#####################

#range from Dave's presentation on space length scales (in km)
range_ = 350 
#from researchgate - Sill of the semivariogram is equal to the variance of the random variable, at large distances, when variables become uncorrelated
sill_ = variance 
#nugget for now can be set to zero, it will change once we quantify the obs uncertainty better
nugget_ = 0 

#create m - a list containing [psill, range, nugget]
m = [sill_, range_, nugget_]

#create d - an array of the distance values at which to calculate the variogram model
d = distance_matrix

#call variogram function
#this calculates the covariance between the observations only (equivalemnt in Simon's code is "S" martix
obs_covariance = exponential_variogram_model(m, d)

print(variogram)
print(variogram.shape)


######################
# set up a global grid for distance matrix computation
#####################

#create global grid for finding prediction points                                                          
global_lats = np.arange(-90., 90.,1.)
global_lons = np.arange(-180., 180.,1.)

#calculate the covariance between the observation points and the prediction points (equivlent in Simon's code is the "Ss" matrix

global_data = {'lat': global_lats, 'lon': global_lons}
global_df = pd.DataFrame(global_data, columns=['lat', 'lon'])
global_distance_matrix = pd.DataFrame(distance_matrix(global_df.values, global_df.values), index=global_df.index, columns=global_df.index)
print('global distance_matrix \n', global_distance_matrix)

d_global = global_distance_matrix
global_covariance = exponential_variogram_model(m, d_global)


timestep = 0
ds_masked_xr = xr.DataArray(ds_masked, coords=cci_pentad_files.coords, dims=cci_pentad_files.dims, attrs=cci_pentad_files.attrs)
#on a global 1-D grid find the index of where the observations are
iid = watermask(ds_masked_xr,timestep)
#extract the CCI SST anomaly values corresponding to the shop lat/lon coordinate points                       
#fake_ship_obs = find_nearest(ds.lat.values, ship_lat)                                                      
fake_ship_obs, lat_idx, lon_idx = find_values(ds_masked_xr, ship_lat, ship_lon, timestep)
cond_df['cci_anomalies'] = fake_ship_obs
#in case some of the values are Nan (because covered by ice)                                                
cond_df = cond_df.dropna()
idx_tuple = np.array([lat_idx, lon_idx])
#output_size = (ds[str(ds_var)].values[timestep,:,:]).shape                                                 
output_size = (ds_masked_xr.values[:,:,timestep]).shape
flattened_idx = np.ravel_multi_index(idx_tuple, output_size, order='C') #row-major; order='F') column-major
uind = np.unique(flattened_idx)

_,ia,_ = intersect_mtlb(iid,uind)
obs_unknown_covariance = global_covariance[ia,:]

#the rest should be the same in kriging
#idea: could try the kriging code with weights provided by the https://gis.stackexchange.com/questions/270274/how-to-calculate-kriging-weights/301362#301362 and see how different they are
"""
