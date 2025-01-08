################
# by A. Faulkner
# for python version 3.0 and up
################

# global
import os
import re
from warnings import warn

# math tools
import numpy as np
import math

# data handling tools
import pandas as pd
import polars as pl
import xarray as xr

from sklearn.metrics.pairwise import haversine_distances
from collections.abc import Callable, Iterable

from matern_and_tm import tau_dist
from .utils import check_cols, find_nearest, select_bounds


_MONTH_31: list[int] = [1, 3, 5, 7, 8, 10, 12]


# QUESTION: What is `sic`?
def extract_sic(sic: xr.DataArray, df: pd.DataFrame) -> pd.DataFrame:
    obs_lat = np.array(df["lat"])
    obs_lon = np.array(df["lon"])

    sic_lat_idx, _ = find_nearest(sic.latitude, obs_lat)
    sic_lon_idx, _ = find_nearest(sic.longitude, obs_lon)

    # sic = sic.values  # variables['climatology'].values
    # print(esa_cci_clim)

    # sic_vals = []  # fake ship obs
    # for i in range(len(sic_lat_idx)):
    #     c = sic[sic_lat_idx[i], sic_lon_idx[i]]
    #     sic_vals.append(c)
    # sic_vals = np.hstack(sic_vals)
    sic_vals = [sic.values[j, i] for j, i in zip(sic_lat_idx, sic_lon_idx)]
    updated_df = df.copy()
    updated_df["sic"] = sic_vals
    return updated_df


def landmask(
    filepath: str,
    min_lat: float = -90,
    max_lat: float = 90,
    min_lon: float = -180,
    max_lon: float = 180,
    lon_var: str = "lon",
    lat_var: str = "lat",
) -> tuple[xr.Dataset, np.ndarray, np.ndarray]:
    # extract land sea mask for the domain
    mask_ds = xr.open_dataset(filepath, engine="netcdf4")
    mask_ds = select_bounds(
        mask_ds, (min_lon, max_lon), (min_lat, max_lat), lon_var, lat_var
    )
    lon = mask_ds.coords[lon_var].values
    lat = mask_ds.coords[lat_var].values
    lon = ((lon + 540.0) % 360.0) - 180.0
    mask_ds.coords[lon_var] = lon
    return mask_ds, lon, lat
    # print(mask_ds)
    #
    # # esa_cci_mask has the same amount of water points as the covariance created
    # # from ESA data
    # try:
    #     print("landmask")
    #     mask_ds = mask_ds.sel(
    #         lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon)
    #     )
    #     # mask = mask_ds.variables["landmask"].values
    #     lat = mask_ds.lat.values
    #     lon = mask_ds.lon.values
    # except KeyError:
    #     print("land_sea_mask")
    #     mask_ds.coords["longitude"] = (
    #         (mask_ds.coords["longitude"] + 540.0) % 360.0
    #     ) - 180.0
    #
    #     mask_ds = mask_ds.sel(
    #         latitude=slice(min_lat, max_lat), longitude=slice(min_lon, max_lon)
    #     )
    #     # mask = mask_ds.variables["land_sea_mask"].values
    #     # mask = mask_ds.variables['landice_sea_mask'].values
    #     lat = mask_ds.latitude.values
    #     lon = mask_ds.longitude.values
    #     print(lon)
    #     lon = ((lon + 540.0) % 360.0) - 180.0
    #     print(lon)
    #     # water is 1, land is 0
    #     mask_ds.coords["longitude"] = lon
    #     print(mask_ds.latitude.values)
    #     print(mask_ds.longitude.values)
    #
    # # gridlon, gridlat = np.meshgrid(esa_cci_lon, esa_cci_lat)
    # return mask_ds, lat, lon


def read_climatology(
    clim_path: str,
    min_lat: float = -90,
    max_lat: float = 90,
    min_lon: float = -180,
    max_lon: float = 180,
    lon_var: str = "lon",
    lat_var: str = "lat",
) -> xr.Dataset:
    clim_ds = xr.open_dataset(clim_path, engine="netcdf4")
    clim_ds = select_bounds(
        clim_ds, (min_lon, max_lon), (min_lat, max_lat), lon_var, lat_var
    )
    return clim_ds


def read_daily_sst_climatology(
    clim_path: str,
    min_lat: float = -90,
    max_lat: float = 90,
    min_lon: float = -180,
    max_lon: float = 180,
    lon_var: str = "lon",
    lat_var: str = "lat",
) -> xr.Dataset:
    filelist = sorted(os.listdir(clim_path))  # _fullpath(dirname)
    print(filelist)
    r = re.compile(r"D\S+.nc")
    filtered_list = list(filter(r.match, filelist))
    print(filtered_list)
    fullpath_list = [os.path.join(clim_path, f) for f in filtered_list]
    clim_ds = xr.open_mfdataset(
        fullpath_list, combine="nested", concat_dim="time", engine="netcdf4"
    )
    print(clim_ds)
    clim_ds = select_bounds(
        clim_ds, (min_lon, max_lon), (min_lat, max_lat), lon_var, lat_var
    )
    print("----------------------")
    print(clim_ds)
    return clim_ds


def days_in_pentad(
    pentad: int,
    pentad_type: str,
    month: int | None = None,
) -> list[int]:
    match pentad_type:
        case "standard":
            return [i + 5 * pentad for i in range(1, 6)]
        case "esa":
            if month is None:
                raise ValueError(
                    "month must not be None if pentad_type is 'esa'"
                )
            dom = (pentad - 1) * 5 + 1
            day_list = [dom + i for i in range(5)]
            if pentad == 6:
                if month == 2:
                    day_list = day_list[:3]
                if month in _MONTH_31:
                    day_list.append(31)
            return day_list
        case _:
            raise ValueError("Unknown pentad type")


def read_daily_climatology(
    clim_path: str,
    doy: int,
    time_type: str,
    month: int | None = None,
    pentad: int | None = None,
) -> xr.Dataset:
    filelist = sorted(os.listdir(clim_path))  # _fullpath(dirname)
    print(filelist)
    match time_type.lower():
        case "pentad":
            pent = doy // 5
            day_list = [i + 5 * pent for i in range(1, 6)]
            # day_list = [doy-2, doy-1, doy, doy+1, doy+2]
            str_list = [
                "D"
                + str(i).zfill(3)
                + "-ESACCI-L4_GHRSST-SSTdepth-OSTIA-GLOB_CDR2.1-v02.0-fv01.0.nc"
                for i in day_list
            ]
            print(day_list)
            print(str_list)
            filtered_list = [i for i in filelist if i in str_list]
            print(filtered_list)

            fullpath_list = [os.path.join(clim_path, f) for f in filtered_list]
            clim_file = xr.open_mfdataset(
                fullpath_list,
                combine="nested",
                concat_dim="time",
                engine="netcdf4",
            )
            print(clim_file)

        case "esa_pentad":
            if month == 2 and pentad == 6:
                day_list = [doy - 2, doy - 1, doy]
            elif pentad == 6 and month in [1, 3, 5, 7, 8, 10, 12]:
                day_list = [doy - 2, doy - 1, doy, doy + 1, doy + 2, doy + 3]
            else:
                day_list = [doy - 2, doy - 1, doy, doy + 1, doy + 2]

            print(pentad)
            print(day_list)
            str_list = [
                "D"
                + str(i).zfill(3)
                + "-ESACCI-L4_GHRSST-SSTdepth-OSTIA-GLOB_CDR2.1-v02.0-fv01.0.nc"
                for i in day_list
            ]
            print(day_list)
            print(str_list)
            filtered_list = [i for i in filelist if i in str_list]
            print(filtered_list)
            fullpath_list = [os.path.join(clim_path, f) for f in filtered_list]
            clim_file = xr.open_mfdataset(
                fullpath_list,
                combine="nested",
                concat_dim="time",
                engine="netcdf4",
            )
            # print(clim_file)

        case "daily":
            r = re.compile("D" + str(doy).zfill(3) + r"\S+.nc")
            filtered_list = list(filter(r.match, filelist))
            print(filtered_list)

            fullpath_list = [os.path.join(clim_path, f) for f in filtered_list]
            clim_file = xr.open_dataset(fullpath_list[0], engine="netcdf4")
            # print(clim_file)

        case _:
            raise ValueError("Unknown time_type value")

    return clim_file


def extract_clim_anom(
    clim_array: xr.DataArray,
    df: pd.DataFrame,
    var_col: str = "sst",
    is_temperature: bool = True,
    clim_lat_name: str = "lat",
    clim_lon_name: str = "lon",
) -> pd.DataFrame:
    """
    Merge a climatology to an observational dataframe and compute an anomaly
    against that climatology for an observed variable.

    Parameters
    ----------
    clim_array : xarray.DataArray
        xarray DataArray containing the climatology values
    df : pandas.DataFrame
        Observational DataFrame containing the values we want to compute the
        anomaly for.
    var_col : str
        Variable column name in the observational DataFrame to compute the
        anomaly from. Is used to name the additional columns in the output
        DataFrame. The anomaly column will be "`var_col`_anomaly" and the
        climatology values are joined as "climatology_`var_col`".
    is_temperature : bool
        Ensure temperature values are converted from degrees Kelvin to
        degrees Celcius
    clim_lat_name : str
        Name of the latitude coordinate in the climatology DataArray.
    clim_lon_name
        Name of the longitude coordinate in the climatology DataArray.

    Returns
    -------
    df : pandas.DataFrame
        The input observational DataFrame with additional climatology and
        anomaly columns for the `var_col` input, named using the variable
        name: The anomaly column will be "`var_col`_anomaly" and the
        climatology values are joined as "climatology_`var_col`".
    """
    obs_lat = np.array(df["lat"])
    obs_lon = np.array(df["lon"])

    clim_lat_idx, _ = find_nearest(clim_array[clim_lat_name], obs_lat)
    clim_lon_idx, _ = find_nearest(clim_array[clim_lon_name], obs_lon)

    climatology = np.asarray(
        [clim_array.values[j, i] for j, i in zip(clim_lat_idx, clim_lon_idx)]
    )

    # for i in range(len(cci_lat_idx)):
    #     c = esa_cci_clim[cci_lat_idx[i], cci_lon_idx[i]]
    #     climatology.append(c)
    # climatology = np.hstack(climatology)
    if is_temperature and (climatology > 200).any():
        # Degrees Kelvin to Degrees Celcius
        climatology = climatology - 273.15
    # print(climatology)
    updated_df: pd.DataFrame = df.copy()
    updated_df[f"climatology_{var_col}"] = climatology

    if pd.api.types.is_string_dtype(updated_df.sst.dtype):
        updated_df[var_col] = pd.to_numeric(
            updated_df[var_col], errors="coerce"
        )

    updated_df[f"{var_col}_anomaly"] = (
        updated_df[var_col] - updated_df[f"climatology_{var_col}"]
    )
    updated_df = updated_df.loc[updated_df[f"{var_col}_anomaly"].notna()]

    return updated_df


# WARN: seems to be trying to apply a mask to the positions in the observation
#       frame - concerns mask positions may be far from data, may get incorrect
#       masking
# INFO: Maybe improve by first adjusting obs to a grid resolution matched by the
#       Mask?
def find_values(
    mask_ds: xr.Dataset,
    lat: Iterable[float],
    lon: Iterable[float],
    mask_var: str = "landmask",
    lon_var: str = "lon",
    lat_var: str = "lat",
) -> tuple[np.ndarray, list[int], list[int], np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    cci (array) - array of cci  vaules for each point in the whole domain
    lat (array) - array of latitudes of the observations
    lon (array) - array of longitudes of the observations
    df (dataframe) - dataframe containing all information for a given day (such
    as location, measurement values)

    Returns
    -------
    Dataframe with added anomalies for each observation point
    """
    mask = mask_ds.variables[mask_var].values
    # print(mask)
    mask_lat = mask_ds.coords[lat_var].values
    # print(mask_lat)
    mask_lon = mask_ds.coords[lon_var].values
    mask_lon = ((mask_lon + 540) % 360) - 180

    # Find nearest mask lat/lon value to each observation lat/lon
    # BUG: Mask set if nearest is quite far away
    # QUESTION: Can we guarantee that the mask is _complete_ and _uniform_?
    mask_lat_idx, grid_lat = find_nearest(mask_lat, lat)
    mask_lon_idx, grid_lon = find_nearest(mask_lon, lon)

    # esa_cci_mask = mask
    # water_point = []
    masked: np.ndarray = np.array(
        [mask[j, i] for j, i in zip(mask_lat_idx, mask_lon_idx)]
    )
    # for i in range(len(cci_lat_idx)):
    #     wp = esa_cci_mask[cci_lat_idx[i], cci_lon_idx[i]]
    #     # print(wp)
    #     # cci_ = ds_masked_xr.values[cci_lat_idx[i],cci_lon_idx[i],
    #     #                            cci_time_idx[i]] #was cci.sst_anomaly
    #     water_point.append(wp)
    # water_point = np.hstack(water_point)

    return masked, mask_lat_idx, mask_lon_idx, grid_lat, grid_lon


# OPTIM: Can we speed this up by looking at unique positions and merging?
def watermask_at_obs_locations(
    lon_bnds: tuple[float, float],
    lat_bnds: tuple[float, float],
    df: pd.DataFrame,
    mask_ds: xr.Dataset,
    # mask_ds_lat: np.ndarray,
    # mask_ds_lon: np.ndarray,
) -> tuple[pd.DataFrame, list[int]]:
    obs_data = df
    # remove ship obs that are outside the chosen domain
    cond_df = obs_data.loc[
        (obs_data["lon"] >= lon_bnds[0])
        & (obs_data["lon"] < lon_bnds[1])
        & (obs_data["lat"] >= lat_bnds[0])
        & (obs_data["lat"] < lat_bnds[1])
    ]

    # create an array of lats and lons for the remaining ship obs
    obs_lat = cond_df["lat"]
    obs_lon = cond_df["lon"]

    # extract the CCI SST anomaly values corresponding to the obs lat/lon
    # coordinate points
    # print(ds_masked_xr)
    its_waterpoint_for_obs, lat_idx, lon_idx, grid_lat, grid_lon = find_values(
        mask_ds, obs_lat, obs_lon
    )
    cond_df["cci_waterpoint"] = its_waterpoint_for_obs
    cond_df["gridcell_lat"] = grid_lat
    cond_df["gridcell_lon"] = grid_lon
    print(cond_df)

    # find the indices for lats and lons of the observations
    idx_tuple = np.array([lat_idx, lon_idx])
    print(f"{idx_tuple = }")
    # below 0 is set as the first timestep, it can be any number as the scenes
    # have the same land/ice mask over all time steps, but it has to be set to
    # extract the 2D field
    try:
        mask = mask_ds.variables["landmask"].values
    except KeyError:
        # mask = mask_ds.variables['land_sea_mask'].values
        mask = mask_ds.variables["landice_sea_mask"].values
    output_size = mask.shape
    # used to be: output_size = (ds[str(ds_var)].values[timestep,:,:]).shape
    print(f"{output_size = }")
    # transform location indices into a flattened 1D index
    flattened_idx = np.ravel_multi_index(
        idx_tuple, output_size, order="C"
    )  # row-major
    # print(f'{flattened_idx = }')
    # print('flattened idx', flattened_idx.shape)

    # add information about the 1D obs index to the dataframe
    cond_df["flattened_idx"] = flattened_idx

    # in case some of the anomaly values are Nan (because covered by ice)
    cond_df = cond_df[cond_df["cci_waterpoint"] == 1]  # cond_df.dropna()
    obs_flat_idx = cond_df["flattened_idx"].values
    return cond_df, obs_flat_idx


# NOTE: Uncorrelated Component
# TODO: Docstring
def obs_covariance(
    df: pd.DataFrame,
    group_col: str = "data_type",
    obs_sig_col: str | None = None,
    obs_sig_map: dict[str, float] | None = None,
) -> np.ndarray:
    """
    Calculates the covariance matrix of the measurements (obervations). This
    is the uncorrelated component of the covariance.

    Parameters
    ----------
    df : pandas.DataFrame
        The observational DataFrame containing values to group by.
    group_col : str
        Name of the group column to use to set observational sigma values.
    obs_sig_col : str | None
        Name of the column containing observational sigma values. If set and
        present in the DataFrame, then this column is used as the diagonal of
        the returned covariance matrix.
    obs_sig_map : dict[str, float] | None
        Mapping between group and observational sigma values used to define
        the diagonal of the returned covariance matrix.

    Returns
    -------
    A single covariance matrix of measurements for ship observations and buoy
    observations
    """
    if obs_sig_col is not None and obs_sig_col in df.columns:
        return np.diag(df.ix[:, obs_sig_col])

    obs_sig_map = obs_sig_map or {}
    data_types: pl.Series = pl.from_pandas(df.ix[:, group_col])
    s = data_types.replace_strict(
        {k: v**2 for k, v in obs_sig_map.items()}, default=0.0
    )
    if s.eq(0.0).all():
        warn("No values in obs_covariance set")
    elif s.eq(0.0).any():
        warn("Some values in obs_covariance not set")

    return np.diag(s)

    # try:
    #     # n_ship = df[df.obs_type <= 5].shape[0]
    #     n_ship = df[df["data_type"] == "ship"].shape[0]
    # except:
    #     n_ship = df[df["data.type"] == "ship"].shape[0]
    # print("n_ship", n_ship)
    # try:
    #     n_buoy = df[df["data_type"] == "buoy"].shape[0]
    #     # n_buoy = df[(df.obs_type == 6) | (df.obs_type == 7)].shape[0]
    # except:
    #     n_buoy = df[df["data.type"] == "buoy"].shape[0]
    # print("n_buoy", n_buoy)
    # # Create covariance matrix for the measurements
    # a1 = np.multiply(np.eye(n_ship), (np.power(sig_ms, 2)))
    # a2 = np.multiply(np.eye(n_buoy), (np.power(sig_mb, 2)))
    # covx = block_diag(a1, a2)  # scipy.linalg.block_diag
    # # print('covx', covx)
    # return covx


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
    lat1 (float) - latitude of point A
    lon1 (float) - longitude of point A
    lat2 (float) - latitude of point B
    lon2 (float) - longitude of point B

    Returns
    -------
    Radial distance between point A and point B
    """
    # approximate radius of earth in km
    R = 6371.0
    lat1r = math.radians(lat1)
    # lon1r = math.radians(lon1)
    lat2r = math.radians(lat2)
    # lon2r = math.radians(lon2)

    dlon = math.radians(lon2 - lon1)
    dlat = lat2r - lat1r

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


# def sampling_uncertainty(
#     obs_idx: np.ndarray,  # I think this is 1d?
#     covx: np.matrix,
#     df: pd.DataFrame,
#     r: float = 40.0,
#     s: float = 0.6,
#     R_earth: float = 6371.0,
# ) -> Tuple[np.matrix, np.matrix]:
#     """
#     Get an updated measurement covariance matrix by adding sampling uncertainty
#     of the observations
#
#     Args
#     ----
#
#     obs_idx : numpy.ndarray[int]
#         Grid-box index for each observation
#     covx : numpy.matrix[float]
#         The covariance matrix
#     df : pandas.DataFrame
#         The DataFrame containing the records, requires "lat" and "lon" columns
#         indicating the positions of the observations
#     r : float
#         Range parameter for distance matrix covariance effect
#     s : float
#         Scale parameter for distance matrix covariance effect
#     R_earth : float
#         Radius of earth. [km]
#
#     Return
#     ------
#
#     covx : numpy.matrix[float]
#         Updated covariance Matrix
#     W : numpy.matrix[float]
#         Weights(?)
#     """
#     # obs_idx = obs_idx.reshape(-1)  # Convert to 1d array
#     unique_obs_idx = np.unique(obs_idx)
#     _N = np.max(unique_obs_idx.shape)
#     _P = np.max(obs_idx.shape)
#
#     # Get positional data: lat, lon as radians (required for
#     # haversine_distances)
#     pos = np.radians(df[["lat", "lon"]].to_numpy())
#
#     W = np.matrix(np.zeros((_N, _P)))
#
#     all_is = np.arange(_P)
#     for i, obs in enumerate(unique_obs_idx):
#         # Where data match the obs id
#         q = all_is[obs_idx == obs]
#
#         W[i, q] = 1 / len(q)  # I think this works...
#         # Get matrix for the data subset
#         C = haversine_distances(pos[q]) * R_earth
#         C = np.exp(-(C**2) / r**2)
#         C = s / 2 * C
#
#         # Update covariance
#         q_idx = np.ix_(q, q)
#         covx[q_idx] = covx[q_idx] + C
#
#         del C, q, q_idx
#
#     del pos, unique_obs_idx, all_is, _N, _P
#
#     return covx, W


def haversine_gaussian(
    df: pd.DataFrame,
    R: float = 6371.0,
    r: float = 40,
    s: float = 0.6,
) -> np.ndarray:
    """
    Gaussian Haversine Model

    Parameters
    ----------
    df : pandas.DataFrame
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
    pos = np.radians(df[["lat", "lon"]].to_numpy())
    C = haversine_distances(pos) * R
    C = np.exp(-(C**2) / r**2)
    return s / 2 * C


def dist_weight(
    df: pd.DataFrame,
    dist_fn: Callable | None,
    **dist_kwargs,
) -> tuple[np.ndarray | None, np.ndarray]:
    """
    Compute the distance and weight matrices over gridboxes for an input Frame.

    This function acts as a wrapper for a distance function, allowing for
    computation of the distances between positions in the same gridbox using any
    distance metric.

    Set dist_fn to None to just compute weights.

    Parameters
    ----------
    df : pandas.DataFrame
        The observation DataFrame, containing the columns required for
        computation of the distance matrix. Contains the "gridbox" column which
        indicates the gridbox for a given observation. The index of the
        DataFrame should match the index ordering for the output distance
        matrix/weights.
    dist_fn : Callable | None
        The function used to compute a distance matrix for all points in a given
        grid-cell. Takes as input a pandas.DataFrame as first argument. Any
        other arguments should be constant over all gridboxes, or can be a
        look-up table that can use values in the DataFrame to specify values
        specific to a gridbox. The function should return a numpy matrix, which
        is the distance matrix for the gridbox only. This wrapper function will
        correctly apply this matrix to the larger distance matrix using the
        index from the DataFrame.

        If dist_fn is None, then no distances are computed and None is returned
        for the dist value.
    **dist_kwargs
        Arguments to be passed to dist_fn. In general these should be constant
        across all gridboxes. It is possible to pass a look-up table that
        contains pre-computed values that are gridbox specific, if the keys can
        be matched to a column in df.

    Returns
    -------
    dist : numpy.matrix | None
        The distance matrix, which contains the same number of rows and columns
        as rows in the input DataFrame df. The values in the matrix are 0 if the
        indices of the row/column are for observations from different gridboxes,
        and non-zero if the row/column indices fall within the same gridbox.
        Consequently, with appropriate re-arrangement of rows and columns this
        matrix can be transformed into a block-diagonal matrix. If the DataFrame
        input is pre-sorted by the gridbox column, then the result is a
        block-diagonal matrix.

        If dist_fn is None, then this value will be None.
    weights : numpy.matrix
        A matrix of weights. This has dimensions n x p where n is the number of
        unique gridboxes and p is the number of observations (the number of rows
        in df). The values are 0 if the row and column do not correspond to the
        same gridbox and equal to the inverse of the number of observations in a
        gridbox if the row and column indices fall within the same gridbox. The
        rows of weights are in a sorted order of the gridbox. Should this be
        incorrect, one should re-arrange the rows after calling this function.
    """
    # QUESTION: Do we want to sort the unique grid-cell values?
    #           Ensures consistency between runs if the frame ordering gets
    #           shuffled in some way.
    gridboxes = sorted(df["gridbox"].unique())
    _n_gridboxes = len(gridboxes)
    _n_obs = len(df)

    weights = np.zeros((_n_gridboxes, _n_obs))
    dist = np.zeros((_n_obs, _n_obs)) if dist_fn is not None else None

    for i, gridbox in enumerate(gridboxes):
        gridbox_idcs = list(df[df["gridbox"] == gridbox].index)
        idcs_array = np.ix_(gridbox_idcs, gridbox_idcs)

        weights[i, gridbox_idcs] = 1 / len(gridbox_idcs)
        if dist_fn is not None and dist is not None:
            dist[idcs_array] = dist_fn(df.loc[gridbox_idcs], **dist_kwargs)

    return dist, weights


def get_weights(df: pd.DataFrame) -> np.ndarray:
    """
    Get just the weight matrices over gridboxes for an input Frame.

    Parameters
    ----------
    df : pandas.DataFrame
        The observation DataFrame, containing the columns required for
        computation of the distance matrix. Contains the "gridbox" column which
        indicates the gridbox for a given observation. The index of the
        DataFrame should match the index ordering for the output weights.

    Returns
    -------
    weights : numpy.matrix
        A matrix of weights. This has dimensions n x p where n is the number of
        unique gridboxes and p is the number of observations (the number of rows
        in df). The values are 0 if the row and column do not correspond to the
        same gridbox and equal to the inverse of the number of observations in a
        gridbox if the row and column indices fall within the same gridbox. The
        rows of weights are in a sorted order of the gridbox. Should this be
        incorrect, one should re-arrange the rows after calling this function.
    """
    _, weights = dist_weight(df, dist_fn=None)
    return weights


# def sampling_uncertainty_old(
#     flattened_idx: np.ndarray,
#     covx: np.ndarray,
#     df: pd.DataFrame,
# ) -> np.ndarray:
#     """
#     Returns an updated measurement covariance matrix by adding sampling
#     uncertainty of the observations
#
#     Parameters
#     ----------
#     uind (list)
#         list of unique (i.e. not repeated) locations of the
#         observations for a given date
#     iid (list)
#         list of all locations of the observations for a given date
#     clim (array)
#         1-D array of climatology for the whole map grid
#     scale_clim (array)
#         scaling climatology parameters (based on the scale from the variogram)
#     mr (array)
#         range parameters from the variogram file
#     covx (array)
#         ship and buoy measurement covariance matrix
#     lat (array)
#         array of latitudes for observation points
#     lon (array)
#         array of longitudes for observation points
#
#     Returns
#     -------
#     Covariance matrix of the ship and buoy measurements including sampling
#     uncertainty
#     Matrix C of the counts of observations based on the radial distance and
#     range and scale parameters
#     Matrix W of weight of each observation on the grid cell
#     """
#     lat = df["lat"].to_numpy()
#     lon = df["lon"].to_numpy()
#     obs_idx = flattened_idx
#     print(obs_idx, obs_idx.shape)
#     unique_obs_idx = np.unique(obs_idx)
#     print(unique_obs_idx, unique_obs_idx.shape)
#     W = np.zeros((int(max(unique_obs_idx.shape)), int(max(obs_idx.shape))))
#     print(W.shape)
#
#     for k in range(max(unique_obs_idx.shape)):
#         range_ = 40.0  # set_number for now
#         scale = 0.6  # set_number for now
#         q = [i for i, x in enumerate(obs_idx) if x == unique_obs_idx[k]]
#
#         # print('obs_idx', obs_idx)
#         # print('unique obs idx', unique_obs_idx)
#         # print('q', q)
#         for i in range(len(q)):
#             qq = q[i]
#             W[k, qq] = np.divide(1, len(q))
#
#         C = np.zeros((len(q), len(q)))
#
#         for jj in range(len(q)):
#             for kk in range(len(q)):
#                 idx1 = q[jj]
#                 idx2 = q[kk]
#                 C[jj, kk] = radial_dist(
#                     lat[idx1], lon[idx1], lat[idx2], lon[idx2]
#                 )
#
#         C = np.exp(-(C**2) / range_**2)
#         C = scale / 2 * C
#         # print('C', C)
#         for jj in range(len(q)):
#             for kk in range(len(q)):
#                 # print(q[jj], q[kk])
#                 covx[q[jj], q[kk]] = covx[q[jj], q[kk]] + C[jj, kk]
#     # print('covx', covx)
#     return covx, W


def bias_uncertainty(
    df: pd.DataFrame,
    covx: np.ndarray,
    group_col: str,
    bias_val_col: str | None = None,
    bias_val_map: dict[str, float] | None = None,
) -> np.ndarray:
    """
    Returns measurements covariance matrix updated by adding bias uncertainty to
    the measurements based on a grouping within the observational data.

    These are the correlated components.

    Parameters
    ----------
    df : pandas.DataFrame
        Observational DataFrame including group information and bias uncertainty
        values for each grouping. It is assumed that a single bias uncertainty
        value applies to the whole group, and is applied as cross terms in the
        covariance matrix (plus to the diagonal).
    covx : numpy.ndarray
        Measurement covariance matrix.
    group_col : str
        Name of the column that can be used to partition the observational
        DataFrame.
    bias_val_col : str | None
        Name of the column containing bias uncertainty values for each of
        the groups identified by 'group_col'. It is assumed that a single bias
        uncertainty value applies to the whole group, and is applied as cross
        terms in the covariance matrix (plus to the diagonal).
    bias_val_map : dict[str, float] | None
        Mapping between values in the group_col and bias uncertainty values,
        if bias_val_col is not in the DataFrame.

    Returns
    -------
    Measurement covariance matrix updated by including bias uncertainty of the
    measurements
    """
    check_cols(df, [group_col])

    pl_df = pl.from_pandas(df)
    bias_val_col = bias_val_col or "_bias_uncert"
    bias_val_map = bias_val_map or {}

    if bias_val_col not in pl_df.columns:
        pl_df = pl_df.with_columns(
            pl.col(group_col)
            .replace_strict(bias_val_map, default=0.0)
            .alias(bias_val_col)
        )
        if pl_df[bias_val_col].eq(0.0).all():
            warn("No bias uncertainty values set")
        elif pl_df[bias_val_col].eq(0.0).any():
            warn("Some bias uncertainty values not set")

    # NOTE: polars is easier for this analysis!
    pl_df = (
        pl_df.select(group_col, bias_val_col)
        .with_row_index("index")
        .group_by(group_col)
        # NOTE: It is expected that the bias value should be the same for all
        #       records within the same group
        .agg(pl.col("index"), pl.col(bias_val_col).first())
    )
    for row in pl_df.rows(named=True):
        if row[bias_val_col] is None:
            print(f"Group {row[group_col]} has no bias uncertainty value set")
            continue
        # INFO: Adding cross-terms to covariance
        inds = np.ix_(row["index"], row["index"])
        covx[inds] = covx[inds] + row[bias_val_col]

    return covx
    #
    # try:
    #     # NOTE: string column
    #     type_id = np.array(df["orig_id"])
    #     # type_id = np.array(df['type_id'])
    # except:
    #     # NOTE: string column
    #     type_id = np.array(df["id.type"])
    # # NOTE: string column
    # vessel_id = np.array(df["id"])
    # # NOTE: integer column
    # vessel_ii = np.array(df["ii"])
    # # print('type id', type_id)
    # # print('vessel id', vessel_id)
    # # print('vessel ii', vessel_ii)
    #
    # # Finally the hard bit, the bias uncertainty, mainly because the ID's are
    # # quite bad!
    # # Okay those with ii == 2 just add the bias uncertainty to the diagonal
    # p = [
    #     i
    #     for i, (vessel, id_) in enumerate(zip(vessel_ii, type_id))
    #     if vessel == 2 and id_ == 0
    # ]
    # for k in range(len(p)):
    #     covx[p[k], p[k]] = covx[p[k], p[k]] + sig_bs**2
    #
    # p = [
    #     i
    #     for i, (vessel, id_) in enumerate(zip(vessel_ii, type_id))
    #     if vessel == 2 and id_ == 1
    # ]
    # for k in range(len(p)):
    #     covx[p[k], p[k]] = covx[p[k], p[k]] + sig_bb**2
    #
    # # Just a quick check on the buoy data that all
    # # measurement_covariance(sig_ms, sig_mb, sig_bs, sig_bb, cond_df,
    # #                        flattened_idx, lat, lon)
    # # the ii's are 3
    # p = [
    #     i
    #     for i, (vessel, id_) in enumerate(zip(vessel_ii, type_id))
    #     if id_ == 1 and vessel != 3
    # ]
    # if len(p) > 0:
    #     print(
    #         "Warning there are %d BUOY measurements that are not classed as 3\n",
    #         len(p),
    #     )
    #
    # I = [i for i, id_ in enumerate(type_id) if id_ == 1]
    # UVID = np.unique(vessel_id[I])
    # # print('UVID', UVID)
    # for u in range(len(UVID)):
    #     p = [
    #         i
    #         for i, (id_, vid_) in enumerate(zip(type_id, vessel_id))
    #         if id_ == 1 and vid_ == UVID[u]
    #     ]
    #     row_idx = p
    #     col_idx = p
    #     # covx[p,p] = covx[p,p] + sig_bb ** 2
    #     covx[np.ix_(row_idx, col_idx)] = (
    #         covx[np.ix_(row_idx, col_idx)] + sig_bb**2
    #     )
    #
    # II = [
    #     i
    #     for i, (vessel, id_) in enumerate(zip(vessel_ii, type_id))
    #     if id_ == 0 and vessel != 2
    # ]
    # UVID = np.unique(vessel_id[II])
    # # print('UVID', UVID)
    # for u in range(len(UVID)):
    #     q = [
    #         i
    #         for i, (id_, vid_) in enumerate(zip(type_id, vessel_id))
    #         if id_ == 0 and vid_ == UVID[u]
    #     ]
    #     # covx[p[k],p[k]] = covx[p[k],p[k]] + sig_bb ** 2
    #     row_idx = q
    #     col_idx = q
    #     covx[np.ix_(row_idx, col_idx)] = (
    #         covx[np.ix_(row_idx, col_idx)] + sig_bs**2
    #     )
    # # print('covx', covx)
    # return covx


# def correlated_uncertainty(df):
#     type_id = np.array(df["type_id"])
#     vessel_id = np.array(df["id"])
#     vessel_ii = np.array(df["ii"])
#     flattened_idx = np.array(df["flattened_idx"])
#     unique_idx = np.unique(flattened_idx).tolist()
#     for i in unique_idx:
#         # print(i)
#         # print(df.loc[df['flattened_idx'] == i])
#         all_vessels_for_this_grid = sum(
#             df.loc[df["flattened_idx"] == i, "id"].values
#         )
#         # print(all_vessels_for_this_grid)
#         unique_vessels = sum(np.unique(all_vessels_for_this_grid))
#         # print(unique_vessels)
#         vessel_ratio_per_obs_grid = unique_vessels / all_vessels_for_this_grid
#         # print(vessel_ratio_per_obs_grid)


# WARN: Memory!! Unnecessary copies!
# QUESTION: Do the indices of obs_covariance and that from dist_weight line up?
def measurement_covariance(
    df: pd.DataFrame,
    sig_ms: float,
    sig_mb: float,
    sig_bs: float,
    sig_bb: float,
) -> tuple[np.ndarray, np.ndarray]:
    # covx = correlated_uncertainty(df)
    # just the basic covariance for number of ship and buoy
    obs_bias_map = {"ship": sig_ms, "buoy": sig_mb}
    covx1 = obs_covariance(df, group_col="data_type", obs_sig_map=obs_bias_map)
    # print(covx1, covx1.shape)
    # adding the weights (no of obs in each grid) + importance based on distance
    # scaled by range and scale (values adapted from the power point
    # presentation)
    # dist, W = dist_weight(df, dist_fn=haversine_gaussian,
    #                       R=6371.0, r=40, s=0.6)
    required_cols = [
        "lat",
        "lon",
        "gridbox",
        "gridcell_lat",
        "gridcell_lon",
        "gridcell_lx",
        "gridcell_ly",
        "gridcell_theta",
    ]
    cols_miss = [c for c in required_cols if c not in df]
    if cols_miss:
        raise ValueError(
            f"Missing columns required for tau computation: {cols_miss}"
        )
    dist, weights = dist_weight(df, dist_fn=tau_dist)
    covx1 = covx1 + dist
    # print(covx1, covx1.shape)
    bias_uncert_map = {"ship": sig_bs, "buoy": sig_bb}
    covx1 = bias_uncertainty(
        df,
        covx1,
        group_col="data_type",
        bias_val_map=bias_uncert_map,
    )
    # print(covx2, covx2.shape)
    return covx1, weights


def esa_cci_monthly_climatology(climatology_path: str) -> xr.DataArray:
    climatology = xr.open_mfdataset(
        climatology_path,
        combine="nested",
        concat_dim="time",
        engine="netcdf4",
    ).chunk({"time": 365})
    climatology_365 = climatology["analysed_sst"]
    # print(climatology_365)
    # dimensions as read in:
    # analysed_sst(time, lat, lon)
    climatology_365 = climatology_365.transpose("lat", "lon", "time")
    climatology_365["time"] = np.arange(1, 366)
    # climatology_365 = climatology_365.assign_coords(doy=('time', np.arange(1,366)))
    # climatology_365.swap_dims({'time': 'doy'})
    # print(climatology_365)
    return climatology_365


def match_climatology_to_obs(climatology_365, obs_df):
    """
    obs_year = obs_df.fake_non_leap_year
    obs_month = obs_df.month
    obs_day = obs_df.day
    obs_doy = int(datetime.date(obs_year,obs_month, obs_day).strftime('%j'))
    #obs_doy = datetime.date(obs_year, obs_month, obs_day).timetiuple().tm_yday

    for
    if obs_doy == np.nan:
        #if the DOY is nan then it means it is leap year and the date is 29 Feb
        #in this case we need to get an average between 28 Feb (DOY 59, Python index 58) and 1 Mar (DOY 60, Python index 59)
        clim_val =np.mean(climatology_365[obs_df.clim_lat_idx, obs_df.clim_lon_idx,58],climatology_365[obs_df.clim_lat_idx, obs_df.clim_lon_idx, 59])
    else:
        #Python does indexing from zero, sowhen indexing into a timestep, we need to substract 1 from DOY
        clim_val = climatology_365[obs_df.clim_lat_idx, obs_df.clim_lon_idx, obs_doy-1]


    clim_doy_idx.append(clim_doy)
    """
    obs_lat = obs_df.lat
    obs_lon = obs_df.lon
    obs_df["lat_idx"], obs_df["lon_idx"] = find_latlon_idx(
        climatology_365, obs_lat, obs_lon
    )
    cci_clim = []  # ESA CCI climatology values

    mask = (obs_df["date"].dt.is_leap_year == 1) & (
        obs_df["date"].dt.dayofyear == 60
    )
    # print(obs_df['date'])
    # print(mask)
    non_leap_df = obs_df[~mask]
    leap_df = obs_df[mask]

    print(non_leap_df)
    print(leap_df)

    non_leap_df["fake_non_leap_year"] = 2010
    non_leap_df["fake_non_leap_date"] = pd.to_datetime(
        dict(
            year=non_leap_df.fake_non_leap_year,
            month=non_leap_df.month,
            day=non_leap_df.day,
        )
    )
    non_leap_df["doy"] = [
        int(i.strftime("%j")) for i in non_leap_df["fake_non_leap_date"]
    ]
    print(non_leap_df.doy)
    non_leap_df["doy_idx"] = non_leap_df.doy - 1
    print(non_leap_df.doy_idx)

    # print(climatology_365, climatology_365.time)
    c = climatology_365.values
    # print(c.shape)

    selected = c[
        np.array(non_leap_df.lat_idx),
        np.array(non_leap_df.lon_idx),
        np.array(non_leap_df.doy_idx),
    ]
    selected = selected - 273.15  # from Kelvin to Celsius
    # print(selected)

    # print(len(non_leap_df.lat_idx), len(non_leap_df.lon_idx),
    #       len(non_leap_df.doy_idx), len(selected))

    # climatology_365.sel(lat=obs_lat, lon=obs_lon, method="nearest")
    # climatology_365.lat.values[lat_idx]
    end_feb = c[
        np.array(leap_df.lat_idx),
        np.array(leap_df.lon_idx),
        np.repeat(np.array([58]), len(leap_df.lat_idx)),
    ]
    # climatology_365[tuple((leap_df.lat_idx, leap_df.lon_idx,
    # np.repeat(np.array([58]), len(leap_df.lat_idx))))] - cannot use tuple!

    beg_mar = c[
        np.array(leap_df.lat_idx),
        np.array(leap_df.lon_idx),
        np.repeat(np.array([59]), len(leap_df.lat_idx)),
    ]
    # climatology_365[tuple((leap_df.lat_idx, leap_df.lon_idx,
    # np.repeat(np.array([59]), len(leap_df.lat_idx))))] - cannot use tuple!

    selected2 = [(g + h) / 2 for g, h in zip(end_feb, beg_mar)]
    selected2 = [i - 273.15 for i in selected2]  # from Kelvin to Celsius
    # print(selected2)

    non_leap_df["climatology"] = selected
    leap_df["climatology"] = selected2
    obs_df = pd.concat([non_leap_df, leap_df])
    # print('joint leap and non-leap observations', obs_df)
    obs_df["obs_anomalies"] = obs_df["sst"] - obs_df["climatology"]
    # df1 = obs_df[['lat', 'lon', 'sst',
    #               'climatology', 'cci_anomalies', 'obs_anomalies']]
    # print(df1)
    # STOP
    """
    #in case some of the values are Nan (because covered by ice)
    obs_df = obs_df.dropna()
    """
    return obs_df


def find_latlon_idx(
    data_array: xr.DataArray,
    lats: np.ndarray,
    lons: np.ndarray,
) -> tuple[list[int], list[int]]:
    """
    Parameters
    ----------

    Returns
    -------
    Dataframe with added anomalies for each observation point
    """
    lat_idx, _ = find_nearest(data_array.lat, lats)
    # print(nc_xr.lat, lat_idx)
    lon_idx, _ = find_nearest(data_array.lon, lons)
    # print(nc_xr.lon, lon_idx)
    return lat_idx, lon_idx


def find_anomaly_values(
    lat_idx: np.ndarray,
    lon_idx: np.ndarray,
    ds_masked_xr: xr.DataArray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    anom_vals = np.array(
        [ds_masked_xr.values[j, i, 0] for j, i in zip(lat_idx, lon_idx)]
    )
    # cci_vals = []  # fake ship obs
    # for i in range(len(cci_lat_idx)):
    #     # cci_ = cci[str(ds_var)].values[timestep,cci_lat_idx[i],cci_lon_idx[i]] #was cci.sst_anomaly
    #     cci_ = ds_masked_xr.values[cci_lat_idx[i], cci_lon_idx[i], 0]
    #     # was - ds_masked_xr.values[cci_lat_idx[i],cci_lon_idx[i],timestep]
    #     # was cci.sst_anomaly
    #     cci_vals.append(cci_)
    # cci_vals = np.hstack(cci_vals)
    # if lats and lons indices index into a SST anomaly field without a value / with NaN
    # see: https://stackoverflow.com/questions/40592630/get-coordinates-of-non-nan-values-of-xarray-dataset
    # Remove positions that are missing
    to_remove = np.argwhere(np.isnan(anom_vals))
    anom_vals = np.delete(anom_vals, to_remove)
    lat_idx = np.delete(lat_idx, to_remove)
    lon_idx = np.delete(lon_idx, to_remove)
    return anom_vals, lat_idx, lon_idx


def find_obs_latlon_idx_on_output_grid(lat_idx, lon_idx, ds_masked_xr, cond_df):
    # find the indices for lats and lons of the observations
    idx_tuple = np.array([lat_idx, lon_idx])
    output_size = (ds_masked_xr.values[:, :, 0]).shape
    # was - (ds_masked_xr.values[:,:,timestep]).shape
    # transform location indices into a flattened 1D index
    flattened_idx = np.ravel_multi_index(
        idx_tuple, output_size, order="C"
    )  # row-major
    # add information about the 1D obs index to the dataframe
    cond_df["flattened_idx"] = flattened_idx
    # print(cond_df)
    # print('flattened idx', flattened_idx)
    return cond_df, flattened_idx


# QUESTION: Is this equivalent to get_weights?

# def counts_for_esa(obs_idx: np.ndarray):
#     # print(obs_idx, obs_idx.shape)
#     unique_obs_idx = np.unique(obs_idx)
#     # print(unique_obs_idx, unique_obs_idx.shape)
#     W = np.zeros((int(max(unique_obs_idx.shape)), int(max(obs_idx.shape))))
#     for k in range(max(unique_obs_idx.shape)):
#         q = [i for i, x in enumerate(obs_idx) if x == unique_obs_idx[k]]
#         for i in range(len(q)):
#             qq = q[i]
#             W[k, qq] = np.divide(1, len(q))
#     return W


# def obs_main(
#     obs_path, lon_bnds, lat_bnds, ds_masked_xr, climatology_path, year=False
# ):
#     #    obs_filename, lon_bnds, lat_bnds, ds_masked_xr, timestep):
#     # df = get_dataframe(obs_path, lon_bnds, lat_bnds, year)
#     df = yearly_processing(obs_path, lon_bnds, lat_bnds, year)
#     df_anomaly, flattened_idx1 = add_esa_anomalies_at_obs_locations(
#         lon_bnds, lat_bnds, ds_masked_xr, df
#     )
#     # print('DF')
#     # print(df)
#
#     # print('DF ANOMALY')
#     # print(df_anomaly)
#     """
#     climatology_365 = esa_cci_monthly_climatology(climatology_path)
#     #cond_df = add_esa_climatology_at_obs_locations(lon_bnds, lat_bnds, climatology_365, df)
#     cond_df = match_climatology_to_obs(climatology_365, df_anomaly)
#
#     obs_lat = np.array(cond_df['lat'])
#     obs_lon = np.array(cond_df['lon'])
#     lat_idx, lon_idx = find_latlon_idx(ds_masked_xr, obs_lat, obs_lon)
#     cond_df, flattened_idx2 = find_obs_latlon_idx_on_output_grid(lat_idx, lon_idx, ds_masked_xr, cond_df)
#     print('COND DF')
#     print(cond_df)
#
#
#     #separate obs dataframes based on observation type
#     ship_df = df[df['obs_type'] <= 5]
#     buoy_df = df[df['obs_type'] == 6]
#     drifters_df = df[df['obs_type'] == 7]
#     #create covariance matrix for observations and weights matrix (i.e. how many observations fall into the same gridbox)
#     #remove obs_type that are not ship (<= 5) or buoy (6 or 7) for the purpose of measurement covariance matrix
#     cond_df_ship_buoy = cond_df[cond_df.obs_type <= 7]
#     #add a for loop to loop over days in month
#     flattened_idx = cond_df_ship_buoy['flattened_idx'][:]
#
#     print('DF SHIP BUOY')
#     print(cond_df_ship_buoy)
#     print(flattened_idx, len(flattened_idx))
#     #oops = cond_df_ship_buoy['obs_anomalies'].isnull().sum() #oops 20772
#     #oops = cond_df_ship_buoy['sst'].isnull().sum() #oops 18985
#     #oops = cond_df_ship_buoy['climatology'].isnull().sum() #oops 3275
#     #print('oops', oops)
#     cond_df_ship_buoy = cond_df_ship_buoy.dropna()
#     """
#     # obs_covariance, W_matrix = measurement_covariance(cond_df_ship_buoy, flattened_idx, sig_ms=1.27, sig_mb=0.23, sig_bs=1.47, sig_bb=0.38)
#     return df_anomaly, flattened_idx1
#     # return cond_df_ship_buoy, flattened_idx #obs_covariance, W_matrix, cond_df, flattened_idx


def read_hadsst_bias(filepath, year, month):
    bias_file = xr.open_dataset(filepath, engine="netcdf4")
    # if subsampling needed:
    # bias_file = bias_file.sel(lat=slice(min_lat,max_lat), lon=slice(min_lon,max_lon))
    print(bias_file)
    obs_date = np.datetime64(f"{year}-{month:02d}-16")  # 'ns'
    print(f"{obs_date = }")
    hadsst_date = nearest(bias_file.time[:], obs_date)
    print(f"{hadsst_date = }")
    bias_file_for_month = bias_file.sel(time=hadsst_date)
    print(bias_file_for_month)
    return bias_file_for_month


def nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))


def match_ellipse_parameters_to_gridded_obs(
    ellipse_monthly_array: xr.DataArray,
    cond_df: pd.DataFrame,
    mask_ds: xr.Dataset,
    mask_var: str = "landmask",
) -> pd.DataFrame:
    el_lat = ellipse_monthly_array.latitude
    el_lon = ellipse_monthly_array.longitude
    print(f"{el_lat = }")
    print(f"{el_lon = }")
    el_lx = ellipse_monthly_array.lx.values.flatten()
    el_ly = ellipse_monthly_array.ly.values.flatten()
    el_theta = ellipse_monthly_array.theta.values.flatten()

    # to remind how obtained:
    # flattened_idx = np.ravel_multi_index(idx_tuple, output_size, order='C') #row-major
    # flattened_idx is a column in the dataframe called "flattened_idx"
    # unique_idx = np.unique(flattened_idx)

    print(f"{cond_df = }")
    output_size = mask_ds.variables[mask_var].values.shape
    print(f"{output_size = }")
    print(f"{el_lx.shape = }")
    gridcells = list(cond_df["flattened_idx"])
    cond_df["gridcell_lx"] = el_lx[gridcells]
    cond_df["gridcell_ly"] = el_ly[gridcells]
    cond_df["gridcell_theta"] = el_theta[gridcells]
    print(cond_df)
    return cond_df


# WARN: Memory!! Unnecessary copies!
def TAO_measurement_covariance(
    df, flattened_idx, sig_ms, sig_mb, sig_bs, sig_bb
):
    # covx = correlated_uncertainty(df)
    # just the basic covariance for number of ship and buoy
    print(f"{df = }")
    df.insert(0, "data_type", "buoy")
    print(f"{df = }")
    covx1 = obs_covariance(
        df, group_col="data_type", obs_sig_map={"ship": sig_ms, "buoy": sig_mb}
    )
    # print(covx1, covx1.shape)
    # adding the weights (no of obs in each grid) + importance based on distance
    # scaled by range and scale (values adapted from the power point
    # presentation)
    dist, weights = dist_weight(
        df, dist_fn=haversine_gaussian, R=6371.0, r=40, s=0.6
    )
    covx1 = covx1 + dist
    print(covx1)
    return covx1, weights
