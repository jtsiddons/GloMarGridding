#!/usr/bin/env python

################
# by A. Faulkner
# for python version 3.0 and up
################

# global
import os

# argument parser
import argparse
from configparser import ConfigParser
from collections import OrderedDict

# math tools
import numpy as np

# data handling tools
from pandas.core.groupby import DataFrameGroupBy
import xarray as xr
import netCDF4 as nc

# self-written modules (from the same directory)
import observations as obs_module
import kriging as krig_module
from utils import days_since_by_month

import warnings

class ConfigParserMultiValues(OrderedDict):
    def __setitem__(self, key, value):
        if key in self and isinstance(value, list):
            self[key].extend(value)
        else:
            super().__setitem__(key, value)

    @staticmethod
    def getlist(value):
        return value.splitlines()


def _get_sst_err_cov(
    current_year: int,
    current_month: int,
    error_covariance_path: str,
    uncorrelated: xr.Dataset,
) -> np.ndarray:
    err_cov_fn = f"HadCRUT.5.0.2.0.error_covariance.{current_year}{current_month:02d}.nc"
    error_cov = xr.open_dataset(os.path.join(error_covariance_path, err_cov_fn))["tas_cov"].values[0]
    uncorrelated_mon = uncorrelated.sel(
        time=np.logical_and(
            uncorrelated.time.dt.month == current_month,
            uncorrelated.time.dt.year == current_year,
        )
    )["tas_unc"].values
    joined_mon = uncorrelated_mon * uncorrelated_mon
    unc_1d = np.reshape(joined_mon, (2592, 1))
    covariance2 = np.diag(np.reshape(unc_1d, (2592)))
    return error_cov + covariance2


def _get_lsat_err_cov(
    current_year: int, current_month: int, error_cov: np.ndarray
) -> np.ndarray:
    date_int = (current_year - 1850) * 12 + (current_month - 1)
    return np.diag(error_cov[date_int, :])


def _get_obs_groups(data_path: str, var: str) -> DataFrameGroupBy:
    obs = xr.open_dataset(data_path).to_dataframe().reset_index()
    obs.rename(columns={"latitude": "lat", "longitude": "lon"}, inplace=True)
    obs.drop_duplicates(subset=["lon", "lat", "time", var], inplace=True)
    obs.dropna(subset=[var], inplace=True)
    obs.reset_index(inplace=True)
    return obs.groupby([obs.time.dt.year, obs.time.dt.month])


def _initialise_ncfile(
    ncfile: nc.Dataset,
    output_lon: np.ndarray,
    output_lat: np.ndarray,
    current_year: int,
    variable: str,
):
    ncfile.createDimension("lat", len(output_lat))  # latitude axis
    ncfile.createDimension("lon", len(output_lon))  # longitude axis
    ncfile.createDimension("time", None)  # unlimited axis

    # Define two variables with the same names as dimensions,
    # a conventional way to define "coordinate variables".
    lat = ncfile.createVariable("lat", np.float32, ("lat",))
    lat.units = "degrees_north"
    lat.long_name = "latitude"
    lon = ncfile.createVariable("lon", np.float32, ("lon",))
    lon.units = "degrees_east"
    lon.long_name = "longitude"
    time = ncfile.createVariable("time", np.float32, ("time",))
    time.units = f"days since {current_year}-01-15"
    time.long_name = "time"
    # print(time)

    # Define a 3D variable to hold the data
    # note: unlimited dimension is leftmost
    krig_anom = ncfile.createVariable(
        f"{variable}_anomaly", np.float32, ("time", "lat", "lon")
    )
    krig_anom.standard_name = f"{variable} anomaly"
    krig_anom.units = "deg C"  # degrees Kelvin

    # Define a 3D variable to hold the data
    krig_uncert = ncfile.createVariable(
        f"{variable}_anomaly_uncertainty", np.float32, ("time", "lat", "lon")
    )
    krig_uncert.units = "deg C"  # degrees Kelvin
    krig_uncert.standard_name = "uncertainty"  # this is a CF standard name

    # Define a 3D variable to hold the data
    grid_obs = ncfile.createVariable(
        "observations_per_gridcell", np.float32, ("time", "lat", "lon")
    )
    # note: unlimited dimension is leftmost
    grid_obs.units = ""  # degrees Kelvin
    grid_obs.standard_name = "Number of observations within each gridcell"

    return lon, lat, time, krig_anom, krig_uncert, grid_obs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-config",
        dest="config",
        required=False,
        default=os.path.join(os.path.dirname(__file__), "config_HadCRUT.ini"),
        help="Path to INI file containing configuration settings",
        type=str,
    )
    parser.add_argument(
        "-year_start", dest="year_start", required=False, help="start year", type=int
    )
    parser.add_argument(
        "-year_stop", dest="year_stop", required=False, help="end year", type=int
    )
    parser.add_argument(
        "-member",
        dest="member",
        required=False,
        help="ensemble member: required argument",
        type=int,
        default=0,
    )
    parser.add_argument(
        "-variable",
        dest="variable",
        required=True,
        help='variable to process: "sst" or "lsat"',
        type=str,
        choices=["sst", "lsat"],
    )
    parser.add_argument(
        "-method",
        dest="method",
        default="ordinary",
        required=False,
        help='Kriging Method - one of "simple" or "ordinary"',
        choices=["simple", "ordinary"],
    )
    parser.add_argument(
        "-interpolation",
        dest="interpolation",
        default="ellipse",
        required=False,
        help='Interpolation covariance - one of "distance" or "ellipse"',
        choices=["distance", "ellipse"],
    )

    parser.add_argument(
        "-remove_obs_mean",
        dest="remove_obs_mean",
        default=0,
        required=False,
        type=int,
        help='Should the global mean be removed? - 0:no, 1:yes, 2:yes but median, 3:yes but spatial mean',
        choices=[0, 1, 2, 3],
    )

    args = parser.parse_args()

    config_file = args.config
    print(config_file)

    # load config options from ini file
    # this is done using an ini config file, which is located in the same direcotry as the python code
    # instantiate
    # ===== MODIFIED =====
    config = ConfigParser(
        strict=False,
        empty_lines_in_values=False,
        dict_type=ConfigParserMultiValues,
        converters={"list": ConfigParserMultiValues.getlist},
    )
    # ===== MODIFIED =====
    # parce existing config file
    config.read(config_file)
    print(config)

    # read values from auxiliary_files section
    # for string use config.get
    # for boolean use config.getboolean
    # for int use config.getint
    # for float use config.getfloat
    # for list (multiple options for same key) use config.getlist

    # set boundaries for the domain
    lon_west: float = config.getfloat("HadCRUT", "lon_west")  # -180.
    lon_east: float = config.getfloat("HadCRUT", "lon_east")  # 180.
    lat_south: float = config.getfloat("HadCRUT", "lat_south")  # -90.
    lat_north: float = config.getfloat("HadCRUT", "lat_north")  # 90.

    # extract the latitude and longitude boundaries from user input
    lon_bnds, lat_bnds = (lon_west, lon_east), (lat_south, lat_north)
    print(lon_bnds, lat_bnds)

    output_lat = np.arange(lat_bnds[0] + 2.5, lat_bnds[-1] + 2.5, 5)
    output_lon = np.arange(lon_bnds[0] + 2.5, lon_bnds[-1] + 2.5, 5)
    print(f"{output_lat =}")
    print(f"{output_lon =}")

    year_start: int = args.year_start or config.getint("HadCRUT", "startyear")
    year_stop: int = args.year_stop or config.getint("HadCRUT", "endyear")
    print(year_start, year_stop)

    # what variable is being processed
    variable: str = args.variable
    hadcrut_var: str = "tos" if variable == "sst" else "tas"

    # path to output directory
    output_directory: str = config.get("HadCRUT", "output_dir")
    output_directory = os.path.join(output_directory, hadcrut_var)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    print(f"{output_directory = }")

    # read in member from comand line
    member: int = args.member

    # var_range = config.getfloat(variable, "range")
    # var_sigma = config.getfloat(variable, "sigma")
    # var_matern = config.getfloat(variable, "matern")

    interpolation_covariance_type: str = args.interpolation
    interpolation_covariance_path: str = config.get(
        variable, f"{interpolation_covariance_type}_interpolation_covariance"
    )
    if interpolation_covariance_type == "distance":
        interp_covariance = np.load(interpolation_covariance_path)
        print("loaded interpolation covariance")
        print(interp_covariance)

    data_path: str = config.get(variable, "observations")
    yr_mo = _get_obs_groups(data_path, hadcrut_var)

    error_covariance_path = config.get(variable, "error_covariance")

    match variable:
        case "sst":
            print(f"Processing for variable {variable} | {hadcrut_var}")
            if config.has_option('sst', "sampling_uncertainty"):
                single_sigma_warn_msg = 'Option sampling_uncertainty for sst is ignored. '
                single_sigma_warn_msg += 'HadCRUT5 only has a single uncorrelated sigma; '
                single_sigma_warn_msg += 'if you are using multiple uncorrelated sigmas (e.g. HadSST4), combine them first.'
                warnings.warn(DeprecationWarning, single_sigma_warn_msg)
            uncorrelated_uncertainty = config.get(variable, "uncorrelated_uncertainty")

            uncorrelated = xr.open_dataset(uncorrelated_uncertainty)

            def get_error_cov(year: int, month: int) -> np.ndarray:
                return _get_sst_err_cov(
                    year, month, error_covariance_path, uncorrelated
                )

        case "lsat":
            print(f"Processing for variable {variable} | {hadcrut_var}")

            error_cov = np.load(
                os.path.join(
                    error_covariance_path, "HadCRUT.5.0.2.0.uncorrelated.npz"
                )
            )["err_cov"]
            print("loaded error covariance")
            print(error_cov)

            def get_error_cov(year: int, month: int) -> np.ndarray:
                return _get_lsat_err_cov(year, month, error_cov)

        case _:
            raise ValueError(f"Bad variable {variable}")

    year_list = range(year_start, year_stop + 1)
    month_list = range(1, 13)

    for current_year in year_list:
        try:
            ncfile.close()  # make sure dataset is not already open.
        except (NameError, RuntimeError):
            pass
        except Exception as e:  # Unknown Error
            raise e

        ncfilename = f"{current_year}_kriged"
        if member:
            ncfilename += f"_member_{member:03d}"
        ncfilename += ".nc"
        ncfilename = os.path.join(output_directory, ncfilename)

        ncfile = nc.Dataset(ncfilename, mode="w", format="NETCDF4_CLASSIC")
        # print(ncfile)

        lon, lat, time, krig_anom, krig_uncert, grid_obs = _initialise_ncfile(
            ncfile, output_lon, output_lat, current_year, variable
        )

        # Write latitudes, longitudes.
        # Note: the ":" is necessary in these "write" statements
        lat[:] = output_lat  # ds.lat.values
        lon[:] = output_lon  # ds.lon.values

        for current_month in month_list:
            timestep = current_month - 1
            print("Current month and year: ", (current_month, current_year))

            ################
            mon_df = yr_mo.get_group((current_year, current_month)).reset_index()
            print(f"{mon_df = }")

            if interpolation_covariance_type == "ellipse":
                interp_covariance = xr.open_dataset(
                    os.path.join(
                        interpolation_covariance_path,
                        f"covariance_{current_month:02d}_v_eq_1p5_{variable}_clipped.nc",
                    )
                )["covariance"].values
                # print(interp_covariance)

            error_covariance = get_error_cov(current_year, current_month)
            print(f"{error_covariance = }")

            ec_1 = error_covariance[~np.isnan(error_covariance)]
            ec_2 = ec_1[np.nonzero(ec_1)]
            print("Non-nan and non-zero error covariance =", ec_2, len(ec_2))
            ec_idx = np.argwhere(
                np.logical_and(~np.isnan(error_covariance), error_covariance != 0.0)
            )
            print("Index of non-nan and non-zero values =", ec_idx, len(ec_idx))

            _, mesh_lat = np.meshgrid(output_lon, output_lat)
            # print(mesh_lat, mesh_lat.shape)
            # print(mesh_lon, mesh_lon.shape)
            # print(mon_ds[variable].values.squeeze().shape)
            print("-----------------")
            # since we're not using any landmask for this run
            # the line below:
            # cond_df, obs_flat_idx = obs_module.watermask_at_obs_locations(lon_bnds, lat_bnds, mon_df, mask_ds, mask_ds_lat, mask_ds_lon)
            # mon_flat_idx = cond_df['flattened_idx'][:]
            # can be substituted with:
            lat_idx, grid_lat = obs_module.find_nearest(output_lat, mon_df.lat)
            lon_idx, grid_lon = obs_module.find_nearest(output_lon, mon_df.lon)

            mon_df["grid_lat"] = grid_lat
            mon_df["grid_lon"] = grid_lon
            mon_df["lat_idx"] = lat_idx
            mon_df["lon_idx"] = lon_idx

            idx_tuple = np.array([lat_idx, lon_idx])
            # print(f'{idx_tuple =}')
            mon_flat_idx = np.ravel_multi_index(
                idx_tuple, mesh_lat.shape, order="C"
            )  # row-major
            # print(f'{mon_flat_idx =}') #it's the same as ec_idx
            # print(f'{sorted(mon_flat_idx) =}')

            # diff = sorted(mon_flat_idx) - np.unique(mon_flat_idx)
            # diff results in 0s as unique idx is the same as sorted mon_flat_idx

            mon_df["gridbox"] = mon_flat_idx
            # print(mon_df)
            mon_df["error_covariance_diagonal"] = error_covariance[
                mon_flat_idx, mon_flat_idx
            ]
            print(mon_df)
            mon_df = mon_df.dropna(subset=["error_covariance_diagonal"])
            mon_df.reset_index(inplace=True, drop=True)
            print(mon_df)

            # count obs per grid for output
            gridbox_counts = mon_df["gridbox"].value_counts()
            gridbox_count_np = gridbox_counts.to_numpy()
            gridbox_id_np = gridbox_counts.index.to_numpy()
            del gridbox_counts
            water_mask = np.copy(mesh_lat)
            grid_obs_2d = krig_module.result_reshape_2d(
                gridbox_count_np, gridbox_id_np, water_mask
            )

            # need to either add weights (which will be just 1 everywhere as obs are gridded)
            # krige obs onto gridded field
            W = obs_module.get_weights(mon_df)
            # print(W)

            grid_idx = np.array(sorted(mon_df["gridbox"]))  # sorted?
            # print(error_covariance, error_covariance.shape)
            error_covariance = error_covariance[grid_idx[:, None], grid_idx[None, :]]
            # print(np.argwhere(np.isnan(np.diag(error_covariance))))
            # print(f'{error_covariance =}, {error_covariance.shape =}')

            print(mon_df)

            anom, uncert = krig_module.kriging(
                grid_idx,
                W,
                mon_df[hadcrut_var].values,
                interp_covariance,
                error_covariance,
                method=args.method,
                remove_obs_mean=args.remove_obs_mean,
            )
            print("Kriging done, saving output")
            print(anom)
            print(uncert)
            print(grid_obs_2d)

            # reshape output into 2D
            anom = np.reshape(anom, mesh_lat.shape)
            uncert = np.reshape(uncert, mesh_lat.shape)

            # Write the data.
            # This writes each time slice to the netCDF
            krig_anom[timestep, :, :] = anom  # ordinary_kriging
            krig_uncert[timestep, :, :] = uncert  # ordinary_kriging
            grid_obs[timestep, :, :] = grid_obs_2d.astype(np.float32)
            print("-- Wrote data")
            print(timestep, current_month)

        # write time
        time[:] = days_since_by_month(current_year, 15)

        print(time)
        # first print the Dataset object to see what we've got
        print(ncfile)
        # close the Dataset.
        ncfile.close()
        print("Dataset is closed!")


if __name__ == "__main__":
    main()
