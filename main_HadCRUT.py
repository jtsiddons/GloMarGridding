#!/usr/bin/env python
"""
Script to run Kriging for HadCRUT.

By A. Faulkner for python version 3.0 and up.
Modified by J. Siddons (2025-01). Requires python >= 3.11.

Encodes the uncertainty from the sampling into the gridded field for the
ensemble. This is done by generating a simulated field and observations and
computing a simulated gridded field.
See: https://doi.org/10.1029/2019JD032361
"""

# global
import os

if "POLARS_MAX_THREADS" not in os.environ:
    os.environ["POLARS_MAX_THREADS"] = "16"

# argument parser
import argparse
import yaml

# math tools
import numpy as np

# data handling tools
import polars as pl
import xarray as xr
import netCDF4 as nc

# self-written modules (from the same directory)
from glomar_gridding.grid import (
    align_to_grid,
    assign_to_grid,
    grid_from_resolution,
)
from glomar_gridding.error_covariance import get_weights
from glomar_gridding.kriging import kriging
from glomar_gridding.utils import days_since_by_month, init_logging
from glomar_gridding.perturbation import scipy_mv_normal_draw

# Debugging
import logging
import warnings

np.random.seed(12345)


parser = argparse.ArgumentParser()
parser.add_argument(
    "-config",
    dest="config",
    required=False,
    default=os.path.join(os.path.dirname(__file__), "config_HadCRUT.ini"),
    help="Path to yaml file containing configuration settings",
    type=str,
)
parser.add_argument(
    "-year_start",
    dest="year_start",
    required=False,
    help="start year",
    type=int,
)
parser.add_argument(
    "-year_stop",
    dest="year_stop",
    required=False,
    help="end year",
    type=int,
)
parser.add_argument(
    "-member",
    dest="member",
    required=True,
    help="ensemble member: required argument",
    type=int,
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
    help="Should the global mean be removed? - 0:no, 1:yes, "
    + "2:yes but median, 3:yes but spatial mean",
    choices=[0, 1, 2, 3],
)


def _parse_args(parser) -> tuple[dict, dict, int, int, int, str, str, str, int]:
    args = parser.parse_args()
    with open(args.config, "r") as io:
        config: dict = yaml.safe_load(io)
    year_start: int = args.year_start or config.get("domain", {}).get(
        "startyear", 1850
    )
    year_stop: int = args.year_stop or config.get("domain", {}).get(
        "endyear", 2023
    )
    variable = args.variable
    var_config: dict = config.get(variable, {})
    if not var_config:
        raise ValueError(f"Cannot get variable configuration for {variable}")

    member = args.member
    if member < 1 or member > 200:
        raise ValueError(
            f"Ensemble member must be between 1 and 200, got {member}"
        )

    return (
        config,
        var_config,
        year_start,
        year_stop,
        member,
        variable,
        args.method,
        args.interpolation,
        args.remove_obs_mean,
    )


def _get_sst_err_cov(
    current_year: int,
    current_month: int,
    error_covariance_path: str,
    # uncorrelated: xr.Dataset,
) -> np.ndarray:
    err_cov_fn = (
        f"HadCRUT.5.0.2.0.error_covariance.{current_year}{current_month:02d}.nc"
    )
    error_cov = xr.open_dataset(
        os.path.join(error_covariance_path, err_cov_fn)
    )["tas_cov"].values[0]
    return error_cov


def _get_lsat_err_cov(
    current_year: int,
    current_month: int,
    error_cov: np.ndarray,
) -> np.ndarray:
    date_int = (current_year - 1850) * 12 + (current_month - 1)
    return np.diag(error_cov[date_int, :])


def _get_obs_groups(
    data_path: str,
    var: str,
    **kwargs,
) -> dict[tuple[object, ...], pl.DataFrame]:
    if kwargs:
        data_path = data_path.format(**kwargs)
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Cannot find observations file {data_path}.")
    obs = pl.from_pandas(
        xr.open_dataset(data_path).to_dataframe().reset_index()
    )
    obs = (
        obs.rename({"latitude": "lat", "longitude": "lon"})
        .unique(subset=["lon", "lat", "time", var])
        .filter(pl.col(var).is_not_nan() & pl.col(var).is_not_null())
        .with_columns(
            [
                pl.col("time").dt.year().alias("year"),
                pl.col("time").dt.month().alias("month"),
            ]
        )
    )
    return obs.partition_by(by=["year", "month"], as_dict=True)


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


def main():  # noqa: C901, D103
    init_logging()
    (
        config,
        var_config,
        year_start,
        year_stop,
        member,
        variable,
        method,
        interpolation_covariance_type,
        remove_obs_mean,
    ) = _parse_args(parser)

    logging.info("Loaded configuration")
    print(f"{config = }")

    # set boundaries for the domain
    lon_west: float = config.get("domain", {}).get("west", -180.0)
    lon_east: float = config.get("domain", {}).get("east", 180.0)
    lat_south: float = config.get("domain", {}).get("south", -90.0)
    lat_north: float = config.get("domain", {}).get("north", 90.0)

    output_grid: xr.DataArray = grid_from_resolution(
        resolution=5.0,
        bounds=[
            (lat_south + 2.5, lat_north + 2.5),
            (lon_west + 2.5, lon_east + 2.5),
        ],
        coord_names=["lat", "lon"],
    )
    output_lat: np.ndarray = output_grid.coords["lat"].values
    output_lon: np.ndarray = output_grid.coords["lon"].values
    logging.info("Initialised Output Grid")
    print(f"{output_lat = }")
    print(f"{output_lon = }")

    # what variable is being processed
    hadcrut_var: str = "tos" if variable == "sst" else "tas"

    # path to output directory
    output_directory: str = config.get("output", {}).get("path")
    output_directory = os.path.join(
        output_directory, interpolation_covariance_type, hadcrut_var
    )
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    print(f"{output_directory = }")

    # var_range = config.getfloat(variable, "range")
    # var_sigma = config.getfloat(variable, "sigma")
    # var_matern = config.getfloat(variable, "matern")

    interpolation_covariance_path: str = var_config.get(
        f"{interpolation_covariance_type}_interpolation_covariance", ""
    )
    interp_covariance = None
    if interpolation_covariance_type == "distance":
        interp_covariance = np.load(interpolation_covariance_path)
        logging.info("loaded interpolation covariance")
        print(f"{interp_covariance = }")

    data_path: str = var_config.get("observations", "")
    yr_mo = _get_obs_groups(data_path, hadcrut_var, member=member)
    logging.info("Loaded Observations")

    error_covariance_path: str = var_config.get("error_covariance", "")

    match variable:
        case "sst":
            print(f"Processing for variable {variable} | {hadcrut_var}")
            if "sampling_uncertainty" in var_config:
                single_sigma_warn_msg = (
                    "Option sampling_uncertainty for sst is ignored. "
                    + "HadCRUT5 only has a single uncorrelated sigma; "
                    + "if you are using multiple uncorrelated sigmas "
                    + "(e.g. HadSST4), combine them first."
                )
                warnings.warn(single_sigma_warn_msg, DeprecationWarning)
            # uncorrelated_uncertainty = config.get(
            #     variable, "uncorrelated_uncertainty"
            # )

            # uncorrelated = xr.open_dataset(uncorrelated_uncertainty)

            def get_error_cov(year: int, month: int) -> np.ndarray:
                return _get_sst_err_cov(
                    year,
                    month,
                    error_covariance_path,  # uncorrelated
                )

        case "lsat":
            print(f"Processing for variable {variable} | {hadcrut_var}")

            error_cov = np.load(
                os.path.join(
                    error_covariance_path, "HadCRUT.5.0.2.0.uncorrelated.npz"
                )
            )["err_cov"]
            logging.info("loaded error covariance")
            print(f"{error_cov = }")

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

        # Draw from N(0, interp_covariance)
        y = None
        if (
            interpolation_covariance_type == "distance"
            and interp_covariance is not None
        ):
            y = scipy_mv_normal_draw(
                np.zeros(interp_covariance.shape[0]),
                interp_covariance,
            )

        ncfilename = f"{current_year}_kriged"
        if member:
            ncfilename += f"_member_{member:03d}"
        ncfilename += ".nc"
        ncfilename = os.path.join(output_directory, ncfilename)

        ncfile = nc.Dataset(ncfilename, mode="w", format="NETCDF4_CLASSIC")

        lon, lat, time, krig_anom, krig_uncert, grid_obs = _initialise_ncfile(
            ncfile, output_lon, output_lat, current_year, variable
        )
        logging.info(f"Initialised output file for {current_year = }")

        # Write latitudes, longitudes.
        # Note: the ":" is necessary in these "write" statements
        lat[:] = output_lat  # ds.lat.values
        lon[:] = output_lon  # ds.lon.values

        for timestep, current_month in enumerate(month_list):
            print("Current month and year: ", (current_month, current_year))

            mon_df: pl.DataFrame = yr_mo.get(
                (current_year, current_month), pl.DataFrame()
            )
            if mon_df.height == 0:
                warnings.warn(
                    f"Current year, month ({current_year}, {current_month}) "
                    + "has no data. Skipping."
                )
                continue
            print(f"{mon_df = }")

            if interpolation_covariance_type == "ellipse":
                interp_covariance = xr.open_dataset(
                    os.path.join(
                        interpolation_covariance_path,
                        f"covariance_{current_month:02d}_v_eq_1p5_{variable}_clipped.nc",
                    )
                )["covariance"].values
                logging.info("Loaded ellipse interpolation covariance")
                y = scipy_mv_normal_draw(
                    np.zeros(interp_covariance.shape[0]),
                    interp_covariance,
                )
                print(f"{interp_covariance = }")

            error_covariance = get_error_cov(current_year, current_month)
            logging.info("Got Error Covariance")
            print(f"{error_covariance = }")

            ec_1 = error_covariance[~np.isnan(error_covariance)]
            ec_2 = ec_1[np.nonzero(ec_1)]
            print("Non-nan and non-zero error covariance =", ec_2, len(ec_2))
            ec_idx = np.argwhere(
                np.logical_and(
                    ~np.isnan(error_covariance), error_covariance != 0.0
                )
            )
            print("Index of non-nan and non-zero values =", ec_idx, len(ec_idx))

            if y is None or interp_covariance is None:
                logging.error("Failed to get interp_covariance or y. Skipping")
                continue

            mon_df = align_to_grid(
                mon_df, output_grid, grid_coords=["lat", "lon"]
            )
            logging.info("Aligned observations to output grid")

            error_cov_diag_at_obs = pl.Series(
                "error_covariance_diagonal",
                np.diag(error_covariance)[mon_df.get_column("grid_idx")],
            )
            mon_df = mon_df.with_columns(error_cov_diag_at_obs)
            mon_df = mon_df.filter(
                pl.col("error_covariance_diagonal").is_not_nan()
                & pl.col("error_covariance_diagonal").is_not_null()
            )
            logging.info("Added error covariance diagonal to the observations")

            # count obs per grid for output
            gridbox_counts = mon_df["grid_idx"].value_counts()
            grid_obs_2d = assign_to_grid(
                gridbox_counts["count"].to_numpy(),
                gridbox_counts["grid_idx"].to_numpy(),
                output_grid,
            )
            logging.info("Got grid_idx counts")
            # need to either add weights (which will be just 1 or 0 everywhere
            # as obs are gridded)
            # krige obs onto gridded field
            W = get_weights(mon_df)
            logging.info("Got Weights")

            grid_idx = mon_df.get_column("grid_idx").to_numpy()

            # Sub-sample error covariance at observation points
            error_covariance = error_covariance[
                grid_idx[:, None], grid_idx[None, :]
            ]
            # Draw simulated observations
            y_obs = y[mon_df.get_column("grid_idx")]
            y_obs_prime: np.ndarray = y_obs + scipy_mv_normal_draw(
                np.zeros(error_covariance.shape[0]),
                error_covariance,
            )

            logging.info("Starting Kriging for observations")
            # Kriging the observations for the ensemble member
            anom, uncert = kriging(
                grid_idx,
                W,
                mon_df.get_column(hadcrut_var).to_numpy(),
                interp_covariance,
                error_covariance,
                method=method,
                remove_obs_mean=remove_obs_mean,
            )
            logging.info("Kriging done for observations")

            # Kriging for the random drawn observations for perturbations
            logging.info("Starting Kriging for random draw")
            # This generates a _simiulated_ gridded field
            simulated_anom, _ = kriging(
                grid_idx,
                W,
                y_obs_prime,
                interp_covariance,
                error_covariance,
                method="simple",
                remove_obs_mean=remove_obs_mean,
            )
            logging.info("Kriging done for random draw")
            logging.info("Computing and applying epsilon to the kriged output")
            epsilon = simulated_anom - y
            # reshape output into 2D
            anom = anom + epsilon
            anom = np.reshape(anom, output_grid.shape)

            print(f"{anom = }")
            print(f"{np.all(np.isnan(anom)) = }")
            print(f"{np.any(np.isnan(anom)) = }")
            print("-" * 10)
            print(f"{uncert = }")
            print(f"{grid_obs_2d = }")

            uncert = np.reshape(uncert, output_grid.shape)
            logging.info("Reshaped kriging outputs")

            # Write the data.
            # This writes each time slice to the netCDF
            krig_anom[timestep, :, :] = anom  # ordinary_kriging
            krig_uncert[timestep, :, :] = uncert  # ordinary_kriging
            grid_obs[timestep, :, :] = grid_obs_2d.astype(np.float32)
            logging.info(
                f"Wrote data for {current_year = }, {current_month = }"
            )

        # write time
        time[:] = days_since_by_month(current_year, 15)

        print(f"{time = }")
        # first print the Dataset object to see what we've got
        # close the Dataset.
        ncfile.close()
        logging.info("Dataset is closed!")


if __name__ == "__main__":
    main()
