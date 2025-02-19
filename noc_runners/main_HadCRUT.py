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
from datetime import datetime
import os
import shutil

from glomar_gridding.distances import euclidean_distance, haversine_distance

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

# self-written modules (from the same directory)
from glomar_gridding.grid import (
    grid_to_distance_matrix,
    map_to_grid,
    assign_to_grid,
    grid_from_resolution,
)
from glomar_gridding.io import load_array, get_recurse
from glomar_gridding.kriging import kriging_ordinary, kriging_simple
from glomar_gridding.perturbation import scipy_mv_normal_draw
from glomar_gridding.utils import (
    init_logging,
    get_date_index,
    get_git_commit,
)
from glomar_gridding.variogram import MaternVariogram, variogram_to_covariance

# Debugging
import logging
import warnings

MULTI: int = 4
ADDER: int = 71


parser = argparse.ArgumentParser()
parser.add_argument(
    "-config",
    dest="config",
    required=False,
    default=os.path.join(os.path.dirname(__file__), "config_HadCRUT.ini"),
    help="Path to yaml file containing configuration settings",
    type=str,
)


def _set_seed(ensemble: int, year: int):
    np.random.seed(ensemble * (10**4) + year)
    return None


def _parse_args(
    parser,
) -> dict:
    args = parser.parse_args()
    with open(args.config, "r") as io:
        config: dict = yaml.safe_load(io)

    return config


def _generate_cov(
    grid: xr.DataArray,
    rng: float,
    sill: float,
    nugget: float,
    nu: float,
    dist_method: str,
) -> np.ndarray:
    match dist_method:
        case "euclidean":
            dist_func = euclidean_distance
        case "haversine":
            dist_func = haversine_distance
        case _:
            raise ValueError(f"Unknown distance method: {dist_method}")
    dist: xr.DataArray = grid_to_distance_matrix(
        grid, dist_func=dist_func, lat_coord="lat", lon_coord="lon"
    )
    variogram = MaternVariogram(
        range=rng, psill=sill, nugget=nugget, nu=nu, method="sklearn"
    )
    cov = variogram_to_covariance(variogram.fit(dist), sill)
    if isinstance(cov, xr.DataArray):
        return cov.values
    return cov


def _get_sst_err_cov(
    current_year: int,
    current_month: int,
    error_covariance_path: str,
    uncorrelated: xr.Dataset,
) -> np.ndarray:
    # Correlated components
    err_cov_fn = (
        f"HadCRUT.5.0.2.0.error_covariance.{current_year}{current_month:02d}.nc"
    )
    error_cov = xr.open_dataset(
        os.path.join(error_covariance_path, err_cov_fn)
    )["tas_cov"].values[0]
    # Uncorrelated components
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
    current_year: int,
    current_month: int,
    error_cov: np.ndarray,
) -> np.ndarray:
    date_int = (current_year - 1850) * 12 + (current_month - 1)
    return np.diag(error_cov[date_int, :])


def _get_obs_groups(
    data_path: str | None,
    var: str,
    year_range: tuple[int, int],
    members: range = range(1, 201),
) -> dict[tuple[object, ...], pl.DataFrame]:
    if data_path is None:
        raise ValueError("'observations_path' key not set in config")

    def _read_file(member: int, data_path=data_path) -> pl.DataFrame:
        data_path = data_path.format(member=member)
        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"{data_path} cannot be found!")
        obs = pl.from_pandas(
            xr.open_dataset(data_path).to_dataframe().reset_index()
        )
        obs = (
            obs.rename({"latitude": "lat", "longitude": "lon"})
            .with_columns(
                [
                    pl.col("time").dt.year().alias("year"),
                    pl.col("time").dt.month().alias("month"),
                    pl.lit(member).alias("member"),
                ]
            )
            .filter(pl.col("year").is_between(*year_range, closed="both"))
            .filter(pl.col(var).is_not_nan() & pl.col(var).is_not_null())
            .unique(subset=["lon", "lat", "time", var])
            .select(["lon", "lat", "year", "month", "member", var])
        )
        return obs

    df = pl.concat(map(_read_file, members), how="diagonal")

    return df.partition_by(by=["year", "month", "member"], as_dict=True)


def _initialise_xarray(
    grid: xr.DataArray,
    variable: str,
    year_range: tuple[int, int],
    member: int,
) -> xr.Dataset:
    # Time dimension is not unlimited
    coords: dict = {
        "time": pl.datetime_range(
            datetime(year_range[0], 1, 15, 12),
            datetime(year_range[1], 12, 15, 12),
            interval="1mo",
            closed="both",
            eager=True,
        ).to_numpy()
    }
    # Add the spatial coordinates of the grid
    coords.update({c: grid.coords[c].values for c in grid.coords})
    ds = xr.Dataset(
        coords=coords,
        attrs={
            "produced": str(datetime.today()),
            "produced_by": "J. T. Siddons",
            "library": "GloMarGridding",
            "url": "https://git.noc.ac.uk/nocsurfaceprocesses/glomar_gridding",
            "git_commit": get_git_commit(),
            "ensemble_member": str(member),
        },
    )

    # Update the attributes of the coordinates
    ds.lat.attr["units"] = "degrees_north"
    ds.lat.attr["long_name"] = "latitude"
    ds.lat.attr["standard_name"] = "latitude"
    ds.lat.attr["axis"] = "Y"

    ds.lon.attr["units"] = "degrees_east"
    ds.lon.attr["long_name"] = "longitude"
    ds.lon.attr["standard_name"] = "longitude"
    ds.lon.attr["axis"] = "X"

    ds.time.attr["long_name"] = "time"

    # Define a 3D variable to hold the data
    ds[f"{variable}_anom"] = xr.DataArray(
        coords=coords,
        name=f"{variable}_anom",
        attrs={
            "standard_name": f"infilled unperturbed {variable} anomaly",
            "long_name": f"infilled unperturbed {variable} anomaly",
            "units": "deg K",  # degrees Kelvin
        },
    )

    # Define a 3D variable to hold the data
    ds[f"{variable}_anom_uncert"] = xr.DataArray(
        coords=coords,
        name=f"{variable}_anom_uncert",
        attrs={
            "standard_name": "kriging uncertainty",
            "long_name": f"{variable} anomaly uncertainty",
            "units": "deg K",  # degrees Kelvin
        },
    )

    # Define a 3D variable to hold the data
    ds["n_obs"] = xr.DataArray(
        coords=coords,
        name="n_obs",
        attrs={
            "standard_name": "Number of observations in each gridcell",
            "units": "",
        },
    )

    # Define a 3D variable to hold the epsilon perturbation value
    ds["epsilon"] = xr.DataArray(
        coords=coords,
        name="epsilon",
        attrs={
            "standard_name": f"{variable} perturbation epsilon",
            "units": "K",
        },
    )

    ds[f"{variable}_anom_perturbed"] = xr.DataArray(
        coords=coords,
        name=f"{variable}_anom_perturbed",
        attrs={
            "standard_name": f"infilled unperturbed {variable} anomaly",
            "long_name": f"infilled unperturbed {variable} anomaly",
            "units": "deg K",  # degrees Kelvin
        },
    )

    return ds


def main():  # noqa: C901, D103
    config = _parse_args(parser)

    config["summary"]["start"] = str(datetime.today())
    config["summary"]["user"] = os.environ["USER"]
    config["summary"]["revision"] = get_git_commit()
    config["summary"]["numpy"] = np.__version__
    config["summary"]["polars"] = pl.__version__
    config["summary"]["xarray"] = xr.__version__

    log_file: str | None = get_recurse(
        config, "setup", "log_file", default=None
    )
    init_logging(log_file)

    logging.info("Loaded configuration")

    # set boundaries for the domain
    lon_west: float = get_recurse(config, "domain", "west", default=-180.0)
    lon_east: float = get_recurse(config, "domain", "east", default=180.0)
    lat_south: float = get_recurse(config, "domain", "south", default=-90.0)
    lat_north: float = get_recurse(config, "domain", "north", default=90.0)
    year_start: int = get_recurse(config, "domain", "startyear", default=1850)
    year_stop: int = get_recurse(config, "domain", "endyear", default=2023)
    member_start: int = get_recurse(config, "domain", "startmember", default=1)
    member_stop: int = get_recurse(config, "domain", "endmember", default=200)

    members = range(member_start, member_stop + 1)
    year_list = range(year_start, year_stop + 1)
    months = range(1, 13)

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
    logging.info(f"{output_lat = }")
    logging.info(f"{output_lon = }")

    # what variable is being processed
    variable: str = get_recurse(config, "domain", "variable", default="sst")
    hadcrut_var: str = "tos" if variable == "sst" else "tas"

    # path to output directory
    interpolation_covariance_type: str = get_recurse(
        config, variable, "interpolation_covariance_type", default="euclidean"
    )
    output_directory: str = get_recurse(
        config, "output", "path", default=os.path.dirname(__file__)
    )
    output_directory = os.path.join(
        output_directory, interpolation_covariance_type, hadcrut_var
    )
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    logging.info(f"{output_directory = }")
    file_copy = os.path.join(output_directory, os.path.basename(__file__))
    config_copy = os.path.join(output_directory, "config.yaml")
    config["summary"]["file_copy"] = file_copy
    logging.info(f"Copying this file to {file_copy}")
    shutil.copyfile(os.path.abspath(__file__), file_copy)

    # var_range = config.getfloat(variable, "range")
    # var_sigma = config.getfloat(variable, "sigma")
    # var_matern = config.getfloat(variable, "matern")

    interpolation_covariance_path: str | None = get_recurse(
        config,
        variable,
        f"{interpolation_covariance_type}_interpolation_covariance_path",
    )
    interp_covariance = None
    if interpolation_covariance_type in ["euclidean", "haversine"]:
        if interpolation_covariance_path is None:
            rng: float = get_recurse(config, variable, "range", default=1300.0)
            sill: float = (
                get_recurse(config, variable, "sigma", default=0.6) ** 2
            )
            nugget: float = get_recurse(config, variable, "nugget", default=0.0)
            nu: float = get_recurse(config, variable, "matern", default=1.5)

            interp_covariance = _generate_cov(
                output_grid,
                rng=rng,
                sill=sill,
                nugget=nugget,
                nu=nu,
                dist_method=interpolation_covariance_type,
            )
            logging.info("created interpolation covariance")
        elif interpolation_covariance_path.endswith(".nc"):
            interp_covariance = load_array(
                interpolation_covariance_path, var="covariance"
            ).values
            logging.info("loaded interpolation covariance")
        else:  # is a numpy file
            interp_covariance = np.load(interpolation_covariance_path)
            logging.info("loaded interpolation covariance")
        print(f"{interp_covariance = }")
    if (
        interpolation_covariance_type == "ellipse"
        and interpolation_covariance_path is None
    ):
        raise ValueError(
            "interpolation_covariance_path must be specified if "
            + "interpolation_covariance_type is 'ellipse'"
        )

    data_path: str = get_recurse(config, variable, "observations_path")
    yr_mo = _get_obs_groups(
        data_path,
        hadcrut_var,
        year_range=(year_start, year_stop),
        members=members,
    )
    logging.info("Loaded Observations")

    error_covariance_path: str = get_recurse(
        config, variable, "error_covariance_path"
    )

    match variable:
        case "sst":
            logging.info(f"Processing for variable {variable} | {hadcrut_var}")
            uncorrelated_uncertainty = config.get(variable, {}).get(
                "uncorrelated_uncertainty", ""
            )
            if not os.path.isfile(uncorrelated_uncertainty):
                raise FileNotFoundError(
                    "Cannot find {uncorrelated_uncertainty = }"
                )

            uncorrelated = xr.open_dataset(uncorrelated_uncertainty)

            def get_error_cov(year: int, month: int) -> np.ndarray:
                return _get_sst_err_cov(
                    year,
                    month,
                    error_covariance_path,
                    uncorrelated,
                )

        case "lsat":
            logging.info(f"Processing for variable {variable} | {hadcrut_var}")

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

    for member in members:
        logging.info(f"Starting for ensemble member {member}")
        ds = _initialise_xarray(
            grid=output_grid,
            variable=variable,
            year_range=(year_start, year_stop),
            member=member,
        )
        out_filename = f"kriged_member_{member:03d}.nc"
        out_filename = os.path.join(output_directory, out_filename)

        for year in year_list:
            # Set the seed based on the ensemble member and year combination for
            # reproducibility
            np.random.seed(_set_seed(member, year))

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

            for month in months:
                timestep = get_date_index(year, month, year_start)
                print("Current month and year: ", (month, year))

                mon_df: pl.DataFrame = yr_mo.get(
                    (year, month, member), pl.DataFrame()
                )
                if mon_df.height == 0:
                    warnings.warn(
                        f"Current year, month ({year}, {month}) "
                        + "has no data. Skipping."
                    )
                    continue
                print(f"{mon_df = }")

                if (
                    interpolation_covariance_type == "ellipse"
                    and interpolation_covariance_path is not None
                ):
                    interp_covariance = xr.open_dataset(
                        os.path.join(
                            interpolation_covariance_path,
                            f"covariance_{month:02d}_v_eq_1p5_{variable}_clipped.nc",
                        )
                    )["covariance"].values
                    logging.info("Loaded ellipse interpolation covariance")
                    y = scipy_mv_normal_draw(
                        np.zeros(interp_covariance.shape[0]),
                        interp_covariance,
                    )
                    print(f"{interp_covariance = }")

                error_covariance = get_error_cov(year, month)
                logging.info("Got Error Covariance")
                print(f"{error_covariance = }")

                if y is None or interp_covariance is None:
                    logging.error(
                        "Failed to get interp_covariance or y. Skipping"
                    )
                    continue

                mon_df = map_to_grid(
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
                logging.info(
                    "Added error covariance diagonal to the observations"
                )

                # count obs per grid for output
                gridbox_counts = mon_df["grid_idx"].value_counts()
                grid_obs_2d = assign_to_grid(
                    gridbox_counts["count"].to_numpy(),
                    gridbox_counts["grid_idx"].to_numpy(),
                    output_grid,
                )
                # need to either add weights (which will be just 1 or 0
                # everywhere as obs are gridded)
                # krige obs onto gridded field

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
                obs_to_obs_cov = (  # LHS
                    interp_covariance[grid_idx[:, None], grid_idx[None, :]]
                    + error_covariance
                )
                obs_to_grid_cov = (  # RHS
                    np.asarray(interp_covariance[grid_idx, :])
                    # + error_covariance
                )

                anom, uncert = kriging_ordinary(
                    obs_to_obs_cov,
                    obs_to_grid_cov,
                    grid_obs=mon_df.get_column(hadcrut_var).to_numpy(),
                    interp_cov=interp_covariance,
                )
                logging.info("Kriging done for observations")

                # Kriging for the random drawn observations for perturbations
                logging.info("Starting Kriging for random draw")
                # This generates a _simiulated_ gridded field
                simulated_anom, _ = kriging_simple(
                    obs_to_obs_cov,
                    obs_to_grid_cov,
                    grid_obs=y_obs_prime,
                    interp_cov=interp_covariance,
                )
                logging.info("Kriging done for random draw")
                logging.info(
                    "Computing and applying epsilon to the kriged output"
                )
                epsilon = simulated_anom - y

                print(f"{anom = }")
                print(f"{np.all(np.isnan(anom)) = }")
                print(f"{np.any(np.isnan(anom)) = }")
                print("-" * 10)
                print(f"{uncert = }")
                print(f"{grid_obs_2d = }")

                # reshape output into 2D
                anom = np.reshape(anom, output_grid.shape)
                uncert = np.reshape(uncert, output_grid.shape)
                epsilon = np.reshape(epsilon, output_grid.shape)
                anom_perturbed = anom + epsilon
                logging.info("Reshaped kriging outputs")

                # Write the data.
                # This writes each time slice to the netCDF
                ds[f"{variable}_anom"][timestep, :, :] = anom
                ds[f"{variable}_anom_perturbed"][timestep, :, :] = (
                    anom_perturbed
                )
                ds[f"{variable}_anom_uncert"][timestep, :, :] = uncert
                ds["epsilon"][timestep, :, :] = epsilon
                ds["n_obs"][timestep, :, :] = grid_obs_2d.astype(np.float32)
                logging.info(f"Wrote data for {year = }, {month = }")

            # first print the Dataset object to see what we've got
            # close the Dataset.
            ds.to_netcdf(out_filename)
            logging.info("Dataset is closed!")

    config["summary"]["end"] = str(datetime.today())
    logging.info("DONE")
    with open(config_copy, "w") as io:
        yaml.safe_dump(config, io)


if __name__ == "__main__":
    main()
