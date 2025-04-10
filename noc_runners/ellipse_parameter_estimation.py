#!/usr/bin/env python

"""Run script for SST and LSAT nonstationary variogram parameter estimation"""

import argparse
import logging
import os
# import psutil

import numpy as np
import xarray as xr
import yaml


from glomar_gridding.covariance_cube import EllipseBuilder
from glomar_gridding.ellipse import EllipseModel
from glomar_gridding.io import get_recurse, load_array
from glomar_gridding.utils import init_logging


parser = argparse.ArgumentParser(
    prog="Ellipse Parameter Estimation",
    description="Run script for SST and LSAT nonstationary variogram parameter estimation",  # noqa: E501
)
parser.add_argument(
    "-c",
    "--config",
    type=str,
    default="ellipse_parameter_estimation_config.yaml",
    help="path to config file",
)
parser.add_argument(
    "-v",
    "--matern-shape",
    type=float,
    default=1.5,
    help="Matern shape parameter",
)
parser.add_argument(
    "-m",
    "--month",
    type=int,
    default=1,
    choices=list(range(1, 13)),
    help="Month to process",
)
# fmt: off
parser.add_argument(
    "-d",
    "--data-type",
    type=str,
    choices=[
        "sst",  # Sea surface temp
        "at",   # Land surface air temp
        "dpt",  # Dew point temp
        "slp",  # Sea level pressure
        "ws",   # Wind speed
        "cl",   # Cloud cover
    ],
    default="sst",
    help="Data type / Variable",
)
# fmt: on
parser.add_argument(
    "-a",
    "--anisotropic",
    action="store_true",
    help="Fit an ellipse rather than a circle",
)
parser.add_argument(
    "-r",
    "--rotated",
    action="store_true",
    help="Can the ellipse be rotated, cannot be set if 'anisotropic' is set",
)
parser.add_argument(
    "-p",
    "--physical-distance",
    action="store_true",
    help="Use physical distances in Ellipse fitting",
)


def parse_args(parser) -> tuple[dict, EllipseModel, str, int]:
    """Parse input arguments"""
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as io:
        conf = yaml.safe_load(io)

    ellipse = EllipseModel(
        anisotropic=args.anisotropic,
        rotated=args.rotated,
        physical_distance=args.physical_distance,
        v=args.matern_shape,
        unit_sigma=True,
    )
    variable = args.data_type
    month = args.month

    return conf, ellipse, variable, month


def load_sst(file: str) -> xr.DataArray:
    """Load SST inputs"""
    arr = load_array(file)
    # icc.add_month_number(arr, "time", name="month_number")
    return arr


def load_at(file: str) -> xr.DataArray:
    """Load LSAT inputs"""
    arr = load_array(file, "land 2 metre temperature")
    return arr


def load_data(conf: dict, variable: str) -> xr.DataArray:
    """Load data"""
    in_path = get_recurse(conf, variable, "in_path")
    if in_path is None:
        raise KeyError(f"Missing key {variable}.in_path in config file")
    match variable:
        case "sst":
            return load_sst(in_path)
        case "at":
            return load_at(in_path)
        case _:
            raise NotImplementedError(
                f"Not implemented for data variable: {variable}"
            )


# def mask_time_union(cube):
#     """
#     Make sure mask is same for all time
#     If a single masking occur,
#     masks all other time as well
#     """
#     cube_mask = cube.data.mask
#     common_mask = np.any(cube_mask, axis=0)
#     cube.data.mask = common_mask
#     return cube


def main():  # noqa: D103
    init_logging(level="WARN")

    logging.info("Start")

    # logging.info(psutil.Process().cpu_affinity())
    # nCPUs = len(psutil.Process().cpu_affinity())
    # logging.info("len(cpu_affinity) = ", nCPUs)

    conf, ellipse, variable, month_value = parse_args(parser)

    v = ellipse.v
    nparms = ellipse.supercategory_n_params
    default_values = [-999.9 for _ in range(nparms)]

    out_path = get_recurse(conf, variable, "out_path")
    if out_path is None:
        raise KeyError(f"Missing key {variable}.out_path in config file")

    logging.info(f"{v = }")

    data_array = load_data(conf, variable)
    data_array = data_array.where(
        data_array.time.dt.month == month_value,
        drop=True,
    )

    # Initialise Output arrays
    outputs: list[np.ndarray] = [
        np.ones_like(data_array.values[0, :, :]) * default
        for default in default_values
    ]

    # Init values set to HadCRUT5 defaults
    # no prior distrubtion set around those value
    init_values = [2000.0, 2000.0, 0]
    # Uniformative prior of parameter range
    fit_bounds = [
        (300.0, 30000.0),
        (300.0, 30000.0),
        (-2.0 * np.pi, 2.0 * np.pi),
    ]

    logging.info(repr(data_array))
    logging.debug(f"{data_array.coord['latitude'] = }")
    logging.debug(f"{data_array.coord['longitude'] = }")
    logging.debug(f"{data_array.coord['time'] = }")

    logging.info("Building covariance matrix")
    cov_cube = EllipseBuilder(data_array.values, data_array.coords)
    logging.info("Covariance matrix completed")

    fit_max_distance = 6000.0 if ellipse.physical_distance else 60.0

    for mask_i, (grid_i, grid_j) in enumerate(
        zip(cov_cube.xi_masked, cov_cube.yi_masked)
    ):
        # current_lon = data_array.coords["longitude"][grid_i]
        # current_lat = data_array.coords["latitude"][grid_j]

        # Note:
        # Possible cause for convergence failure are ENSO grid points;
        # max_distance is originally introduced to keep moving window
        # fits consistent (i.e. always using 20x20 deg squares around
        # central gp), but is too small for ENSO signals.
        # Now with global inputs this can be relaxed, and use of global
        # inputs will ensure correlations from far away grid points be
        # accounted for <--- this cannot be done for moving window fits.
        result = cov_cube.fit_ellipse_model(
            mask_i,
            matern_ellipse=ellipse,
            max_distance=fit_max_distance,
            guesses=init_values,
            bounds=fit_bounds,
            # n_jobs=nCPUs,
        )
        for output, param in zip(outputs, result["ModelParams"]):
            output[grid_i, grid_j] = param

    output_coordinates = xr.Coordinates(
        {
            "latitude": cov_cube.coords["latitude"],
            "longitude": cov_cube.coords["longitude"],
        }
    )
    output_dataset = xr.Dataset(coords=output_coordinates)
    # TODO: Attrs
    for output, (variable_name, unit) in zip(
        outputs, ellipse.supercategory_params.items()
    ):
        output_dataset[variable_name] = output
        output_dataset[variable_name].attrs["standard_name"] = variable_name
        output_dataset[variable_name].attrs["unit"] = unit

    logging.info("Grid box loop is completed")

    vstring = str(v).replace(".", "p")
    vstring = "_v_eq_" + vstring

    outpath = out_path + "matern_physical_distances" + vstring
    os.makedirs(out_path)

    outncfilename = os.path.join(outpath, f"{variable}_{month_value:02d}.nc")
    logging.info("Saving outputs")
    output_dataset.to_netcdf(outncfilename)


if __name__ == "__main__":
    main()
