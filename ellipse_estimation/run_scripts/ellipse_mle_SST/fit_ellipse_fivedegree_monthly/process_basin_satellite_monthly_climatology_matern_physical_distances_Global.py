"""Run script for SST and LSAT nonstationary variogram parameter estimation"""

import argparse
import logging
from pathlib import Path

import iris
from iris import coord_categorisation as icc
from iris.fileformats import netcdf as inc
from iris.util import equalise_attributes
import numpy as np
import yaml

import psutil

from glomar_gridding import covariance_cube
from glomar_gridding.ellipse import MaternEllipseModel
from glomar_gridding.utils import init_logging
from glomar_gridding.io import get_recurse


parser = argparse.ArgumentParser(
    prog="Ellipse Parameter Estimation",
    description="Run script for SST and LSAT nonstationary variogram parameter estimation",  # noqa: E501
)
parser.add_argument(
    "-c",
    "--config",
    type=str,
    default="process_basin_satellite_monthly_climatology_matern_physical_distances_Global.yaml",
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


def parse_args(parser) -> tuple[dict, MaternEllipseModel, str, int]:
    """Parse input arguments"""
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as io:
        conf = yaml.safe_load(io)

    ellipse = MaternEllipseModel(
        anisotropic=args.anisotropic,
        rotated=args.rotated,
        physical_distance=args.physical_distance,
        v=args.matern_shape,
        unit_sigma=True,
    )
    variable = args.data_type
    month = args.month

    return conf, ellipse, variable, month


def load_sst(file: str) -> iris.cube.Cube:
    """Load SST inputs"""
    cube = iris.load_cube(file)
    icc.add_month_number(cube, "time", name="month_number")
    return cube


def load_at(file: str) -> iris.cube.Cube:
    """Load LSAT inputs"""
    cube = iris.load_cube(file, "land 2 metre temperature")
    icc.add_month_number(cube, "time", name="month_number")
    return cube


def load_data(conf: dict, variable: str) -> iris.cube.Cube:
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


def mask_time_union(cube):
    """
    Make sure mask is same for all time
    If a single masking occur,
    masks all other time as well
    """
    cube_mask = cube.data.mask
    common_mask = np.any(cube_mask, axis=0)
    cube.data.mask = common_mask
    return cube


def main():  # noqa: D103
    init_logging(level="WARN")

    logging.info("Start")

    logging.info(psutil.Process().cpu_affinity())
    nCPUs = len(psutil.Process().cpu_affinity())
    logging.info("len(cpu_affinity) = ", nCPUs)

    conf, ellipse, variable, month_value = parse_args(parser)

    v = ellipse.v
    nparms = ellipse.supercategory_n_params
    defval = [-999.9 for _ in range(nparms)]

    everyother = 1

    out_path = get_recurse(conf, variable, "out_path")
    if out_path is None:
        raise KeyError(f"Missing key {variable}.out_path in config file")

    logging.info(f"{v = }")

    additional_constraints = iris.Constraint(month_number=month_value)
    data_cube = load_data(conf, variable)
    data_cube = data_cube.extract(additional_constraints)
    data_cube = mask_time_union(data_cube)

    data_cube_time_length = len(data_cube.coord("time").points)
    logging.info(repr(data_cube))

    # Init values set to HadCRUT5 defaults
    # no prior distrubtion set around those value
    init_values = [2000.0, 2000.0, 0]
    # Uniformative prior of parameter range
    fit_bounds = [
        (300.0, 30000.0),
        (300.0, 30000.0),
        (-2.0 * np.pi, 2.0 * np.pi),
    ]

    super_cube_list = iris.cube.CubeList()

    logging.info(repr(data_cube))
    logging.debug(f"{data_cube.coord('latitude') = }")
    logging.debug(f"{data_cube.coord('longitude') = }")
    logging.debug(f"{data_cube.coord('time') = }")

    logging.info("Building covariance matrix")
    cov_cube = covariance_cube.CovarianceCube(data_cube)
    logging.info("Covariance matrix completed")

    data_cube_not_template = data_cube[data_cube_time_length // 2]
    for zonal, zonal_slice in enumerate(
        data_cube_not_template.slices(["longitude"])
    ):
        # Zonal slices
        logging.info(f"{zonal} {repr(zonal_slice)}")
        if (zonal % everyother) != 0:
            continue
        zonal_cube_list = iris.cube.CubeList()
        for box_count, invidiual_box in enumerate(zonal_slice.slices([])):
            if (box_count % everyother) != 0:
                continue
            logging.info(f"{zonal} || {box_count} {repr(invidiual_box)}")

            current_lon = invidiual_box.coord("longitude").points[0]
            current_lat = invidiual_box.coord("latitude").points[0]
            logging.info(f"{zonal} || {box_count} {current_lon} {current_lat}")
            if np.ma.is_masked(invidiual_box.data):
                xy, actual_latlon = (
                    cov_cube.find_nearest_xy_index_in_cov_matrix(
                        [current_lon, current_lat], use_full=True
                    )
                )
                logging.debug(cov_cube.data_cube)
                logging.debug(cov_cube.data_cube.coord("latitude"))
                logging.debug(cov_cube.data_cube.coord("longitude"))
                template_cube = cov_cube._make_template_cube2(
                    (current_lon, current_lat)
                )
                ans = covariance_cube.create_output_cubes(
                    template_cube,
                    model_type=ellipse.model_type,
                    additional_meta_aux_coords=[
                        covariance_cube.make_v_aux_coord(v)
                    ],
                    default_values=defval,
                )["param_cubelist"]
                ansH = "MASKED"
            else:
                # Nearest valid point
                xy, actual_latlon = (
                    cov_cube.find_nearest_xy_index_in_cov_matrix(
                        [current_lon, current_lat]
                    )
                )
                # Note:
                # Possible cause for convergence failure are ENSO grid points;
                # max_distance is originally introduced to keep moving window
                # fits consistent (i.e. always using 20x20 deg squares around
                # central gp), but is too small for ENSO signals.
                # Now with global inputs this can be relaxed, and use of global
                # inputs will ensure correlations from far away grid points be
                # accounted for <--- this cannot be done for moving window fits.
                ansX = cov_cube.fit_ellipse_model(
                    xy,
                    matern_ellipse=ellipse,
                    max_distance=60.0,
                    guesses=init_values,
                    bounds=fit_bounds,
                    n_jobs=nCPUs,
                )
                ans = ansX["Model_as_1D_cube"]
                ansH = (ansX["Model"].x, ansX["Model"].x[-1] * 180.0 / np.pi)
            ans_lon = ans[0].coord("longitude").points
            ans_lat = ans[0].coord("latitude").points
            logging.debug(
                f"{zonal} || "
                + f"{box_count} {xy} {actual_latlon} {ans_lon} {ans_lat} {ansH}"
            )
            for individual_ans in ans:
                zonal_cube_list.append(individual_ans)
                zonal_cube_list.concatenate()
        for zonal_ans_cube in zonal_cube_list:
            super_cube_list.append(zonal_ans_cube)
            equalise_attributes(super_cube_list)
    logging.info("Grid box loop is completed")
    equalise_attributes(super_cube_list)
    try:
        super_cube_list = super_cube_list.concatenate()
    except Exception as e:
        logging.error(f"Error concatenating cubelist: {e}")

    vstring = str(v).replace(".", "p")
    vstring = "_v_eq_" + vstring

    outpath = out_path + "matern_physical_distances" + vstring + "/"
    Path(outpath).mkdir(parents=True, exist_ok=True)

    outncfilename = outpath + variable + "_"
    outncfilename += f"{month_value:02d}.nc"
    logging.debug("Results to be saved...")
    logging.debug(super_cube_list)
    logging.info(f"Saving results to {outncfilename}")
    inc.save(super_cube_list, outncfilename)
    logging.info("Completed")


if __name__ == "__main__":
    main()
