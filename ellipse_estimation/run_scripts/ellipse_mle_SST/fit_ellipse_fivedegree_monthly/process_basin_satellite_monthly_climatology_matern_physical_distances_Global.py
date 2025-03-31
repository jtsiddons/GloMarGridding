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

from ellipse_estimation import cube_covariance
from glomar_gridding.ellipse import MaternEllipseModel
from glomar_gridding.utils import init_logging


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
    help="Month to process",
)
parser.add_argument(
    "-d",
    "--data-type",
    type=str,
    choices=["sst", "lsat"],
    default="sst",
    help="Data type / Variable",
)
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


def parse_args(parser):
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        conf = yaml.safe_load(f)

    ellipse = MaternEllipseModel(
        anisotropic=args.anisotropic,
        rotated=args.rotated,
        physical_distance=args.physical_distance,
        v=args.matern_shape,
        unit_sigma=True,
    )
    dat_type = args.data_type
    month = args.month

    return conf, ellipse, dat_type, month


def load_sst(conf):
    """Load SST inputs"""
    ncfile = conf["sst_in"]
    cube = iris.load_cube(ncfile)
    icc.add_month_number(cube, "time", name="month_number")
    return cube


def load_lsat(conf):
    """Load LSAT inputs"""
    ncfile = conf["lsat_in"]
    cube = iris.load_cube(ncfile, "land 2 metre temperature")
    icc.add_month_number(cube, "time", name="month_number")
    return cube


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
    #
    init_logging(level="WARN")
    #
    logging.info("Start")
    #
    logging.info(psutil.Process().cpu_affinity())
    nCPUs = len(psutil.Process().cpu_affinity())
    logging.info("len(cpu_affinity) = ", nCPUs)

    conf, ellipse, dat_type, month_value = parse_args(parser)
    #
    v = ellipse.v
    nparms = ellipse.supercategory_n_params
    defval = [-999.9 for _ in range(nparms)]

    everyother = 1

    if dat_type == "lsat":
        data_loader = load_lsat
        outpath_base = conf["lsat_out_base"]
        print("dat_type is lsat")
    else:
        data_loader = load_sst
        outpath_base = conf["sst_out_base"]
        print("dat_type is sst")
    #
    logging.info(f"{v = }")
    #
    additional_constraints = iris.Constraint(month_number=month_value)
    surftemp_cube = data_loader(conf)
    surftemp_cube = surftemp_cube.extract(additional_constraints)
    surftemp_cube = mask_time_union(surftemp_cube)
    #
    surftemp_cube_time_length = len(surftemp_cube.coord("time").points)
    logging.info(repr(surftemp_cube))
    #
    # Init values set to HadCRUT5 defaults
    # no prior distrubtion set around those value
    init_values = [2000.0, 2000.0, 0]
    # Uniformative prior of parameter range
    fit_bounds = [
        (300.0, 30000.0),
        (300.0, 30000.0),
        (-2.0 * np.pi, 2.0 * np.pi),
    ]
    #
    super_cube_list = iris.cube.CubeList()
    #
    logging.info(repr(surftemp_cube))
    logging.debug(f"{surftemp_cube.coord('latitude') = }")
    logging.debug(f"{surftemp_cube.coord('longitude') = }")
    logging.debug(f"{surftemp_cube.coord('time') = }")
    print("Large cube built for cov caculations:", repr(surftemp_cube))
    logging.info("Building covariance matrix")
    super_sst_cov = cube_covariance.CovarianceCube(surftemp_cube)
    logging.info("Covariance matrix completed")
    #
    sst_cube_not_template = surftemp_cube[surftemp_cube_time_length // 2]
    for zonal, zonal_slice in enumerate(
        sst_cube_not_template.slices(["longitude"])
    ):
        # Zonal slices
        logging.info(f"{zonal} {repr(zonal_slice)}")
        if (zonal % everyother) != 0:
            continue
        zonal_cube_list = iris.cube.CubeList()
        for box_count, invidiual_box in enumerate(zonal_slice.slices([])):
            #
            if (box_count % everyother) != 0:
                continue
            logging.info(f"{zonal} || {box_count} {repr(invidiual_box)}")
            #
            current_lon = invidiual_box.coord("longitude").points[0]
            current_lat = invidiual_box.coord("latitude").points[0]
            logging.info(f"{zonal} || {box_count} {current_lon} {current_lat}")
            if np.ma.is_masked(invidiual_box.data):
                xy, actual_latlon = (
                    super_sst_cov.find_nearest_xy_index_in_cov_matrix(
                        [current_lon, current_lat], use_full=True
                    )
                )
                logging.debug(super_sst_cov.data_cube)
                logging.debug(super_sst_cov.data_cube.coord("latitude"))
                logging.debug(super_sst_cov.data_cube.coord("longitude"))
                template_cube = super_sst_cov._make_template_cube2(
                    (current_lon, current_lat)
                )
                ans = cube_covariance.create_output_cubes(
                    template_cube,
                    model_type=ellipse.model_type,
                    additional_meta_aux_coords=[
                        cube_covariance.make_v_aux_coord(v)
                    ],
                    default_values=defval,
                )["param_cubelist"]
                ansH = "MASKED"
            else:
                # Nearest valid point
                xy, actual_latlon = (
                    super_sst_cov.find_nearest_xy_index_in_cov_matrix(
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
                ansX = super_sst_cov.fit_ellipse_model(
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
    #
    vstring = str(v).replace(".", "p")
    vstring = "_v_eq_" + vstring
    #
    outpath = outpath_base + "matern_physical_distances" + vstring + "/"
    Path(outpath).mkdir(parents=True, exist_ok=True)
    #
    outncfilename = outpath + dat_type + "_"
    outncfilename += f"{month_value:02d}.nc"
    logging.debug("Results to be saved...")
    logging.debug(super_cube_list)
    logging.info(f"Saving results to {outncfilename}")
    inc.save(super_cube_list, outncfilename)
    logging.info("Completed")


if __name__ == "__main__":
    main()
