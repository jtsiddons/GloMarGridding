"""
Stiches non stationary variogram parameters to form a covariance matrix
"""

from os import path as os_path
from pathlib import Path
import socket
import sys

# Imported modules
import iris
from iris.coords import DimCoord, AuxCoord
from iris.cube import Cube
from iris.fileformats import netcdf as inc
import pandas as pd
import numpy as np
import yaml

# My own modules
from ellipse_estimation import (
    cube_covariance_nonstationary_stich as cube_cov_stich,
)


def masklookuptable(cube, csv_unit):
    """Writes the row column index that matches a specific grid box"""
    longrid, latgrid = np.meshgrid(
        cube.coord("longitude").points, cube.coord("latitude").points
    )
    latgrid = np.ma.masked_where(cube.data.mask, latgrid)
    longrid = np.ma.masked_where(cube.data.mask, longrid)
    lats_unmasked = np.ma.compressed(latgrid)
    lons_unmasked = np.ma.compressed(longrid)
    latlons_unmasked = np.column_stack((lats_unmasked, lons_unmasked))
    row_lookup = {
        "row_num": np.arange(latlons_unmasked.shape[0]),
        "lat": latlons_unmasked[:, 0],
        "lon": latlons_unmasked[:, 1],
    }
    row_lookup = pd.DataFrame(row_lookup)
    row_lookup.to_csv(csv_unit)


def set_sklearn_haversine():
    """Determine if sklearn haversine should be used or not"""
    host_name = socket.gethostname()
    # True - NOC; False - JASMIN
    if ".noc." in host_name:
        use_sklearn_haversine = True
    elif ".jasmin." in host_name:
        use_sklearn_haversine = False
    else:
        use_sklearn_haversine = False
    return use_sklearn_haversine


def main():
    """MAIN"""
    delta_x_method = "Modified_Met_Office"
    degree_dist = False
    max_dist = 1.5e8
    use_sklearn_haversine = set_sklearn_haversine()

    # 1    2     3             4            5            6
    # ${v} ${mm} ${everyother} ${highamadj} ${usejoblib} ${njobs}
    argv = sys.argv
    mm = int(
        argv[1]
    )  # month in months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    v = float(argv[2])  # v in vs = [0.5, 1.5, 3] etc.

    # fix_indefinite = True 1 or False 0
    if argv[3] == "0":
        fix_indefinite = False
    elif argv[3] == "1":
        fix_indefinite = True
    else:
        raise ValueError(
            "Unknown argv4 fix_indefinite: " + argv[3] + "; 0 or 1 only"
        )

    verbose = False

    # joblib multiprocessing
    if argv[4] == "0":
        use_joblib = False
        n_jobs = 1
    elif argv[4] == "1":
        use_joblib = True
        n_jobs = int(argv[6])
    else:
        raise ValueError(
            "Unknown argv5 (use_joblib): " + argv[4] + "; 0 or 1 only"
        )

    print("Execution option: ")
    print("use_joblib = ", use_joblib)
    print("n_jobs = ", n_jobs)
    print("fix_indefinite = ", fix_indefinite)

    with open("API_mask_esa_cov_5p0_sst.yaml", "r") as f:
        stich_intel = yaml.safe_load(f)
    base_path = stich_intel["base_path"]
    data_path = base_path + stich_intel["data_path"]
    out_suffix = "sst"
    if not fix_indefinite:
        out_suffix += "_without_psd_check"
    outpath = base_path + stich_intel["outpath"]
    print("Outputs will be saved to ", outpath + "/")
    Path(outpath).mkdir(parents=True, exist_ok=True)

    print(mm)
    mm_str = str(mm).zfill(2)
    print("Generating covariance matrix")

    # Read the local fitted sigma parameters
    vstring = str(v).replace(".", "p")
    vstring = "_v_eq_" + vstring
    fitpath_dist = data_path
    fitpath_file = fitpath_dist + "sst_" + str(mm).zfill(2) + ".nc"

    # Check file exists; if yes, read else stops.
    assert os_path.isfile(fitpath_file), fitpath_file + " is not found."
    cube_matern_dist = iris.load(fitpath_file)
    print(cube_matern_dist)
    Lx = cube_matern_dist.extract("Lx")[0]
    Ly = cube_matern_dist.extract("Ly")[0]
    theta = cube_matern_dist.extract("theta")[0]
    sigma = cube_matern_dist.extract("standard_deviation")[0]
    print("Lx info:")
    print(Lx)
    print("Lx shape:")
    print(Lx.coord("latitude").points)
    print(Lx.coord("longitude").points)
    print("Lx number of valid grid points:")
    print(Lx.data.compressed().shape)

    covariance_file = outpath + "/covariance_"
    covariance_file += mm_str + vstring + "_" + out_suffix + ".nc"

    if os_path.isfile(covariance_file):
        print(covariance_file, "exists; bypassing.")
        return
    print("Creating ", covariance_file)

    # stich is an instance to CovarianceCube_PreStichedLocalEstimates
    # See cube_covariance_nonstationary_stich.py in nonstationary_cov module
    stich = cube_cov_stich.CovarianceCube_PreStichedLocalEstimates(
        Lx,
        Ly,
        theta,
        sigma,
        v=v,
        delta_x_method=delta_x_method,
        degree_dist=degree_dist,
        max_dist=max_dist,
        check_positive_definite=fix_indefinite,
        use_sklearn_haversine=use_sklearn_haversine,
        use_joblib=use_joblib,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    # Extracts covariance and correlation matrix
    # from CovarianceCube_PreStichedLocalEstimates instance
    stich_cov = stich.cov_ns
    stich_cor = stich.cor_ns

    # Create dummy row and column coordinates for covariance and correlation matrix
    nrows = stich.cov_ns.shape[0]
    dim_row = DimCoord(
        np.arange(nrows, dtype=int), long_name="dim_0", units="1"
    )
    dim_col = DimCoord(
        np.arange(nrows, dtype=int), long_name="dim_1", units="1"
    )

    # v_coord as an Aux Coord stating the Matern parameter
    v_coord = AuxCoord(v, long_name="matern_nu")
    det_coord = AuxCoord(stich.cov_det, long_name="covariance_determinant")
    eig_coord = AuxCoord(stich.cov_eig[0], long_name="smallest_eigenvalue")
    pd_check_coord = AuxCoord(
        int(stich.check_positive_definite),
        long_name="positive_semidefinite_check_enabled",
    )

    # Define the iris cube
    cov_cube = Cube(stich_cov, dim_coords_and_dims=[(dim_row, 0), (dim_col, 1)])
    cor_cube = Cube(stich_cor, dim_coords_and_dims=[(dim_row, 0), (dim_col, 1)])
    cov_cube.data = cov_cube.data.astype(np.float32)
    cor_cube.data = cor_cube.data.astype(np.float32)

    # Add metadata
    for aux_coord in [v_coord, det_coord, eig_coord, pd_check_coord]:
        cov_cube.add_aux_coord(aux_coord)
        cor_cube.add_aux_coord(aux_coord)
    cov_cube.units = "K**2"
    cor_cube.units = "1"
    cov_cube.rename("covariance")
    cor_cube.rename("correlation")

    # Write to file
    cov_list = iris.cube.CubeList()
    cov_list.append(cov_cube)
    cov_list.append(cor_cube)
    print("Writing covariance file: ", covariance_file)
    inc.save(cov_list, covariance_file)

    csv_file = outpath + "/row_lookup_"
    csv_file += mm_str + vstring + "_" + out_suffix + ".csv"
    print("Writing row-column lookup table: ", csv_file)
    masklookuptable(Lx, csv_file)

    print("Complete")


if __name__ == "__main__":
    main()
