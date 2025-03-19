"""
Apply eigenvalue clipping to ERA5 LSAT nonstationary covariances
"""

import iris
from iris.fileformats import netcdf as inc
import numpy as np
from scipy import linalg as linalg_scipy
import yaml

from ellipse_estimation import repair_damaged_covariance as rdc


def main():
    """
    Check and correct for positive definitness for 12 months of covariances
    """
    #
    with open("clip_sst_cov.yaml", "r") as f:
        clip_intel = yaml.safe_load(f)
    nc_path = clip_intel["nc_path"]
    nc_infiles = [
        nc_path + clip_intel["uncorrected"].replace("MM", str(mm + 1).zfill(2))
        for mm in range(12)
    ]
    nc_outfiles = [
        nc_path + clip_intel["output"].replace("MM", str(mm + 1).zfill(2))
        for mm in range(12)
    ]
    #
    explained_var_target = 0.95
    #
    for nc_infile, nc_outfile in zip(nc_infiles, nc_outfiles):
        ns_cov_cubes = iris.load(nc_infile)
        ns_cov = ns_cov_cubes.extract("covariance")[0]
        ns_cor = ns_cov_cubes.extract("correlation")[0]
        ns_cov_arr = np.array(ns_cov.data)
        print(nc_infile, nc_outfile)
        print(repr(ns_cov))
        det = np.linalg.det(ns_cov_arr)
        print(det)
        #
        cov_fixer = rdc.Laloux_CovarianceClean(ns_cov_arr.data)
        ns_cov_arr_clipped = cov_fixer.eig_clip_via_cov(
            method="explained_variance",
            method_parms={"target": explained_var_target},
        )
        det_clipped = np.linalg.det(ns_cov_arr_clipped)
        print(det_clipped)
        eigvals_clipped = linalg_scipy.eigh(
            ns_cov_arr_clipped, eigvals_only=True, subset_by_index=[0, 10]
        )
        print(eigvals_clipped)
        #
        out_cubes = iris.cube.CubeList()
        ns_cov2 = ns_cov.copy()
        ns_cov2.data = ns_cov_arr_clipped
        out_cubes.append(ns_cov2)
        ns_cor2 = ns_cor.copy()
        ns_cor_arr_clipped = cov_fixer._cov2cor(ns_cov_arr_clipped)
        ns_cor2.data = ns_cor_arr_clipped
        out_cubes.append(ns_cor2)
        #
        print(out_cubes)
        inc.save(out_cubes, nc_outfile)


if __name__ == "__main__":
    main()
