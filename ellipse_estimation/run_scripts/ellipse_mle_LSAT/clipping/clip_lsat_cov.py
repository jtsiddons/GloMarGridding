import iris
from iris.fileformats import netcdf as inc
import numpy as np
from scipy import linalg as linalg_scipy

nc_path = '/noc/mpoc/surface/ERA5_SURFTEMP_500deg_monthly/ANOMALY/SpatialScales/locally_build_covariances/'
nc_infiles = [nc_path+'covariance_'+str(mm+1).zfill(2)+'_v_eq_1p5_lsat_without_psd_check.nc' for mm in range(12)]
nc_outfiles = [nc_path+'covariance_'+str(mm+1).zfill(2)+'_v_eq_1p5_lsat_clipped.nc' for mm in range(12)]

# 7.55 Bun et al 2017
# noise_lambda_threshold = (1+np.sqrt(2592/45))**2
explained_var_target = 0.95

for nc_infile, nc_outfile in zip(nc_infiles, nc_outfiles):
    ns_cov_cubes = iris.load(nc_infile)
    ns_cov = ns_cov_cubes.extract('covariance')[0]
    ns_cor = ns_cov_cubes.extract('correlation')[0]
    ns_cov_arr = np.array(ns_cov.data)
    print(nc_infile, nc_outfile)
    print(repr(ns_cov))
    det = np.linalg.det(ns_cov_arr)
    print(det)
    #
    w, v = linalg_scipy.eigh(ns_cov_arr)
    print(w[:10])
    print(w[10:])
    trace = np.trace(ns_cov_arr)
    mmm = -1
    while True:
        if np.sum(w[mmm:])/trace > explained_var_target:
            break
        mmm -= 1
    print(mmm)
    print(np.average(w[:mmm]))
    w_clipped = np.zeros_like(w)
    w_clipped[:mmm] = np.average(w[:mmm])
    w_clipped[mmm:] = w[mmm:]
    ns_cov_arr_clipped = v @ np.diag(w_clipped) @ v.T
    print(ns_cov_arr_clipped)
    #
    det = np.linalg.det(ns_cov_arr)
    print(det)
    eigvals = linalg_scipy.eigh(ns_cov_arr_clipped,
                                eigvals_only=True,
                                subset_by_index=[0, 10])
    print(eigvals)
    #
    sdevs = np.sqrt(np.diag(ns_cov_arr_clipped))
    sdevs_inv = np.reciprocal(sdevs)
    #
    out_cubes = iris.cube.CubeList()
    #
    ns_cov2 = ns_cov.copy()
    ns_cov2.data = ns_cov_arr_clipped
    out_cubes.append(ns_cov2)
    #
    ns_cor2 = ns_cor.copy()
    ns_cor_arr_clipped = np.diag(sdevs_inv) @ ns_cov_arr_clipped @ np.diag(sdevs_inv)
    np.fill_diagonal(ns_cor_arr_clipped, 1.0)
    ns_cor2.data = ns_cor_arr_clipped
    out_cubes.append(ns_cor2)
    #
    print(out_cubes)
    inc.save(out_cubes, nc_outfile)
