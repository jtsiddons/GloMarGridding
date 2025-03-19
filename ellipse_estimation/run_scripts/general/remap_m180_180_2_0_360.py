"""
Remaps -180 180 covariances to 0-360 (for some datasets)
This is for a 36x72 grid
"""

import pandas as pd
import numpy as np
import iris
from iris.fileformats import netcdf as inc
# from scipy.linalg import cholesky

ncpaths = [
    "/noc/mpoc/surface_data/ESA_CCI5deg_month_extra/ANOMALY/SpatialScales/locally_build_covariances/",
    "/noc/mpoc/surface/ERA5_SURFTEMP_500deg_monthly/ANOMALY/SpatialScales/locally_build_covariances/",
]
obsnames = ["sst", "lsat"]

for ncpath, obsname in zip(ncpaths, obsnames):
    mms = [m + 1 for m in range(12)]
    for mm in mms:
        ncfile = (
            ncpath
            + "covariance_"
            + str(mm).zfill(2)
            + "_v_eq_1p5_"
            + obsname
            + "_clipped.nc"
        )
        csvfile = (
            ncpath
            + "row_lookup_"
            + str(mm).zfill(2)
            + "_v_eq_1p5_"
            + obsname
            + "_without_psd_check.csv"
        )
        outfile = (
            ncpath
            + "covariance_"
            + str(mm).zfill(2)
            + "_v_eq_1p5_"
            + obsname
            + "_clipped_0_360.nc"
        )
        # Lfile = ncpath+'covariance_'+str(mm).zfill(2)+'_v_eq_1p5_'+obsname+'_clipped_L_0_360.npy'
        latlon_lookup = pd.read_csv(csvfile)
        latlon_lookup["lon_alt"] = latlon_lookup["lon"].apply(
            lambda lon: lon + 360 if lon < 0 else lon
        )
        latlon_lookup_alt = latlon_lookup.sort_values(by=["lat", "lon_alt"])
        latlon_lookup_alt["new_row_num"] = np.arange(2592)
        print(latlon_lookup)
        print(latlon_lookup_alt)
        H = np.zeros([2592, 2592])
        for new_row, old_row in zip(
            latlon_lookup_alt["new_row_num"], latlon_lookup_alt["row_num"]
        ):
            H[new_row, old_row] = 1
        print(H)
        assert np.all(np.sum(H, axis=0) == 1), (
            "Your mapping matrix is not working! Only a single 1 per axis=0"
        )
        assert np.all(np.sum(H, axis=1) == 1), (
            "Your mapping matrix is not working! Only a single 1 per axis=1"
        )
        cubes = iris.load(ncfile)
        new_cubes = iris.cube.CubeList()
        for varname in ["covariance", "correlation"]:
            cube = cubes.extract(varname)[0]
            print(repr(cube))
            new_cube = cube.copy()
            og_type = cube.data.dtype
            HCHT = H @ cube.data @ H.T
            HCHT = HCHT.astype(og_type)
            new_cube.data = HCHT
            new_cubes.append(new_cube)
        # new_cov = new_cubes.extract('covariance')[0]
        # L = cholesky(new_cov.data, lower=True)
        print(mm, ": Saving to ", outfile)
        inc.save(new_cubes, outfile)
        # print(mm, 'Saving to ', Lfile)
        # np.save(Lfile, L)
