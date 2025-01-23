"""
Functions for computing covariance matrices or analysing covariance matrices.
"""

from collections.abc import Callable
import numpy as np

from sklearn.decomposition import PCA


import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import xarray as xr

from .variogram import (
    ExponentialVariogram,
    calculate_distance_matrix,
    haversine_distance,
    variogram_to_covariance,
)


# QUESTION: Can I delete this triple quote section?

###############################################################
# DATA READ-IN
###############################################################
# def _preprocess(x, lon_bnds, lat_bnds):
#     return x.sel(lon=slice(*lon_bnds), lat=slice(*lat_bnds))
#
# lon_west  = -98
# lon_east  = 20
# lat_south = 0
# lat_north = 68
# # extract the latitude and longitude boundaries from user input
# lon_bnds, lat_bnds = (lon_west, lon_east), (lat_south, lat_north)
# print(lon_bnds, lat_bnds)
#
# adding a pre-processing function to subselect only that region when extracting
# the data from the path
# partial_func = partial(_preprocess, lon_bnds=lon_bnds, lat_bnds=lat_bnds)


# def read_in_data():
#     # load data
#     # path = sys.argv[1]
#     path = "/noc/mpoc/surface_data/ESA_CCI1deg_pent/ANOMALY/*.nc"
#     ds = xr.open_mfdataset(
#         str(path), combine="nested", concat_dim="time", engine="netcdf4"
#     )  # preprocess=partial_func, engine="netcdf4")
#     print(ds)
#
#     sst = ds.variables["sst_anomaly"]
#     lat = ds.variables["lat"][:].values
#     lon = ds.variables["lon"][:].values
#     ds.close()
#
#     gridlon, gridlat = np.meshgrid(lon, lat)
#     print(gridlat)
#     print(gridlon)
#     lat_1d = gridlat.flatten(
#         order="C"
#     )  # same order of reshaping for lat,lon and data
#     lon_1d = gridlon.flatten(
#         order="C"
#     )  # same order of reshaping for lat, lon and data
#
#     nt, nlat, nlon = sst.shape
#     print(nt, nlat, nlon)
#
#     sst_flat_with_nans = sst.to_numpy().reshape(
#         (nt, nlat * nlon), order="C"
#     )  # same rder of reshaping for lat,lon and data
#     return lat_1d, lon_1d, sst_flat_with_nans


#################################################################
# DATA PREPROCESSING IN TERMS OF LAND/OCEAN MASK
#################################################################
# DELETE: Mask - see mask.mask_from_obs
def mask_land_and_ice(
    sst_flat_with_nans: np.ndarray,
    lat_1d: np.ndarray,
    lon_1d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # in main three_step_kriging code pentad_1d_time comes already transposed so
    # first line is not necessary
    # however, if you're running it only from this file and not use three_step
    # kriging (so you use read_in_data function, the sst_flat_with_nans needs to
    # be transposed)
    # it only works when you have (space,time) order of dimensions
    cci_mask = np.copy(sst_flat_with_nans).T  # .T)

    # # cci_mask = np.copy(sst_flat_with_nans)
    # for line in cci_mask:
    #     mask = np.isnan(line).any()
    #     line[mask] = np.nan

    # Mask the land points
    cci_mask = np.ma.masked_array(cci_mask, np.isnan(cci_mask))
    land = cci_mask.sum(axis=0).mask
    ocean_points = ~land
    print(cci_mask.shape)
    # keep only oceanic grid-points
    cci_mask = cci_mask[:, ocean_points]
    # keep only oceanic lats and lons
    # ocean_lat = np.ma.masked_where(~ocean,lat_1d) #lat_1d[ocean]
    # ocean_lon = np.ma.masked_where(~ocean, lon_1d) #lon_1d[ocean]
    # ocean_lat = np.reshape(ocean_lat, lat.shape, order='C')
    # ocean_lon = np.reshape(ocean_lon, lon.shape, order='C')
    ocean_lat = lat_1d[ocean_points]
    ocean_lon = lon_1d[ocean_points]
    return cci_mask, ocean_points, ocean_lat, ocean_lon


# in order to get same result as in Matlab (or three_step_kriging.py you need to
# transpose matrix for covariance calculation)
# plt.imshow(np.flipud(np.cov(cci_mask.T)))
# plt.colorbar()
# plt.clim(-0.7,1)
# plt.show()


######################################################
# COVARIANCE ON ORIGINAL EMPIRICAL DATASET
######################################################
def calculate_empirical_covariance(
    sst_flat_with_nans: np.ndarray,
    lat_1d: np.ndarray,
    lon_1d: np.ndarray,
) -> np.ndarray:
    cci_mask, _, _, _ = mask_land_and_ice(sst_flat_with_nans, lat_1d, lon_1d)
    print("cci mask shape", cci_mask.shape)
    covariance_empirical = np.cov(cci_mask.T)
    return covariance_empirical


# quick variance and std check - would be good to have them 0 and 1
# can be done using:
# from sklearn import preprocessing
# scaler  = preprocessing.StandardScaler()
# scaler_sst = scaler.fit(X)
# X = scaler_sst.transform(X)
# print(np.mean(cci_mask))
# print(np.std(cci_mask))


#####################################################
# EOF AND PCA ON ORIGINAL EMPIRICAL DATASET
#####################################################
def calculate_pca_and_eof(
    cci_mask: np.ndarray,
    chosen_ipc: int | None = None,
) -> tuple[int, np.ndarray, np.ndarray]:
    # instantiates the PCA object
    skpca = PCA()
    # fit
    skpca.fit(cci_mask)

    style.use("fivethirtyeight")
    _, ax = plt.subplots(figsize=(6, 6))
    ax.plot(skpca.explained_variance_ratio_[0:10] * 100)
    ax.plot(skpca.explained_variance_ratio_[0:10] * 100, "ro")
    plt.show()

    ipc = (
        chosen_ipc
        if chosen_ipc is not None
        else np.where(skpca.explained_variance_ratio_.cumsum() >= 0.70)[0][0]
    )
    print("ipc", ipc)

    # The Principal Components (PCs) are obtained by using the transform method
    # of the pca object (skpca)
    PCs = skpca.transform(cci_mask)
    PCs = PCs[:, :ipc]
    print("PCs", PCs.shape)

    # The Empirical Orthogonal Functions (EOFs) are contained in the components_
    # attribute of the pca object (skpca)
    EOFs = skpca.components_
    EOFs = EOFs[:ipc, :]
    print("EOFs", EOFs.shape)
    return ipc, PCs, EOFs


def reconstruct_covariance_from_eofs(
    sst_flat_with_nans: np.ndarray,
    lat_1d: np.ndarray,
    lon_1d: np.ndarray,
    chosen_ipc: int | None = None,
) -> np.ndarray:
    cci_mask, ocean_points, ocean_lat, ocean_lon = mask_land_and_ice(
        sst_flat_with_nans, lat_1d, lon_1d
    )

    ipc, PCs, EOFs = calculate_pca_and_eof(cci_mask, chosen_ipc=chosen_ipc)

    # QUESTION: should lat/lons here be inputs, or output of mask_land_and_ice?
    EOF_recons = np.ones((ipc, len(ocean_lat) * len(ocean_lon))) * -999.0
    for i in range(ipc):
        EOF_recons[i, ocean_points] = EOFs[i, :]
    EOF_recons = np.ma.masked_values(
        np.reshape(
            EOF_recons, (ipc, len(ocean_lat), len(ocean_lon)), order="C"
        ),
        -999.0,
    )
    print("Reconstructed", EOF_recons.shape)

    plt.imshow(
        EOF_recons[2, :, :],
        origin="lower",
        interpolation="nearest",
        aspect="auto",
    )
    plt.title("3rd EOF for the ESA CCI SST anomalies")
    plt.colorbar()
    plt.show()

    dat_from_eofs = PCs @ EOFs
    covariance_from_eof = np.cov(dat_from_eofs.T)
    return covariance_from_eof


"""
cci_unmasked = np.ma.getdata(cci_mask, subok=True)
print('cci unmasked', cci_unmasked.shape)
print('EOFs T', (EOFs.T).shape)
tvecs = cci_unmasked @ EOFs.T
print('tvecs', tvecs.shape)
vecs = EOFs
print('vecs', vecs.shape)
dat_from_eofs = tvecs @ vecs
dat_from_eofs2 = PCs @ EOFs
diff = dat_from_eofs2 - dat_from_eofs
print(diff)
"""


################################################################
# PLOTTING FUNCTION FOR COMPARISON FOR EOF RECONSTRUCTION
################################################################
def plot_empirical_reconstructed_covariance(
    cci_mask: np.ndarray,
    dat_from_eofs: np.ndarray,
) -> None:
    orig_cov = np.cov(cci_mask.T)
    rec_cov = np.cov(dat_from_eofs.T)
    diff_cov = orig_cov - rec_cov
    cov_data = [orig_cov, rec_cov, diff_cov]
    fig, axes = plt.subplots(nrows=1, ncols=len(cov_data))
    fig.suptitle(
        "Original CCI SST anomaly covariance (1), covariance using "
        + "reconstructed data from 3 EOFs (2), and difference (1-2)"
    )
    cbar_ax = fig.add_axes((0.85, 0.15, 0.05, 0.7))
    for i, ax in enumerate(axes.flat):
        im = ax.imshow(np.flipud(cov_data[i]), vmin=-0.7, vmax=1.0)
        if i == len(axes.flat) - 1:
            fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, cax=cbar_ax)
    fig.subplots_adjust(right=0.8)
    fig.show()
    return None


# data_xr = xr.DataArray(orig_cov)
# data_xr.to_netcdf('original_covariance_noice.nc')


##################################################################
# CREATING A COVARIANCE FROM VARIOGRAM
##################################################################
# sst_flat_with_nans, lat_1d, lon_1d = read_in_data()
def covariance_from_variogram(
    sst_flat_with_nans: np.ndarray,
    lat_1d: np.ndarray,
    lon_1d: np.ndarray,
    variance: np.ndarray,
    dist_func: Callable = haversine_distance,
) -> np.ndarray:
    _, _, ocean_lat, ocean_lon = mask_land_and_ice(
        sst_flat_with_nans, lat_1d, lon_1d
    )

    # compute pairwise distances
    # or distance matrix using pandas
    df = pd.DataFrame({"lat": ocean_lat, "lon": ocean_lon})

    distance_matrix: np.ndarray = calculate_distance_matrix(
        df, dist_func=dist_func
    )
    print("squareform(pdist) \n", distance_matrix)

    # call variogram function
    # this calculates the covariance on global grid (equivalent in Simon's code
    # is "Ss" martix)
    # "S" matrix in Simon's code is calculated using same m,d parameters but for
    # the observation points distance matrix (not all ocean points)
    # covariance_variogram: np.ndarray = variogram(distance_matrix, variance)
    covariance_variogram = ExponentialVariogram(
        psill=variance,
        nugget=0.0,
        range=350.0,
    ).fit(distance_matrix)
    return variogram_to_covariance(covariance_variogram, variance)
