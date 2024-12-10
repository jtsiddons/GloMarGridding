################
# by A. Faulkner
# for python version 3.0 and up
################

from typing import Literal
import numpy as np

from utils import adjust_small_negative

KrigMethod = Literal["simple", "ordinary"]


def intersect_mtlb(a, b):
    """
    Returns data common between two arrays, a and b, in a sorted order and index vectors for a and b arrays
    Reproduces behaviour of Matlab's intersect function

    Parameters
    ----------
    a (array) - 1-D array
    b (array) - 1-D array

    Returns
    -------
    1-D array, c, of common values found in two arrays, a and b, sorted in order
    List of indices, where the common values are located, for array a
    List of indices, where the common values are located, for array b
    """
    a1, ia = np.unique(a, return_index=True)
    b1, ib = np.unique(b, return_index=True)
    aux = np.concatenate((a1, b1))
    aux.sort()
    c = aux[:-1][aux[1:] == aux[:-1]]
    return c, ia[np.isin(a1, c)], ib[np.isin(b1, c)]


def kriging(
    obs_idx: np.ndarray,
    weights: np.ndarray,
    obs: np.ndarray,
    interp_cov: np.ndarray,
    error_cov: np.ndarray,
    remove_obs_mean: int = 0,
    obs_bias: np.ndarray | None = None,
    method: KrigMethod = "simple",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform Kriging using a chosen method.

    Get array of krigged observations and anomalies for all grid points in the
    domain.

    Parameters
    ----------
    obs_idx : np.ndarray[int]
        Grid indices with observations.
    weights : np.ndarray[float]
        Weight matrix (inverse of counts of observations).
    obs : np.ndarray[float]
        All point observations/measurements for the chosen date.
    interp_cov : np.ndarray[float]
        interpolation covariance of all output grid points (each point in time and all points
        against each other).
    error_cov : np.ndarray[float]
        Measurement/Error covariance matrix.
    remove_obs_mean: int
        Should the mean or median from grib_obs be removed and added back onto grib_obs?
        0 = No (default action)
        1 = the mean is removed
        2 = the median is removed
        3 = the spatial meam os removed
    obs_bias : np.ndarray[float] | None
        Bias of all measurement points for a chosen date (corresponds to x_obs).
    method : KrigMethod
        The kriging method to use to fill in the output grid. One of "simple" or "ordinary".

    Returns
    -------
    z_obs : np.ndarray[float]
        Full set of values for the whole domain derived from the observation
        points using the chosen kriging method.
    dz : np.ndarray[float]
        Uncertainty associated with the chosen kriging method.
    """

    if obs_bias is not None:
        print("With bias")
        grid_obs = weights @ (obs - obs_bias)
    else:
        grid_obs = weights @ obs

    if len(grid_obs) > 1:
        grid_obs = np.squeeze(grid_obs)

    assert remove_obs_mean in [0, 1, 2, 3], "Unknown remove_obs_mean value"
    grid_obs_av = None
    if remove_obs_mean == 1:
        grid_obs_av = np.ma.average(grid_obs)
        grid_obs = grid_obs - grid_obs_av
    elif remove_obs_mean == 2:
        grid_obs_av = np.ma.median(grid_obs)
        grid_obs = grid_obs - grid_obs_av
    elif remove_obs_mean == 3:
        grid_obs_av = get_spatial_mean(grid_obs, error_cov)
        grid_obs = grid_obs - grid_obs_av

    print(f"{grid_obs.shape = }")

    if error_cov.shape == interp_cov.shape:
        print(
            "Error covariance supplied is of the same size as interpolation covariance, subsetting to indices of observation grids"
        )
        error_cov = error_cov[obs_idx[:, None], obs_idx[None, :]]

    print(f"{error_cov =}, {error_cov.shape =}")

    # S is the spatial covariance between all "measured" grid points
    # Plus the covariance due to the measurements, i.e. measurement noise, bias
    # noise, and sampling noise (R)
    S = np.asarray(interp_cov[obs_idx[:, None], obs_idx[None, :]])
    S += weights @ error_cov @ weights.T
    print(f"{S =}, {S.shape =}")
    # Ss is the covariance between to be "predicted" grid points (i.e. all) and
    # "measured" points
    Ss = np.asarray(interp_cov[obs_idx, :])
    print(f"{Ss =}, {Ss.shape =}")

    if method.lower() == "simple":
        print("Performing Simple Kriging")
        z_obs, dz = kriging_simple(S, Ss, grid_obs, interp_cov)
    elif method.lower() == "ordinary":
        print("Performing Ordinary Kriging")
        z_obs, dz = kriging_ordinary(S, Ss, grid_obs, interp_cov)
    else:
        raise NotImplementedError(
            f"Kriging method {method} is not implemented. "
            'Expected one of "simple" or "ordinary"'
        )

    if grid_obs_av is not None:
        z_obs = z_obs + grid_obs_av

    return z_obs, dz


def unmasked_kriging(
    unmask_idx: np.ndarray,
    unique_obs_idx: np.ndarray,
    weights: np.ndarray,
    obs: np.ndarray,
    interp_cov: np.ndarray,
    error_cov: np.ndarray,
    remove_obs_mean: int = 0,
    obs_bias: np.ndarray | None = None,
    method: KrigMethod = "simple",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform Kriging on a masked grid using a chosen method.

    Get array of krigged observations and anomalies for all grid points in the
    domain.

    Parameters
    ----------
    unmask_idx : np.ndarray[int]
        Indices of all un-masked points for chosen date.
    unique_obs_idx : np.ndarray[int]
        Unique indices of all measurement points for a chosen date, representative of the indices of gridboxes, which have => 1 measurement.
    weights : np.ndarray[float]
        Weight matrix (inverse of counts of observations).
    obs : np.ndarray[float]
        All point observations/measurements for the chosen date.
    interp_cov : np.ndarray[float]
        Interpolation covariance of all output grid points (each point in time and all points
        against each other).
    error_cov : np.ndarray[float]
        Measurement/Error covariance matrix.
    remove_obs_mean: int
        Should the mean or median from grib_obs be removed and added back onto grib_obs?
        0 = No (default action)
        1 = the mean is removed
        2 = the median is removed
        3 = the spatial meam os removed
    obs_bias : np.ndarray[float] | None
        Bias of all measurement points for a chosen date (corresponds to x_obs).
    method : KrigMethod
        The kriging method to use to fill in the output grid. One of "simple" or "ordinary".

    Returns
    -------
    z_obs : np.ndarray[float]
        Full set of values for the whole domain derived from the observation
        points using the chosen kriging method.
    dz : np.ndarray[float]
        Uncertainty associated with the chosen kriging method.
    """
    obs_idx = get_unmasked_obs_indices(unmask_idx, unique_obs_idx)

    return kriging(
        obs_idx,
        weights,
        obs,
        interp_cov,
        error_cov,
        remove_obs_mean,
        obs_bias,
        method,
    )


def get_unmasked_obs_indices(
    unmask_idx: np.ndarray,
    unique_obs_idx: np.ndarray,
) -> np.ndarray:
    """
    Get grid indices with observations from un-masked grid-box indices and unique
    grid-box indices with observations.

    Parameters
    ----------
    unmask_idx : np.ndarray[int]
        List of all unmasked grid-box indices.
    unique_obs_idx : np.ndarray[int]
        Indices of grid-boxes with observations.

    Returns
    -------
    obs_idx : np.ndarray[int]
        Subset of grid-box indices containing observations that are unmasked.
    """
    unmask_idx = np.squeeze(unmask_idx) if unmask_idx.ndim > 1 else unmask_idx
    _, obs_idx, _ = intersect_mtlb(unmask_idx, unique_obs_idx)
    # index of the sorted unique (iid) in the full iid array
    obs_idx = obs_idx.astype(int)

    return obs_idx


def kriging_simple(
    S: np.ndarray,
    Ss: np.ndarray,
    grid_obs: np.ndarray,
    interp_cov: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform Simple Kriging

    Parameters
    ----------
    S : np.ndarray[float]
        Spatial covariance between all measured grid points plus the
        covariance due to measurements (i.e. measurement noise, bias noise, and
        sampling noise).
    Ss : np.ndarray[float]
        Covariance between the all (predicted) grid points and measured points.
    grid_obs : np.ndarray[float]
        Gridded measurements (all measurement points averaged onto the output gridboxes).
    interp_cov : np.ndarray[float]
        interpolation covariance of all output grid points (each point in time and all points
        against each other).

    Returns
    -------
    z_obs : np.ndarray[float]
        Full set of values for the whole domain derived from the observation
        points using simple kriging.
    dz : np.ndarray[float]
        Uncertainty associated with the simple kriging.
    """
    G = np.linalg.solve(S, Ss).T
    print(f"{G.shape = }")
    print(f"{grid_obs.shape =}")
    z_obs = G @ grid_obs
    print(f"{z_obs =}")

    G = G @ Ss
    print(f"{G =}")
    dz_squared = np.diag(interp_cov - G)
    adjust_small_negative(dz_squared)
    dz = np.sqrt(dz_squared)
    print(f"{dz =}")
    dz[np.isnan(dz)] = 0.0

    print("Simple Kriging Complete")
    return z_obs, dz


def kriging_ordinary(
    S: np.ndarray,
    Ss: np.ndarray,
    grid_obs: np.ndarray,
    interp_cov: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform Ordinary Kriging

    Parameters
    ----------
    S : np.ndarray[float]
        Spatial covariance between all measured grid points plus the
        covariance due to measurements (i.e. measurement noise, bias noise, and
        sampling noise).
    Ss : np.ndarray[float]
        Covariance between the all (predicted) grid points and measured points.
    grid_obs : np.ndarray[float]
        Gridded measurements (all measurement points averaged onto the output gridboxes).
    interp_cov : np.ndarray[float]
        Interpolation covariance of all output grid points (each point in time and all points
        against each other).

    Returns
    -------
    z_obs : np.ndarray[float]
        Full set of values for the whole domain derived from the observation
        points using ordinary kriging.
    dz : np.ndarray[float]
        Uncertainty associated with the ordinary kriging.
    """
    # Convert to ordinary kriging, add Lagrangian multiplier
    N, M = Ss.shape
    S = np.block([[S, np.ones((N, 1))], [np.ones((1, N)), 0]])
    Ss = np.concatenate((Ss, np.ones((1, M))), axis=0)
    grid_obs = np.append(grid_obs, 0)

    G = np.linalg.solve(S, Ss).T
    alpha = G[:, -1]
    z_obs = G @ grid_obs

    G = G @ Ss
    dz_squared = np.diag(interp_cov - G) - alpha
    adjust_small_negative(dz_squared)
    dz = np.sqrt(dz_squared)
    # dz[np.isnan(dz)] = 0.0

    print("Ordinary Kriging Complete")
    return z_obs, dz


def result_reshape_2d(result_1d, iid, grid_2d):
    """
    Returns reshaped krigged output array, from 1-D into 2-D reproducing Matlab's functionality (going over all rows first, before moving onto columns)

    Parameters
    ----------
    result_1d (array) - krigged output array in 1-D
    iid (array) - mask array indicating locations of water regions in the domain
    grid_2d (array) - 2-D array of the output domain

    Returns
    -------
    Krigged result over water areas on a 2-D domain grid with masked land areas
    """
    result_1d = result_1d.astype("float")
    grid_2d = grid_2d.astype("float")

    if grid_2d.ndim > 1:
        grid_2d_flat = grid_2d.flatten()

    to_modify = np.zeros(grid_2d_flat.shape)

    if iid.ndim > 1:
        iid = np.squeeze(iid)
    indexes = iid

    landmask = np.copy(grid_2d_flat)
    # print(landmask)

    if ~np.isnan(landmask).any():
        landmask = landmask.astype("float")
        landmask[landmask == 0] = np.nan

    replacements = result_1d
    for index, replacement in zip(indexes, replacements):
        to_modify[index] = replacement
    # to_modify = to_modify * landmask
    to_modify = np.where(np.isnan(landmask), np.nan, to_modify)
    result_2d = np.reshape(to_modify, (grid_2d.shape))
    return result_2d


def watermask(ds_masked):
    try:
        water_mask = np.copy(ds_masked.variables["landmask"][:, :])
    except KeyError:
        # water_mask = np.copy(ds_masked.variables['land_sea_mask'][:,:])
        water_mask = np.copy(ds_masked.variables["landice_sea_mask"][:, :])
    """
    water_mask[~np.isnan(water_mask)] = 1
    water_mask[np.isnan(water_mask)] = 0
    """

    # print(np.shape(water_mask))
    water_idx = np.asarray(
        np.where(water_mask.flatten() == 1)
    )  # this idx is returned as a row-major
    # water_idx = np.asarray(np.where(water_mask.flatten(order='F') == 1)) #this idx is returned as a column-major
    print(f"{water_idx=}")
    return water_mask, water_idx


def kriging_main(
    covariance,
    cond_df,
    ds_masked,
    flattened_idx,
    obs_cov,
    W,
    bias=False,
    kriging_method: KrigMethod = "simple",
):
    try:
        obs = cond_df["sst_anomaly"].values  # cond_df['cci_anomalies'].values
    except KeyError:
        obs = cond_df["obs_anomalies"].values
    """
    print('CHECK BIAS AND SST HAVE SAME LENGHT')
    print(len(obs))
    print(len(bias))
    """
    # print('1 - DONE')
    # water_mask, water_idx = watermask(ds, ds_var, timestep)
    water_mask, water_idx = watermask(ds_masked)
    obs_idx = flattened_idx
    unique_obs_idx = np.unique(obs_idx)
    # print('2 - DONE')
    # _,ia,_ = intersect_mtlb(water_idx,unique_obs_idx)
    # W = np.zeros((int(max(unique_obs_idx.shape)),int(max(obs_idx.shape))))
    # for k in range(max(unique_obs_idx.shape)):
    # q = [i for i, x in enumerate(obs_idx) if x == unique_obs_idx[k]]
    # for i in range(len(q)):
    # qq = q[i]
    # W[k,qq] = np.divide(1, len(q))
    obs_bias = None
    if bias:
        obs_bias = cond_df["hadsst_bias"].values
    z_obs, dz = unmasked_kriging(
        water_idx,
        unique_obs_idx,
        W,
        obs,
        covariance,
        obs_cov,
        obs_bias=obs_bias,
        method=kriging_method,
    )
    # print('3 - DONE')
    return (
        result_reshape_2d(z_obs, water_idx, water_mask),
        result_reshape_2d(dz, water_idx, water_mask),
    )


# def krige_for_esa_values_only(
#     iid, uind, W, x_obs, cci_covariance, bias=False, clim=False
# ):
#     # this basicallt removes the R matrix (which is the obs covariance that has noise, uncertainty etc) as we use actuall ESA CCI SST anomaly values that we take as ground truth and therefore we assume they have no uncertainty or bias
#     """
#     Returns arrays of krigged observations and anomalies for all grid points in the domain
#
#     Parameters
#     ----------
#     iid (list) - ID of all measurement points for the chosen date
#     x_obs (list) - all point observations for the chosen date
#     clim (list) - climatology values for all observation points
#     bias (list) - bias for all observation points
#     covx (array) - measurement covariance matrix
#     cci_covariance (array) - covariance of all CCI grid points (each point in time and all points against each other)
#     df (dataframe) - dataframe containing all information and observations for a chosen date
#
#     Returns
#     -------
#     Full set of values for the whole domain derived from observation points using Simple Kriging
#     Uncertainty associated using Simple Kriging
#     Full set of values for the whole domain derived from observation points using Ordinary Kriging
#     Uncertainty associated using Ordinary Kriging
#     """
#     # So Now we can get to the part where we krige the grid
#     # bias = np.array(bias)
#     # bias[np.isnan(bias)] = 0
#     if iid.ndim > 1:
#         iid = np.squeeze(iid)
#     print("iid", iid)
#     print("uind", uind)
#     _, ia, _ = intersect_mtlb(iid, uind)
#     ia = ia.astype(int)
#     print("ia", ia)
#
#     # ICOADS obs
#     sst_obs = W @ x_obs  # - clim[ia] - bias[ia]
#     print("SST OBS", sst_obs)
#
#     # S is the spatial covariance between all "measured" grid points
#     covar = np.copy(cci_covariance)
#     S = covar[ia[:, None], ia[None, :]]
#     print("S", S)
#
#     # Ss is the covariance between to be "predicted" grid points (i.e. all) and "measured" points
#     Ss = covar[ia, :]
#     print("Ss", Ss)
#
#     # G is the weight vector for Simple Kriging
#     G = np.transpose(Ss) @ np.linalg.inv(S)
#     z_obs_sk = G @ sst_obs
#     print("G", G)
#     print("z obs sk", z_obs_sk)
#     CG = G @ Ss
#     print("CG", CG)
#     diagonal = np.diag(covar - CG)
#     # diagonal[abs(diagonal) < 1e-15] = 0.0
#     dz_sk = np.sqrt(diagonal)
#     dz_sk[np.isnan(dz_sk)] = 0.0
#     print("dz_sk", dz_sk)
#     print("Simple Kriging Done")
#
#     # Now we will convert to ordinary kriging
#     S_ = np.concatenate((S, np.ones((len(ia), 1))), axis=1)
#     S = np.concatenate((S_, np.ones((1, len(ia) + 1))), axis=0)
#     # add a Lagrangian multiplier
#     S[-1, -1] = 0
#
#     Ss = np.concatenate((Ss, np.ones((1, len(iid)))), axis=0)
#
#     G = np.transpose(Ss) @ np.linalg.inv(np.matrix(S))
#     CG = G @ Ss
#
#     sst_obs0 = np.append(sst_obs, 0)
#     z_obs_ok = np.transpose(G @ sst_obs0)
#
#     alpha = G[:, -1]
#
#     diagonal = (np.diag(covar - CG)).reshape(-1, 1)
#     dz_ok = np.sqrt(diagonal - alpha)
#
#     print("Ordinary Kriging Done")
#     # get rid of resulting double brackets
#     a = np.squeeze(np.asarray(z_obs_sk))
#     b = np.squeeze(np.asarray(dz_sk))
#     c = np.squeeze(np.asarray(z_obs_ok))
#     d = np.squeeze(np.asarray(dz_ok))
#     return a, b, c, d


def get_spatial_mean(grid_obs: np.ndarray, covx: np.ndarray) -> float:
    """
    Get spatial mean accounting for auto-correlation.

    Parameters
    ==========

    grid_obs : np.ndarray
        Vector containing observations
    covx : np.ndarray
        Observation covariance matrix

    Returns
    =======
    spatial_mean : float
        The spatial mean defined as (1^T x C^{-1} x 1)^{-1} * (1^T x C^{-1} x z)

    Reference
    =========
    https://www.css.cornell.edu/faculty/dgr2/_static/files/distance_ed_geostats/ov5.pdf
    """
    n = len(grid_obs)
    ones = np.ones(n)
    invcov = ones.T @ np.linalg.inv(covx)

    return float(1 / (invcov @ ones) * (invcov @ grid_obs))
