"""
Functions for performing Kriging.

Interpolation using a Gaussian Process. Available methods are Simple and
Ordinary Kriging.
"""
################
# by A. Faulkner
# for python version >= 3.11
################

from typing import Literal
import numpy as np

from .utils import adjust_small_negative, intersect_mtlb

KrigMethod = Literal["simple", "ordinary"]


def kriging(  # noqa: C901
    obs_idx: np.ndarray,
    weights: np.ndarray,
    obs: np.ndarray | None,
    interp_cov: np.ndarray | None,
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
        Grid indices with observations. It is expected that this should be an
        ordering that lines up with the 1st dimension of weights. If
        `observations.dist_weights` or `observations.get_weights` was used to
        get the weights then this is the ordering of
        `sorted(df["gridbox"].unique())`, which is a sorting on lat and lon
    weights : np.ndarray[float]
        Weight matrix (inverse of counts of observations).
    obs : np.ndarray[float]
        All point observations/measurements for the chosen date.
    interp_cov : np.ndarray[float]
        interpolation covariance of all output grid points (each point in time
        and all points against each other).
    error_cov : np.ndarray[float]
        Measurement/Error covariance matrix.
    remove_obs_mean: int
        Should the mean or median from grib_obs be removed and added back onto
        grib_obs?
        0 = No (default action)
        1 = the mean is removed
        2 = the median is removed
        3 = the spatial meam os removed
    obs_bias : np.ndarray[float] | None
        Bias of all measurement points for a chosen date (corresponds to x_obs).
    method : KrigMethod
        The kriging method to use to fill in the output grid. One of "simple"
        or "ordinary".

    Returns
    -------
    z_obs : np.ndarray[float]
        Full set of values for the whole domain derived from the observation
        points using the chosen kriging method.
    dz : np.ndarray[float]
        Uncertainty associated with the chosen kriging method.
    """
    if obs is None or interp_cov is None:
        raise ValueError(
            "Observations and interpolation covariance must be supplied"
        )
    if obs_bias is not None:
        print("With bias")
        grid_obs = weights @ (obs - obs_bias)
    else:
        grid_obs = weights @ obs

    grid_obs = np.squeeze(grid_obs) if len(grid_obs) > 1 else grid_obs

    match remove_obs_mean:
        case 0:
            grid_obs_av = None
        case 1:
            grid_obs_av = np.ma.average(grid_obs)
            grid_obs = grid_obs - grid_obs_av
        case 2:
            grid_obs_av = np.ma.median(grid_obs)
            grid_obs = grid_obs - grid_obs_av
        case 3:
            grid_obs_av = get_spatial_mean(grid_obs, error_cov)
            grid_obs = grid_obs - grid_obs_av
        case _:
            raise ValueError("Unknown 'remove_obs_mean' value")

    print(f"{grid_obs.shape = }")

    if error_cov.shape == interp_cov.shape:
        print(
            "Error covariance supplied is of the same size as interpolation "
            + "covariance, subsetting to indices of observation grids"
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

    match method.lower():
        case "simple":
            print("Performing Simple Kriging")
            z_obs, dz = kriging_simple(S, Ss, grid_obs, interp_cov)
        case "ordinary":
            print("Performing Ordinary Kriging")
            z_obs, dz = kriging_ordinary(S, Ss, grid_obs, interp_cov)
        case _:
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
        Unique indices of all measurement points for a chosen date,
        representative of the indices of gridboxes, which have => 1 measurement.
    weights : np.ndarray[float]
        Weight matrix (inverse of counts of observations).
    obs : np.ndarray[float]
        All point observations/measurements for the chosen date.
    interp_cov : np.ndarray[float]
        Interpolation covariance of all output grid points (each point in time
        and all points
        against each other).
    error_cov : np.ndarray[float]
        Measurement/Error covariance matrix.
    remove_obs_mean: int
        Should the mean or median from obs be removed and added back onto obs?
        0 = No (default action)
        1 = the mean is removed
        2 = the median is removed
        3 = the spatial meam os removed
    obs_bias : np.ndarray[float] | None
        Bias of all measurement points for a chosen date (corresponds to x_obs).
    method : KrigMethod
        The kriging method to use to fill in the output grid. One of "simple"
        or "ordinary".

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
    Get grid indices with observations from un-masked grid-box indices and
    unique grid-box indices with observations.

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


# TODO: Handle mean
def kriging_simple(
    S: np.ndarray,
    Ss: np.ndarray,
    grid_obs: np.ndarray,
    interp_cov: np.ndarray,
    mean: float | np.ndarray = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform Simple Kriging assuming a constant known mean.

    Parameters
    ----------
    S : np.ndarray[float]
        Spatial covariance between all measured grid points plus the
        covariance due to measurements (i.e. measurement noise, bias noise, and
        sampling noise).
    Ss : np.ndarray[float]
        Covariance between the all (predicted) grid points and measured points.
    grid_obs : np.ndarray[float]
        Gridded measurements (all measurement points averaged onto the output
        gridboxes).
    interp_cov : np.ndarray[float]
        interpolation covariance of all output grid points (each point in time
        and all points against each other).
    mean : float
        The constant mean of the output field.

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
    dz_squared = adjust_small_negative(dz_squared)
    dz = np.sqrt(dz_squared)
    print(f"{dz =}")
    dz[np.isnan(dz)] = 0.0

    print("Simple Kriging Complete")
    return z_obs + mean, dz


def kriging_ordinary(
    S: np.ndarray,
    Ss: np.ndarray,
    grid_obs: np.ndarray,
    interp_cov: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform Ordinary Kriging with unknown but constant mean.

    Parameters
    ----------
    S : np.ndarray[float]
        Spatial covariance between all measured grid points plus the
        covariance due to measurements (i.e. measurement noise, bias noise, and
        sampling noise).
    Ss : np.ndarray[float]
        Covariance between the all (predicted) grid points and measured points.
    grid_obs : np.ndarray[float]
        Gridded measurements (all measurement points averaged onto the output
        gridboxes).
    interp_cov : np.ndarray[float]
        Interpolation covariance of all output grid points (each point in time
        and all points against each other).

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
    dz_squared = adjust_small_negative(dz_squared)
    dz = np.sqrt(dz_squared)
    # dz[np.isnan(dz)] = 0.0

    print("Ordinary Kriging Complete")
    return z_obs, dz


def get_spatial_mean(
    grid_obs: np.ndarray,
    covx: np.ndarray,
) -> float:
    """
    Compute the spatial mean accounting for auto-correlation.

    Parameters
    ----------
    grid_obs : np.ndarray
        Vector containing observations
    covx : np.ndarray
        Observation covariance matrix

    Returns
    -------
    spatial_mean : float
        The spatial mean defined as (1^T x C^{-1} x 1)^{-1} * (1^T x C^{-1} x z)

    Reference
    ---------
    https://www.css.cornell.edu/faculty/dgr2/_static/files/distance_ed_geostats/ov5.pdf
    """
    n = len(grid_obs)
    ones = np.ones(n)
    invcov = ones.T @ np.linalg.inv(covx)

    return float(1 / (invcov @ ones) * (invcov @ grid_obs))
