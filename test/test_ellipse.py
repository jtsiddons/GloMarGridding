"""Tests for Ellipse Parameter estimation"""

import os
import pytest

import numpy as np
import scipy as sp
import xarray as xr

from glomar_gridding.io import load_array, load_dataset
from glomar_gridding.covariance_tools import eigenvalue_clip
from glomar_gridding.ellipse import (
    EllipseModel,
    EllipseBuilder,
    EllipseCovarianceBuilder,
)


def chisq(
    sigma_hat: np.ndarray,
    sigma_actual: np.ndarray,
    n: int,
) -> float:
    """
    For cases here have fixed parameters is easy
    So each grid point as well as the average estimated parameter (the estimator)
    should be within 1-2x of the standard error of the actual parameter
    What is the likelihood of the estimator relative to the actual?

    For variable params... we will just compare the regional average

    W = -n*np.log(np.linalg.det(inside_the_brackets))-n*p+n*np.trace(inside_the_brackets)
    W ~ ChiSquare(p*(p+1)/2)

    How to:
    https://www.stat.pitt.edu/sungkyu/course/2221Fall13/lec3.pdf
    https://cran.r-project.org/web//packages/mvhtests/mvhtests.pdf <<< R docs (argh)
    """  # noqa: E501
    # unbiased estimator to sigma_hat for all practical purpose n/n-1 ~ 1
    sigma_hat_unbiased = (n / (n - 1)) * sigma_hat
    sigma_actual_inv = np.linalg.inv(sigma_actual)

    # Number of parameters (aka the dimension of the covariance) -- 2
    p = sigma_actual.shape[0]
    inside_the_brackets = sigma_actual_inv @ sigma_hat_unbiased

    W = (
        -n * np.log(np.linalg.det(inside_the_brackets))
        - n * p
        + n * np.trace(inside_the_brackets)
    )
    p_val = sp.stats.chi2.sf(W, p * (p + 1) / 2)
    return p_val


def initialise_const_arrays(
    Lx: float,
    Ly: float,
    theta: float,
    stdev: float,
    size: tuple[int, int],
) -> tuple[np.ndarray, ...]:
    Lx_arr = np.full(size, Lx)
    Ly_arr = np.full(size, Ly)
    theta_arr = np.full(size, theta)
    stdev_arr = np.full(size, stdev)
    return Lx_arr, Ly_arr, theta_arr, stdev_arr


def initialise_covariance(
    Lx: float,
    Ly: float,
    theta: float,
    stdev: float,
    v: float,
    size: tuple[int, int],
) -> np.ndarray:
    Lx_arr, Ly_arr, theta_arr, stdev_arr = initialise_const_arrays(
        Lx, Ly, theta, stdev, size
    )
    lons = np.arange(size[1], dtype=np.float32)
    lats = np.arange(size[0], dtype=np.float32)
    out = EllipseCovarianceBuilder(
        Lx_arr,
        Ly_arr,
        theta_arr,
        stdev_arr,
        v=v,
        lons=lons,
        lats=lats,
    ).cov_ns
    return eigenvalue_clip(
        out,
        method="explained_variance",
        target_variance_fraction=0.99,
    )


def get_test_data(
    cov: np.ndarray,
    n: int,
) -> np.ndarray:
    s = cov.shape[0]
    return np.random.multivariate_normal(np.zeros(s), cov, size=n)


@pytest.mark.parametrize(
    "v, params, size",
    [
        (
            1.5,
            {"Lx": 1500, "Ly": 800, "theta": np.pi / 3, "stdev": 0.6},
            (10, 6),
        ),
        (1.5, {"Lx": 3600, "Ly": 1700, "theta": 0.2, "stdev": 1.2}, (8, 8)),
    ],
)
def test_const_Ellipse(v, params, size):
    # TEST: That ellipse stuff is self-consistent
    #       If one generates data from a covariance derived from known
    #       ellipse parameters, test that you get the same covariance out
    #       after estimating ellipse parameters from data drawn from that
    #       initial covariance matrix
    np.random.seed(40814)

    # Generate Test Data from A Known Covariance (from known Ellipse Params)
    n = 1500
    true_cov = initialise_covariance(**params, v=v, size=size)
    test_data = get_test_data(true_cov, n=n)
    in_cov = np.cov(test_data.T)
    test_data = test_data.reshape((n, *size))
    coord_dict = {
        "time": np.arange(n),
        "longitude": np.arange(size[1], dtype=np.float32),
        "latitude": np.arange(size[0], dtype=np.float32),
    }
    coords = xr.Coordinates(coord_dict)

    # Define Ellipse Model
    ellipse = EllipseModel(
        anisotropic=True,
        rotated=True,
        physical_distance=True,
        v=v,
        unit_sigma=True,
    )
    ellipse_builder = EllipseBuilder(test_data, coords)

    # Set-up output fields
    v = ellipse.v
    nparams = ellipse.supercategory_n_params
    default_values = [0.0 for _ in range(nparams)]
    init_values = [300.0, 300.0, 0.0]
    fit_bounds = [
        (300.0, 30000.0),
        (300.0, 30000.0),
        (-2.0 * np.pi, 2.0 * np.pi),
    ]
    fit_max_distance = 10_000.0

    # Estimate Ellipse Parameters
    ellipse_params = ellipse_builder.compute_params(
        default_value=default_values,
        matern_ellipse=ellipse,
        bounds=fit_bounds,
        guesses=init_values,
        max_distance=fit_max_distance,
    )

    Lx = ellipse_params["Lx"].values
    Ly = ellipse_params["Ly"].values
    theta = ellipse_params["theta"].values
    stdev = ellipse_params["standard_deviation"].values

    simulated_cov = EllipseCovarianceBuilder(
        Lx,
        Ly,
        theta,
        stdev,
        lons=coords["longitude"].values,
        lats=coords["latitude"].values,
        v=v,
    ).cov_ns
    simulated_cov = eigenvalue_clip(
        simulated_cov,
        method="explained_variance",
        target_variance_fraction=0.99,
    )

    # NOTE: a low p-value from chi-sq test indicates difference, want 1-p for
    #       similarity
    p = chisq(simulated_cov, in_cov, n)
    assert (1 - p) < 5e-2


def test_ellipse_covariance():
    """Test covariance result matches known result (from @stchan)"""
    in_file = os.path.join(
        os.path.dirname(__file__), "data", "Atlantic_Ocean_07.nc"
    )
    expected_file = os.path.join(
        os.path.dirname(__file__), "data", "cov_no_hfix.nc"
    )
    expected = load_array(expected_file, "covariance").values

    ds = load_dataset(in_file)
    Lx = ds["lx"][50:70, 50:70]
    Lxs = Lx.values
    lats = Lx.latitude
    lons = Lx.longitude
    # xx, yy = np.meshgrid(lons, lats)

    mask = Lxs > 1e5

    Lys = ds["ly"][50:70, 50:70].values
    thetas = ds["theta"][50:70, 50:70].values
    stdevs = ds["standard_deviation"][50:70, 50:70].values

    ellipseCov = EllipseCovarianceBuilder(
        np.ma.masked_where(mask, Lxs),
        np.ma.masked_where(mask, Lys),
        np.ma.masked_where(mask, thetas),
        np.ma.masked_where(mask, stdevs),
        lats,
        lons,
        v=0.5,
    )

    assert np.allclose(ellipseCov.cov_ns, expected, rtol=1e-5)

    # TEST: correlation matrix
    ellipseCov.calculate_cor()
    assert hasattr(ellipseCov, "cor_ns")
    assert np.isclose(1, np.max(np.diag(ellipseCov.cor_ns)))


def test_ellipse_covariance_methods():
    """Test that all 3 covariance methods yield the same result"""
    in_file = os.path.join(
        os.path.dirname(__file__), "data", "Atlantic_Ocean_07.nc"
    )

    ds = load_dataset(in_file)
    Lx = ds["lx"][50:70, 50:70]
    Lxs = Lx.values
    lats = Lx.latitude
    lons = Lx.longitude
    # xx, yy = np.meshgrid(lons, lats)

    mask = Lxs > 1e5

    Lys = ds["ly"][50:70, 50:70].values
    thetas = ds["theta"][50:70, 50:70].values
    stdevs = ds["standard_deviation"][50:70, 50:70].values

    cov_array = EllipseCovarianceBuilder(
        np.ma.masked_where(mask, Lxs),
        np.ma.masked_where(mask, Lys),
        np.ma.masked_where(mask, thetas),
        np.ma.masked_where(mask, stdevs),
        lats,
        lons,
        v=0.5,
    ).cov_ns

    cov_batched = EllipseCovarianceBuilder(
        np.ma.masked_where(mask, Lxs),
        np.ma.masked_where(mask, Lys),
        np.ma.masked_where(mask, thetas),
        np.ma.masked_where(mask, stdevs),
        lats,
        lons,
        v=0.5,
        covariance_method="batched",
        batch_size=100,
    ).cov_ns

    cov_loop = EllipseCovarianceBuilder(
        np.ma.masked_where(mask, Lxs),
        np.ma.masked_where(mask, Lys),
        np.ma.masked_where(mask, thetas),
        np.ma.masked_where(mask, stdevs),
        lats,
        lons,
        v=0.5,
        covariance_method="low_memory",
    ).cov_ns

    assert np.allclose(cov_array, cov_batched, rtol=1e-5)
    assert np.allclose(cov_array, cov_loop, rtol=1e-5)


def test_ellipse_covariance_rescale():
    """Test covariance result matches known result (from @stchan)"""
    in_file = os.path.join(
        os.path.dirname(__file__), "data", "Atlantic_Ocean_07.nc"
    )

    ds = load_dataset(in_file)
    Lx = ds["lx"][50:70, 50:70]
    Lxs = Lx.values
    lats = Lx.latitude
    lons = Lx.longitude
    # xx, yy = np.meshgrid(lons, lats)

    mask = Lxs > 1e5

    Lys = ds["ly"][50:70, 50:70].values
    thetas = ds["theta"][50:70, 50:70].values
    stdevs = ds["standard_deviation"][50:70, 50:70].values

    ellipseCov = EllipseCovarianceBuilder(
        np.ma.masked_where(mask, Lxs),
        np.ma.masked_where(mask, Lys),
        np.ma.masked_where(mask, thetas),
        np.ma.masked_where(mask, stdevs),
        lats,
        lons,
        v=0.5,
    )

    ellipseCov.uncompress_cov()

    assert ellipseCov.cov_ns.shape[0] == ellipseCov.cov_ns.shape[1]
    assert ellipseCov.cov_ns.shape[0] == len(Lxs) ** 2
