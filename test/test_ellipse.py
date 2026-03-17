"""Tests for Ellipse Parameter estimation"""

import os

import numpy as np
import pytest
import xarray as xr

from glomar_gridding.covariance_tools import eigenvalue_clip
from glomar_gridding.ellipse import (
    EllipseBuilder,
    EllipseCovarianceBuilder,
    EllipseModel,
)
from glomar_gridding.io import load_array, load_dataset
from glomar_gridding.utils import cov_2_cor, uncompress_masked


def frob(mat: np.ndarray) -> float:
    return float(np.linalg.norm(mat, ord="fro"))  # codespell:ignore


def correlation_distance(cov1: np.ndarray, cov2: np.ndarray) -> float:
    """
    Correlation Matrix Distance

    from: https://www.researchgate.net/publication/4194743_Correlation_Matrix_Distance_a_Meaningful_Measure_for_Evaluation_of_Non-Stationary_MIMO_Channels
    """
    cor1 = cov_2_cor(cov1)
    cor2 = cov_2_cor(cov2)
    num = np.trace(np.matmul(cor1, cor2))
    denom = frob(cor1) * frob(cor2)
    return 1 - num / denom


def initialise_const_arrays(
    Lx: float,
    Ly: float | None,
    theta: float | None,
    stdev: float,
    size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray]:
    Lx_arr = np.full(size, Lx)
    Ly_arr = np.full(size, Ly) if Ly is not None else None
    theta_arr = np.full(size, theta) if theta is not None else None
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
    "name, v, params, size",
    [
        (
            "ellipse",
            1.5,
            {"Lx": 1500, "Ly": 800, "theta": np.pi / 3, "stdev": 0.6},
            (10, 6),
        ),
        (
            "rotated ellipses",
            1.5,
            {"Lx": 3600, "Ly": 1700, "theta": 0.2, "stdev": 1.2},
            (8, 8),
        ),
        (
            "unrotated ellipses",
            1.5,
            {"Lx": 3600, "Ly": 1700, "theta": None, "stdev": 1.2},
            (8, 8),
        ),
        (
            "circles",
            1.5,
            {"Lx": 1200, "Ly": None, "theta": None, "stdev": 1.2},
            (8, 8),
        ),
    ],
)
def test_const_Ellipse(name, v, params, size):
    # TEST: That ellipse stuff is self-consistent
    #       If one generates data from a covariance derived from known
    #       ellipse parameters, test that you get the same covariance out
    #       after estimating ellipse parameters from data drawn from that
    #       initial covariance matrix
    np.random.seed(40814)

    # Generate Test Data from A Known Covariance (from known Ellipse Params)
    n = 5_000
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
        anisotropic=params["Ly"] is not None,
        rotated=params["theta"] is not None,
        physical_distance=True,
        v=v,
        unit_sigma=True,
    )
    ellipse_builder = EllipseBuilder(test_data, coords)

    # Set-up output fields
    v = ellipse.v
    nparams = ellipse.supercategory_n_params
    default_values = [0.0 for _ in range(nparams)]
    # init_values = [300.0, 300.0, 0.0]
    # fit_bounds = [
    #     (300.0, 30000.0),
    #     (300.0, 30000.0),
    #     (-0.5 * np.pi, 0.5 * np.pi),
    # ]
    fit_max_distance = 10_000.0

    # Estimate Ellipse Parameters
    ellipse_params = ellipse_builder.compute_params(
        default_value=default_values,
        matern_ellipse=ellipse,
        # bounds=fit_bounds,
        # guesses=init_values,
        max_distance=fit_max_distance,
        delta_x_method="Modified_Met_Office",
    )

    Lx = (
        ellipse_params["Lx"].values
        if params["Ly"] is not None
        else ellipse_params["R"].values
    )
    Ly = ellipse_params["Ly"].values if params["Ly"] is not None else None
    theta = (
        ellipse_params["theta"].values if params["theta"] is not None else None
    )
    stdev = ellipse_params["standard_deviation"].values

    ellipse_cov = EllipseCovarianceBuilder(
        Lx,
        Ly,
        theta,
        stdev,
        lons=coords["longitude"].values,
        lats=coords["latitude"].values,
        v=v,
    ).cov_ns
    ellipse_cov = eigenvalue_clip(
        ellipse_cov,
        method="explained_variance",
        target_variance_fraction=0.99,
    )

    assert np.allclose(ellipse_cov, in_cov, rtol=5e-2)

    cmd = correlation_distance(in_cov, ellipse_cov)
    assert cmd < 1e-4


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

    cmd = correlation_distance(ellipseCov.cov_ns, expected)
    assert cmd < 1e-4
    assert np.allclose(ellipseCov.cov_ns, expected, rtol=1e-5)

    # TEST: correlation matrix
    ellipseCov.calculate_cor()
    assert hasattr(ellipseCov, "cor_ns")
    assert np.isclose(1, np.max(np.diag(ellipseCov.cor_ns)))


def test_ellipse_covariance_self_consistency():
    """Test covariance result matches known result (from @stchan)"""
    in_file = os.path.join(
        os.path.dirname(__file__), "data", "Atlantic_Ocean_07.nc"
    )
    Lxs = load_array(in_file, "lx")[50:70, 50:70]
    in_coords = Lxs.coords
    mask = Lxs.values > 1e5

    cov_file = os.path.join(os.path.dirname(__file__), "data", "cov_no_hfix.nc")

    known_cov = load_array(cov_file, "covariance").values
    n = 1_000
    coord_dict = {
        "time": np.arange(n),
        "longitude": in_coords["longitude"].values,
        "latitude": in_coords["latitude"].values,
    }
    coords = xr.Coordinates(coord_dict)

    test_data = get_test_data(known_cov, n)
    test_data = np.array(
        [
            uncompress_masked(test_data[i, :], mask, fill_value=np.nan)
            for i in range(n)
        ]
    )
    test_data = test_data.reshape((n, *Lxs.shape))
    test_data = np.ma.masked_where(np.isnan(test_data), test_data)

    ellipse = EllipseModel(
        anisotropic=True,
        rotated=True,
        physical_distance=True,
        v=0.5,
        unit_sigma=True,
    )
    ellipse_builder = EllipseBuilder(test_data, coords)

    # Set-up output fields
    v = ellipse.v
    nparams = ellipse.supercategory_n_params
    default_values = [-999.0 for _ in range(nparams)]
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
        delta_x_method="Modified_Met_Office",
    )

    Lx = ellipse_params["Lx"].values
    Ly = ellipse_params["Ly"].values
    theta = ellipse_params["theta"].values
    stdev = ellipse_params["standard_deviation"].values

    ellipseCov = EllipseCovarianceBuilder(
        np.ma.masked_less(Lx, -900.0),
        np.ma.masked_less(Ly, -900.0),
        np.ma.masked_less(theta, -900.0),
        np.ma.masked_less(stdev, -900.0),
        in_coords["latitude"].values,
        in_coords["longitude"].values,
        v=v,
    ).cov_ns

    cdm = correlation_distance(ellipseCov, ellipse_builder.cov)
    assert cdm < 1e-3


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
