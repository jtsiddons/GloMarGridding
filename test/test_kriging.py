"""
Test of ordinary kriging against the results from the example available from
the GeoStats.jl Julia package. The example can be found at

https://juliaearth.github.io/GeoStatsDocs/stable/interpolation/#Kriging

Here we shrink the domain to 20 x 20 for speed/memory for simple testing
purposes and we replace the variogram with a MaternVariogram with the following
parameters:

* range : 35.0
* sill (variance) : 4.0
* nugget : 0.0
* shape (nu) : 1.5

The GeoStats.jl implementation of MaternVariogram is slightly different see:

https://juliaearth.github.io/GeoStatsDocs/stable/variograms/#GeoStatsFunctions.MaternVariogram

We need to adjust the range to our range value: ((our) range = range / 3) for
GeoStats.jl range parameter.

The intent of this test is to make sure that our implementation of Ordinary
Kriging is correct.
"""

import pytest  # noqa: F401
import os
import numpy as np
import polars as pl
import xarray as xr
from itertools import product

from sklearn.metrics.pairwise import euclidean_distances

from glomar_gridding.grid import (
    map_to_grid,
    grid_from_resolution,
    grid_to_distance_matrix,
)
from glomar_gridding.variogram import MaternVariogram
from glomar_gridding.kriging import (
    SimpleKriging,
    constraint_mask,
    kriging_ordinary,
    OrdinaryKriging,
    kriging_simple,
    _extended_inverse,
)


def _load_results() -> np.ndarray:
    data_path = os.path.join(
        os.path.dirname(__file__), "data", "geostatsjl_ord_krig_results.dat"
    )
    with open(data_path, "r") as io:
        lines = io.readlines()
    converted = [float(line) for line in lines]
    # NOTE: Julia uses "F" style ordering
    return np.reshape(converted, (20, 20), "F")


EXPECTED = _load_results()


def test_ordinary_kriging() -> None:  # noqa: D103
    grid = grid_from_resolution(1, [(1, 21), (1, 21)], ["lat", "lon"])
    obs = pl.DataFrame(
        {
            "lat": [5.0, 15.0, 10.0],
            "lon": [5.0, 10.0, 15.0],
            "val": [1.0, 0.0, 1.0],
        }
    ).pipe(map_to_grid, grid, grid_coords=["lat", "lon"])
    dist: xr.DataArray = grid_to_distance_matrix(grid, euclidean_distances)

    variogram = MaternVariogram(range=35 / 3, psill=4.0, nugget=0.0, nu=1.5)

    covariance: xr.DataArray = variogram.fit(dist)  # type: ignore

    grid_idx = obs.get_column("grid_idx").to_numpy()
    obs_vals = obs.get_column("val").to_numpy()

    S = covariance.values[grid_idx[:, None], grid_idx[None, :]]
    SS = covariance.values[grid_idx, :]
    k, _ = kriging_ordinary(S, SS, obs_vals, covariance.values)

    assert np.allclose(EXPECTED, np.reshape(k, (20, 20), "C"))  # noqa: S101
    return None


def test_ordinary_kriging_class() -> None:  # noqa: D103
    grid = grid_from_resolution(1, [(1, 21), (1, 21)], ["lat", "lon"])
    obs = pl.DataFrame(
        {
            "lat": [5.0, 15.0, 10.0],
            "lon": [5.0, 10.0, 15.0],
            "val": [1.0, 0.0, 1.0],
        }
    ).pipe(map_to_grid, grid, grid_coords=["lat", "lon"])
    dist: xr.DataArray = grid_to_distance_matrix(grid, euclidean_distances)

    variogram = MaternVariogram(range=35 / 3, psill=4.0, nugget=0.0, nu=1.5)

    covariance: xr.DataArray = variogram.fit(dist)  # type: ignore

    grid_idx = obs.get_column("grid_idx").to_numpy()
    obs_vals = obs.get_column("val").to_numpy()

    OKrige = OrdinaryKriging(covariance=covariance.values)

    k = OKrige.solve(obs_vals, grid_idx)

    assert np.allclose(EXPECTED, np.reshape(k, (20, 20), "C"))  # noqa: S101
    return None


def test_ordinary_kriging_class_from_weights() -> None:  # noqa: D103
    grid = grid_from_resolution(1, [(1, 21), (1, 21)], ["lat", "lon"])
    obs = pl.DataFrame(
        {
            "lat": [5.0, 15.0, 10.0],
            "lon": [5.0, 10.0, 15.0],
            "val": [1.0, 0.0, 1.0],
        }
    ).pipe(map_to_grid, grid, grid_coords=["lat", "lon"])
    dist: xr.DataArray = grid_to_distance_matrix(grid, euclidean_distances)

    variogram = MaternVariogram(range=35 / 3, psill=4.0, nugget=0.0, nu=1.5)

    covariance: xr.DataArray = variogram.fit(dist)  # type: ignore

    grid_idx = obs.get_column("grid_idx").to_numpy()
    obs_vals = obs.get_column("val").to_numpy()

    obs_obs_cov = covariance.values[grid_idx[:, None], grid_idx[None, :]]
    obs_grid_cov = covariance.values[grid_idx, :]
    N, M = obs_grid_cov.shape
    obs_obs_cov = np.block(
        [[obs_obs_cov, np.ones((N, 1))], [np.ones((1, N)), 0]]
    )
    obs_grid_cov = np.concatenate((obs_grid_cov, np.ones((1, M))), axis=0)
    kriging_weights = np.linalg.solve(obs_obs_cov, obs_grid_cov).T

    OKrige = OrdinaryKriging(covariance=covariance.values)
    OKrige.set_kriging_weights(kriging_weights)

    k = OKrige.solve(obs_vals, grid_idx)

    assert np.allclose(EXPECTED, np.reshape(k, (20, 20), "C"))  # noqa: S101
    return None


def test_ordinary_kriging_class_from_inv() -> None:  # noqa: D103
    grid = grid_from_resolution(1, [(1, 21), (1, 21)], ["lat", "lon"])
    obs = pl.DataFrame(
        {
            "lat": [5.0, 15.0, 10.0],
            "lon": [5.0, 10.0, 15.0],
            "val": [1.0, 0.0, 1.0],
        }
    ).pipe(map_to_grid, grid, grid_coords=["lat", "lon"])
    dist: xr.DataArray = grid_to_distance_matrix(grid, euclidean_distances)

    variogram = MaternVariogram(range=35 / 3, psill=4.0, nugget=0.0, nu=1.5)

    covariance: xr.DataArray = variogram.fit(dist)  # type: ignore

    grid_idx = obs.get_column("grid_idx").to_numpy()
    obs_vals = obs.get_column("val").to_numpy()

    S = covariance.values[grid_idx[:, None], grid_idx[None, :]]
    S_inv = np.linalg.inv(S)

    OKrige = OrdinaryKriging(covariance=covariance.values)

    S_ext_inv = OKrige.extended_inverse(S_inv)
    OKrige.kriging_weights_from_inverse(S_ext_inv, grid_idx)

    k = OKrige.solve(obs_vals, grid_idx)

    assert np.allclose(EXPECTED, np.reshape(k, (20, 20), "C"))  # noqa: S101
    return None


def test_ordinary_kriging_class_methods() -> None:  # noqa: D103
    grid = grid_from_resolution(1, [(1, 21), (1, 21)], ["lat", "lon"])
    obs = pl.DataFrame(
        {
            "lat": [5.0, 15.0, 10.0],
            "lon": [5.0, 10.0, 15.0],
            "val": [1.0, 0.0, 1.0],
        }
    ).pipe(map_to_grid, grid, grid_coords=["lat", "lon"])

    grid_idx = obs.get_column("grid_idx").to_numpy()
    obs_vals = obs.get_column("val").to_numpy()

    dist: xr.DataArray = grid_to_distance_matrix(grid, euclidean_distances)

    variogram = MaternVariogram(range=35 / 3, psill=4.0, nugget=0.0, nu=1.5)

    covariance: xr.DataArray = variogram.fit(dist)  # type: ignore

    err_cov = np.full(covariance.shape, np.nan)
    err_cov_vals = np.random.rand(3, 3)
    err_cov_vals = np.dot(err_cov_vals, err_cov_vals.T)
    idx = list(product(grid_idx, grid_idx))
    for i, val in zip(idx, err_cov_vals.flatten()):
        err_cov[*i] = val

    OKrige = OrdinaryKriging(covariance=covariance.values)
    k = OKrige.solve(obs_vals, grid_idx, error_cov=err_cov)
    u = OKrige.get_uncertainty(grid_idx)
    a = OKrige.constraint_mask(grid_idx)

    assert k.shape == a.shape == u.shape

    S = covariance.values[grid_idx[:, None], grid_idx[None, :]] + err_cov_vals
    SS = covariance.values[grid_idx, :]
    k2, u2 = kriging_ordinary(S, SS, obs_vals, covariance.values)

    assert np.allclose(k2, k)

    assert np.allclose(u2, u)

    return None


def test_simple_kriging_class_methods() -> None:  # noqa: D103
    grid = grid_from_resolution(1, [(1, 21), (1, 21)], ["lat", "lon"])
    obs = pl.DataFrame(
        {
            "lat": [5.0, 15.0, 10.0],
            "lon": [5.0, 10.0, 15.0],
            "val": [1.0, 0.0, 1.0],
        }
    ).pipe(map_to_grid, grid, grid_coords=["lat", "lon"])

    grid_idx = obs.get_column("grid_idx").to_numpy()
    obs_vals = obs.get_column("val").to_numpy()

    dist: xr.DataArray = grid_to_distance_matrix(grid, euclidean_distances)

    variogram = MaternVariogram(range=35 / 3, psill=4.0, nugget=0.0, nu=1.5)

    covariance: xr.DataArray = variogram.fit(dist)  # type: ignore

    err_cov = np.full(covariance.shape, np.nan)
    err_cov_vals = np.random.rand(3, 3)
    err_cov_vals = np.dot(err_cov_vals, err_cov_vals.T)
    idx = list(product(grid_idx, grid_idx))
    for i, val in zip(idx, err_cov_vals.flatten()):
        err_cov[*i] = val

    SKrige = SimpleKriging(covariance=covariance.values)
    k = SKrige.solve(obs_vals, grid_idx, error_cov=err_cov)
    u = SKrige.get_uncertainty(grid_idx)
    a = SKrige.constraint_mask(grid_idx)

    assert k.shape == a.shape == u.shape

    S = covariance.values[grid_idx[:, None], grid_idx[None, :]] + err_cov_vals
    SS = covariance.values[grid_idx, :]
    k2, u2 = kriging_simple(S, SS, obs_vals, covariance.values)

    a2 = constraint_mask(S, SS, covariance.values)

    assert np.allclose(k2, k)

    assert np.allclose(u2, u)

    assert np.allclose(a2, a)

    return None


@pytest.mark.parametrize(
    "name, n",
    [
        ("n = 10", 10),
        ("n = 25", 25),
        ("n = 100", 100),
        ("n = 2592", 2592),
    ],
)
def test_inverse_trick(name, n):
    A = np.random.rand(n, n)
    S = np.dot(A, A.T)

    Sinv = np.linalg.inv(S)
    Sinv_ext = _extended_inverse(Sinv)

    S_ext = np.block([[S, np.ones((n, 1))], [np.ones((1, n)), 0]])

    assert np.allclose(Sinv_ext, np.linalg.inv(S_ext))
