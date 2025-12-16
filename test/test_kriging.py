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

import os
from itertools import product

import numpy as np
import polars as pl
import pytest  # noqa: F401
import xarray as xr
from sklearn.metrics.pairwise import euclidean_distances

from glomar_gridding.grid import (
    grid_from_resolution,
    grid_to_distance_matrix,
    map_to_grid,
    Grid,
)
from glomar_gridding.kriging import (
    OrdinaryKriging,
    SimpleKriging,
    _extended_inverse,
    constraint_mask,
    kriging_ordinary,
    kriging_simple,
)
from glomar_gridding.stochastic import StochasticKriging
from glomar_gridding.variogram import MaternVariogram


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

    OKrige = OrdinaryKriging(
        covariance=covariance.values, idx=grid_idx, obs=obs_vals
    )

    k = OKrige.solve()

    assert np.allclose(EXPECTED, np.reshape(k, (20, 20), "C"))  # noqa: S101
    return None


def test_ordinary_kriging_class_from_grid_class() -> None:
    grid = Grid.from_resolution(1, [(1, 21), (1, 21)], ["lat", "lon"])
    grid.distance_matrix(dist_func=euclidean_distances)
    grid.covariance_matrix(
        variogram="matern",
        range=35 / 3,
        psill=4.0,
        nugget=0.0,
        nu=1.5,
    )

    obs = pl.DataFrame(
        {
            "lat": [5.0, 15.0, 10.0],
            "lon": [5.0, 10.0, 15.0],
            "val": [1.0, 0.0, 1.0],
        }
    )

    obs = grid.map_observations(obs, obs_col="val")

    grid.kriging(kriging_method="ordinary")
    k = grid.krige.solve()

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
    print(kriging_weights.shape)
    print(len(grid_idx))

    OKrige = OrdinaryKriging(
        covariance=covariance.values, idx=grid_idx, obs=obs_vals
    )
    OKrige.set_kriging_weights(kriging_weights)

    k = OKrige.solve()

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

    OKrige = OrdinaryKriging(
        covariance=covariance.values, idx=grid_idx, obs=obs_vals
    )

    S_ext_inv = OKrige.extended_inverse(S_inv)
    OKrige.kriging_weights_from_inverse(S_ext_inv)

    k = OKrige.solve()

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

    OKrige = OrdinaryKriging(
        covariance=covariance.values,
        idx=grid_idx,
        obs=obs_vals,
        error_cov=err_cov,
    )
    k = OKrige.solve()
    u = OKrige.get_uncertainty()
    a = OKrige.constraint_mask()

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

    SKrige = SimpleKriging(
        covariance=covariance.values,
        idx=grid_idx,
        obs=obs_vals,
        error_cov=err_cov,
    )
    k = SKrige.solve()
    u = SKrige.get_uncertainty()
    a = SKrige.constraint_mask()

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
        ("n = 1000", 1000),
    ],
)
def test_inverse_trick(name, n):
    np.random.seed(31900)
    A = np.random.rand(n, n)
    S = np.dot(A, A.T)

    Sinv = np.linalg.inv(S)
    Sinv_ext = _extended_inverse(Sinv)

    S_ext = np.block([[S, np.ones((n, 1))], [np.ones((1, n)), 0]])

    assert np.allclose(Sinv_ext, np.linalg.inv(S_ext))


def test_stochastic_kriging_class_methods() -> None:  # noqa: D103
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

    StochKrige = StochasticKriging(
        covariance=covariance.values,
        idx=grid_idx,
        obs=obs_vals,
        error_cov=err_cov,
    )
    k = StochKrige.solve()
    u = StochKrige.get_uncertainty()
    a = StochKrige.constraint_mask()

    assert k.shape == a.shape == u.shape

    S = covariance.values[grid_idx[:, None], grid_idx[None, :]] + err_cov_vals
    SS = covariance.values[grid_idx, :]
    k2, u2 = kriging_ordinary(S, SS, obs_vals, covariance.values)

    assert hasattr(StochKrige, "gridded_field")
    assert np.allclose(k2, StochKrige.gridded_field)
    assert np.allclose(u2, u)

    sk_weights = StochKrige.simple_kriging_weights
    delattr(StochKrige, "simple_kriging_weights")
    StochKrige.set_simple_kriging_weights(sk_weights)
    a2 = StochKrige.constraint_mask()

    assert np.allclose(a, a2)

    return None


def test_filter_bad_error_cov_values() -> None:
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
    # Add a nan on the diagonal
    err_cov_vals = np.dot(err_cov_vals, err_cov_vals.T)
    err_cov_vals[2, 2] = np.nan
    idx = list(product(grid_idx, grid_idx))
    for i, val in zip(idx, err_cov_vals.flatten()):
        err_cov[*i] = val

    expected_warn = (
        "Have nans or zeros on the error covariance diagonal. "
        + f"At positions {grid_idx[2]}. Filtering input accordingly"
    )
    with pytest.warns(UserWarning, match=expected_warn):
        OKrige = OrdinaryKriging(
            covariance.values,
            idx=grid_idx,
            obs=obs_vals,
            error_cov=err_cov,
        )

    assert (OKrige.idx == grid_idx[:2]).all()
    assert (OKrige.obs == obs_vals[:2]).all()
    assert OKrige.error_cov.shape == (2, 2)


def test_filter_bad_error_cov_values_stochastic() -> None:
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
    # Add a nan on the diagonal
    err_cov_vals = np.dot(err_cov_vals, err_cov_vals.T)
    err_cov_vals[2, 2] = np.nan
    idx = list(product(grid_idx, grid_idx))
    for i, val in zip(idx, err_cov_vals.flatten()):
        err_cov[*i] = val

    expected_warn = (
        "Have nans or zeros on the error covariance diagonal. "
        + f"At positions {grid_idx[2]}. Filtering input accordingly"
    )
    with pytest.warns(UserWarning, match=expected_warn):
        OKrige = StochasticKriging(
            covariance.values,
            idx=grid_idx,
            obs=obs_vals,
            error_cov=err_cov,
        )

    assert (OKrige.idx == grid_idx[:2]).all()
    assert (OKrige.obs == obs_vals[:2]).all()
    assert OKrige.error_cov.shape == (2, 2)
