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

from sklearn.metrics.pairwise import euclidean_distances

from glomar_gridding.grid import (
    align_to_grid,
    grid_from_resolution,
    grid_to_distance_matrix,
)
from glomar_gridding.variogram import MaternVariogram
from glomar_gridding.kriging import kriging_ordinary


def _load_results() -> np.ndarray:
    data_path = os.path.join(
        os.path.dirname(__file__), "data", "geostatsjl_ord_krig_results.dat"
    )
    with open(data_path, "r") as io:
        lines = io.readlines()
    converted = [float(line) for line in lines]
    # NOTE: Julia uses "F" style ordering
    return np.reshape(converted, (20, 20), "F")


def test_ordinary_kriging() -> None:  # noqa: D103
    expected = _load_results()
    grid = grid_from_resolution(1, [(1, 21), (1, 21)], ["lat", "lon"])
    obs = pl.DataFrame(
        {
            "lat": [5.0, 15.0, 10.0],
            "lon": [5.0, 10.0, 15.0],
            "val": [1.0, 0.0, 1.0],
        }
    ).pipe(align_to_grid, grid, grid_coords=["lat", "lon"])
    dist: xr.DataArray = grid_to_distance_matrix(grid, euclidean_distances)

    variogram = MaternVariogram(range=35 / 3, psill=4.0, nugget=0.0, nu=1.5)

    covariance: xr.DataArray = variogram.fit(dist)

    grid_idx = obs.get_column("grid_idx").to_numpy()
    obs_vals = obs.get_column("val").to_numpy()

    S = covariance.values[grid_idx[:, None], grid_idx[None, :]]
    SS = covariance.values[grid_idx, :]
    k, _ = kriging_ordinary(S, SS, obs_vals, covariance.values)

    assert np.allclose(expected, np.reshape(k, (20, 20), "C"))  # noqa: S101
    return None
