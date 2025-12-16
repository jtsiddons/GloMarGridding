import numpy as np
import pytest

from glomar_gridding.grid import grid_from_resolution, grid_to_distance_matrix
from glomar_gridding.variogram import (
    ExponentialVariogram,
    GaussianVariogram,
    MaternVariogram,
    SphericalVariogram,
    variogram_to_covariance,
)

GRID = grid_from_resolution(
    5, [(-25, 25), (-25, 25)], ["latitude", "longitude"]
)

DIST = grid_to_distance_matrix(
    GRID, lat_coord="latitude", lon_coord="longitude"
)


@pytest.mark.parametrize(
    "variogram_model, parameters, variance",
    [
        (
            GaussianVariogram,
            {"psill": 1.2, "nugget": 0, "range": 1200},
            1.2,
        ),
        (
            GaussianVariogram,
            {"psill": 1.2, "nugget": 0.101, "range": 1200},
            1.2,
        ),
        (
            ExponentialVariogram,
            {"psill": 1.2, "nugget": 0, "range": 1200},
            1.2,
        ),
        (
            SphericalVariogram,
            {"psill": 1.2, "nugget": 0, "range": 1200},
            1.2,
        ),
        (
            MaternVariogram,
            {
                "psill": 1.2,
                "nugget": 0,
                "range": 1200,
                "nu": 1.5,
                "method": "sklearn",
            },
            1.2,
        ),
        (
            MaternVariogram,
            {
                "psill": 1.2,
                "nugget": 0,
                "range": 1200,
                "nu": 0.5,
                "method": "sklearn",
            },
            1.2,
        ),
        (
            MaternVariogram,
            {
                "psill": 1.2,
                "nugget": 0,
                "range": 1200,
                "nu": 3.0,
                "method": "sklearn",
            },
            1.2,
        ),
        (
            MaternVariogram,
            {
                "psill": 1.2,
                "nugget": 0,
                "range": 1200,
                "nu": 1.5,
                "method": "gstat",
            },
            1.2,
        ),
        (
            MaternVariogram,
            {
                "psill": 1.2,
                "nugget": 0,
                "range": 1200,
                "nu": 1.5,
                "method": "karspeck",
            },
            1.2,
        ),
        (
            MaternVariogram,
            {
                "psill": 1.2,
                "nugget": 0.105,
                "range": 1200,
                "nu": 1.5,
                "method": "karspeck",
            },
            1.2,
        ),
    ],
)
def test_variogram(variogram_model, parameters, variance):
    # TEST: Generates a valid covariance
    variogram = variogram_model(**parameters)
    variogram_result_xr = variogram.fit(DIST)
    variogram_result = variogram.fit(DIST.values)

    # TEST: xarray and numpy versions are the same
    assert np.allclose(variogram_result, variogram_result_xr.values)
    variogram_result_xr = None

    covariance = variogram_to_covariance(variogram_result, variance)

    evals = np.linalg.eigvalsh(covariance)

    # TEST: symmetric
    assert np.allclose(covariance, covariance.T)
    # TEST: positive definite
    assert (evals > 0).all()

    return None
