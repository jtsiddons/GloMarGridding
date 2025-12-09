import pytest  # noqa: F401

import numpy as np

from glomar_gridding.spline import ThinPlateSpline, SphericalThinPlateSpline


def test_tps_spline() -> None:
    np.random.seed(314159)
    # TEST: lambda = 0 does not smooth
    n_samps = 100

    X = 1 - 2 * np.random.rand(n_samps, 2)
    X[:, 0] *= np.pi / 2
    X[:, 1] *= np.pi
    y = 0.5 - (
        np.sin(2 * X[:, 0])
        + np.cos(2 * X[:, 1])
        + 0.1 * np.random.randn(n_samps)
    )

    tps = ThinPlateSpline(X=X, y=y)
    tps.fit(lam=0.0, set_results=True)
    y_0 = tps.solve()

    assert np.allclose(y_0, y)

    stps = SphericalThinPlateSpline(X=X, y=y)
    stps.fit(lam=0.0, set_results=True)
    y_0 = stps.solve()

    assert np.allclose(y_0, y)
    return None
