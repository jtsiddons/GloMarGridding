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

    tps = ThinPlateSpline(X, y)
    y_0 = tps.fit(lam=0.0).y_fit

    assert np.allclose(y_0, y)

    stps = SphericalThinPlateSpline(X, y)
    y_0 = stps.fit(lam=0.0).y_fit

    assert np.allclose(y_0, y)
    return None


def test_gcv() -> None:
    np.random.seed(314159)
    n_samps = 100

    X = 1 - 2 * np.random.rand(n_samps, 2)
    X[:, 0] *= np.pi / 2
    X[:, 1] *= np.pi
    y = 0.5 - (
        np.sin(2 * X[:, 0])
        + np.cos(2 * X[:, 1])
        + 0.1 * np.random.randn(n_samps)
    )

    tps = ThinPlateSpline(X, y)
    lam = tps.estimate_lambda_gcv()
    assert lam is not None
    assert lam > 0
    return None


def test_predict() -> None:
    np.random.seed(314159)
    n_samps = 100

    X = 1 - 2 * np.random.rand(n_samps, 2)
    X[:, 0] *= np.pi / 2
    X[:, 1] *= np.pi
    y = 0.5 - (
        np.sin(2 * X[:, 0])
        + np.cos(2 * X[:, 1])
        + 0.1 * np.random.randn(n_samps)
    )
    X_grid = np.mgrid[-0.5:0.5:25j, -1:1:50j]
    X_test = X_grid.reshape(2, -1).T
    X_test *= np.pi

    tps = ThinPlateSpline(X, y)
    tps.fit(lam=0, set_results=True)
    y_pred, _ = tps.predict(X_test)
    assert y_pred.shape == (X_test.shape[0],)

    return None


def test_var() -> None:
    np.random.seed(314159)
    n_samps = 50

    X = 1 - 2 * np.random.rand(n_samps, 2)
    X[:, 0] *= np.pi / 2
    X[:, 1] *= np.pi
    y = 0.5 - (
        np.sin(2 * X[:, 0])
        + np.cos(2 * X[:, 1])
        + 0.1 * np.random.randn(n_samps)
    )

    tps = ThinPlateSpline(X, y)
    result = tps.fit()
    var = result.var

    assert isinstance(var, float)
    assert np.all(var > 0)


# def test_gcv_loss() -> None:
#     np.random.seed(314159)
#     n_samps = 50
#
#     X = 1 - 2 * np.random.rand(n_samps, 2)
#     X[:, 0] *= np.pi / 2
#     X[:, 1] *= np.pi
#     y = 0.5 - (
#         np.sin(2 * X[:, 0])
#         + np.cos(2 * X[:, 1])
#         + 0.1 * np.random.randn(n_samps)
#     )
#
#     tps = ThinPlateSpline(X=X, y=y)
#     lam = tps.estimate_lambda_gcv(method="bounded", bounds=(1e-6, 1e2))
#     print(f"{lam = }")
