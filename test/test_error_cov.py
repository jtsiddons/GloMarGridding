import numpy as np
import polars as pl
import pytest  # noqa: F401

from glomar_gridding.distances import haversine_gaussian
from glomar_gridding.error_covariance import (
    correlated_components,
    dist_weight,
    get_weights,
    uncorrelated_components,
)
from glomar_gridding.kriging import prep_obs_for_kriging


def test_uncorr() -> None:
    df = pl.DataFrame(
        {
            "data_type": ["ship", "ship", "ship", "buoy", "buoy", "ows"],
            "val": [1.2, 1.2, 1.2, 0.3, 0.3, 3.0],
        }
    )

    e_cov = uncorrelated_components(df, "data_type", obs_sig_col="val")
    assert np.all(e_cov == np.diag(np.diag(e_cov)))

    df = df.with_columns(pl.from_numpy(np.diag(e_cov)).to_series().alias("res"))

    assert df.select(pl.col("val").eq(pl.col("res"))).to_series().all()

    mapping = {"ship": 1.2, "buoy": 0.3}

    e_cov_mapped = uncorrelated_components(df, "data_type", obs_sig_map=mapping)
    expected = np.asarray([1.44, 1.44, 1.44, 0.09, 0.09, 0.0])

    assert np.all(np.diag(e_cov_mapped) == expected)


def test_corr() -> None:
    df = pl.DataFrame(
        {
            "data_type": ["ship", "ship", "ship", "buoy", "buoy", "ows"],
        }
    )
    mapping = {"ship": 1.2, "buoy": 0.3}
    e_cov = correlated_components(df, "data_type", bias_sig_map=mapping)
    expected = np.asarray([1.44, 1.44, 1.44, 0.09, 0.09, 0.0])

    assert (e_cov == e_cov.T).all()
    assert (expected == np.diag(e_cov)).all()

    # TEST: check same type is val ^ 2, diff type is 0
    assert e_cov[0, 4] == 0.0
    assert e_cov[0, 1] == 1.44
    assert e_cov[3, 4] == 0.09
    assert e_cov[4, 5] == 0.0


def test_weights() -> None:
    n = 20
    n_u_grid_pts = 5
    u_grid_pts = pl.int_range(0, n_u_grid_pts, eager=True)
    obs = pl.Series("obs", np.random.rand(n))
    grid_pts = u_grid_pts.sample(n - n_u_grid_pts, with_replacement=True).alias(
        "grid_idx"
    )
    # Ensure have all unique indexes
    grid_pts = grid_pts.extend(pl.Series([0, 1, 2, 3, 4]))

    df = grid_pts.to_frame()
    df = df.with_columns(obs)
    lens = (
        df.group_by("grid_idx").len().sort("grid_idx").select("len").to_series()
    )

    weights = get_weights(df)

    assert weights.shape == (n_u_grid_pts, n)
    assert np.allclose(np.sum(weights, axis=1), 1.0)
    assert (np.sum(weights != 0, axis=1) == lens.to_numpy()).all()

    _, grid_obs = prep_obs_for_kriging(
        grid_pts.to_numpy(), u_grid_pts.to_numpy(), weights, obs.to_numpy()
    )

    assert len(grid_obs) == n_u_grid_pts

    summ = (
        df.group_by("grid_idx")
        .mean()
        .sort("grid_idx")
        .get_column("obs")
        .to_numpy()
    )
    assert np.allclose(grid_obs, summ)


def test_distweight() -> None:
    df = pl.DataFrame(
        {
            "lon": [23.1, 23.9, 23.45, 45.1, 45.6],
            "lat": [-19.8, -19.2, -19.0, 71.4, 71.6],
            "grid_idx": [0, 0, 0, 1, 1],
        }
    )

    dist, weights = dist_weight(df, haversine_gaussian, s=0.14)

    assert (np.diag(dist) == 0.14 / 2).all()
    assert dist[0, 1] != 0.0
    assert dist[0, 4] == 0.0

    weights_2 = get_weights(df)

    assert (weights == weights_2).all()
