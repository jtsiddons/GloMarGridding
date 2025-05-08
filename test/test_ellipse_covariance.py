import os
import numpy as np
from glomar_gridding.io import load_array, load_dataset
from glomar_gridding.ellipse_covariance import EllipseCovarianceBuilder


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

    ellipseCov.uncompress_cov()

    assert ellipseCov.cov_ns.shape[0] == ellipseCov.cov_ns.shape[1]
    assert ellipseCov.cov_ns.shape[0] == len(Lxs) ** 2
