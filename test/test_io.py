import os

import xarray as xr

from glomar_gridding.interpolation_covariance import load_covariance
from glomar_gridding.io import load_array, load_dataset

PATH: str = os.path.join(os.path.dirname(__file__), "data")


def test_dataset_loading() -> None:
    ds_path = os.path.join(PATH, "Atlantic_Ocean_07.nc")
    ds_path_fill = os.path.join(PATH, r"Atlantic_Ocean_{val:02d}.nc")

    ds = load_dataset(ds_path)
    assert isinstance(ds, xr.Dataset)
    assert "lx" in ds

    ds_fill = load_dataset(ds_path_fill, val=7)
    assert isinstance(ds_fill, xr.Dataset)
    return None


def test_dataarray_loading() -> None:
    da_path = os.path.join(PATH, "Atlantic_Ocean_07.nc")
    da_path_fill = os.path.join(PATH, r"Atlantic_Ocean_{val:02d}.nc")

    da = load_array(da_path, "lx")
    assert isinstance(da, xr.DataArray)

    da_fill = load_array(da_path_fill, "lx", val=7)
    assert isinstance(da_fill, xr.DataArray)
    return None


def test_covariance_loading() -> None:
    cov_path = os.path.join(PATH, "cov_no_hfix.nc")
    cov = load_covariance(cov_path)

    assert cov.shape[0] == cov.shape[1]
