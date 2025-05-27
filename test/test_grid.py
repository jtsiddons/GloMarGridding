from datetime import datetime
import os
import pytest  # noqa: F401
import numpy as np
import polars as pl
import xarray as xr
from itertools import product

from glomar_gridding.grid import grid_from_resolution, cross_coords
from glomar_gridding.climatology import (
    join_climatology_by_doy,
    read_climatology,
)
from glomar_gridding.mask import (
    mask_observations,
    mask_dataset,
    mask_array,
)


def new_grid() -> xr.DataArray:
    """Get a new grid for the test"""
    return grid_from_resolution(
        resolution=5,
        bounds=[(-87.5, 90), (-177.5, 180)],
        coord_names=["latitude", "longitude"],
    )


def new_dataframe(n) -> pl.DataFrame:
    lons = 180 - 360 * np.random.rand(n)
    lats = 90 - 180 * np.random.rand(n)
    var = 4 * np.random.rand(n)

    return pl.DataFrame(
        {
            "longitude": lons,
            "latitude": lats,
            "var": var,
        }
    )


def test_grid():
    grid = new_grid()

    assert grid.shape == (36, 72)


def test_cross_grid():
    grid = new_grid()
    crossed_grid = cross_coords(
        grid.coords,
        lat_coord="latitude",
        lon_coord="longitude",
    )
    new_array = xr.DataArray(coords=crossed_grid)

    assert new_array.shape == (2592, 2592)
    crossed_coords = list(
        zip(crossed_grid["latitude_1"], crossed_grid["longitude_1"])
    )
    calc_cross = list(product(grid["latitude"], grid["longitude"]))

    assert crossed_coords == calc_cross


def test_masking_frame():
    mask = new_grid()
    shape = mask.shape
    mask_vals = np.random.rand(*shape) > 0.85

    mask.data = mask_vals

    n_obs = 2000
    df = new_dataframe(n_obs)

    df_masked = df.pipe(
        mask_observations,
        mask=mask,
        varnames="var",
        obs_coords=["latitude", "longitude"],
        align_to_mask=True,
    )

    n_masked = df_masked.filter(pl.col("mask")).height
    assert n_masked > 0
    assert df_masked.filter(pl.col("var").is_nan()).height == n_masked


def test_masking_dataset():
    mask = new_grid()
    shape = mask.shape
    mask_vals = np.random.rand(*shape) > 0.85

    mask.data = mask_vals

    ds = xr.Dataset(coords=mask.coords)

    ds["my_var"] = xr.DataArray(
        coords=mask.coords, name="my_var", data=np.random.rand(*shape) * 4 + 2
    )
    ds["my_other_var"] = xr.DataArray(
        coords=mask.coords,
        name="my_other_var",
        data=np.random.randn(*shape) + 3,
    )

    ds_masked = mask_dataset(ds, mask, varnames="my_var")

    assert np.sum(np.isnan(ds_masked["my_other_var"])) == 0
    assert np.sum(np.isnan(ds_masked["my_var"])) != 0

    assert (mask.values == np.isnan(ds_masked["my_var"].values)).all()

    ds_masked = mask_dataset(ds, mask, varnames=["my_var", "my_other_var"])

    assert np.sum(np.isnan(ds_masked["my_other_var"])) != 0
    assert np.sum(np.isnan(ds_masked["my_var"])) != 0

    assert (mask.values == np.isnan(ds_masked["my_var"].values)).all()

    da = ds["my_var"]
    assert isinstance(da, xr.DataArray)

    da_masked = mask_array(da, mask)
    assert np.sum(np.isnan(da_masked.values)) > 0


def test_clim():
    file = os.path.join(
        os.path.dirname(__file__), "data", "HadSST2_Jan_Clim.nc"
    )

    clim = read_climatology(file, lat_var="latitude", lon_var="longitude")
    if "doy" not in clim.coords:
        clim.coords["doy"] = clim.coords["time"].dt.dayofyear

    assert isinstance(clim, xr.Dataset)
    assert np.max(clim["longitude"]) == 179.5

    n_obs = 2500
    df = new_dataframe(n_obs)
    dates = (
        pl.datetime_range(
            datetime(2009, 1, 1, 0),
            datetime(2009, 2, 1, 0),
            interval="1h",
            closed="left",
            eager=True,
        )
        .sample(
            n_obs,
            with_replacement=True,
        )
        .alias("datetime")
    )
    sst = (
        pl.from_numpy(17 - 19 * np.random.rand(n_obs)).to_series().alias("sst")
    )
    df = df.with_columns(dates, sst)

    df = join_climatology_by_doy(
        df,
        clim,
        lat_col="latitude",
        lon_col="longitude",
        date_col="datetime",
        var_col="sst",
        clim_var="sst",
        temp_from_kelvin=False,
    )

    assert df.height == n_obs
    assert "sst_climatology" in df.columns
    assert "sst_anomaly" in df.columns
