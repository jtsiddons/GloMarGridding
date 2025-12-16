import os
from datetime import datetime
from itertools import product

import numpy as np
import polars as pl
import pytest  # noqa: F401
import xarray as xr

from glomar_gridding.climatology import (
    join_climatology_by_doy,
    read_climatology,
)
from glomar_gridding.grid import cross_coords, grid_from_resolution, Grid
from glomar_gridding.io import load_array
from glomar_gridding.mask import (
    mask_array,
    mask_dataset,
    mask_from_obs_array,
    mask_from_obs_frame,
    mask_observations,
)


TEST_GRID = Grid.from_resolution(
    resolution=5,
    bounds=[(-87.5, 90), (-177.5, 180)],
    coord_names=["latitude", "longitude"],
)


def new_grid() -> xr.DataArray:
    """Get a new grid for the test"""
    return grid_from_resolution(
        resolution=5,
        bounds=[(-87.5, 90), (-177.5, 180)],
        coord_names=["latitude", "longitude"],
        definition="center",
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


def test_grid_definition():
    grid = new_grid()
    grid2 = grid_from_resolution(
        resolution=5,
        bounds=[(-90, 90), (-180, 180)],
        coord_names=["latitude", "longitude"],
        definition="left",
    )
    assert np.all(np.equal(grid.latitude, grid2.latitude))
    assert np.all(np.equal(grid.longitude, grid2.longitude))


def test_grid():
    grid = new_grid()

    assert grid.shape == (36, 72)
    assert grid.shape == TEST_GRID.shape


def test_grid_class():
    grid = Grid.from_resolution(
        resolution=5,
        bounds=[(-87.5, 90), (-177.5, 180)],
        coord_names=["latitude", "longitude"],
    )
    assert grid.shape == (36, 72)
    assert grid.coord_names == ["latitude", "longitude"]
    assert grid.index_map.height == 2592

    mask = np.random.rand(*grid.shape) > 0.75
    expected = 2592 - np.sum(mask)

    grid.add_mask(mask)
    assert grid.index_map.height == expected

    grid.distance_matrix(lat_coord="latitude", lon_coord="longitude")
    assert grid.dist.shape == (expected, expected)

    vals = np.ones(expected, dtype=float)

    vals_on_grid = grid.assign_values(vals, fill_value=0.0)

    assert np.all(mask == (vals_on_grid == 0.0))


def test_cross_grid():
    grid = new_grid()
    crossed_grid = cross_coords(grid.coords)
    new_array = xr.DataArray(coords=crossed_grid)

    assert new_array.shape == (2592, 2592)
    crossed_coords = list(
        zip(crossed_grid["latitude_1"], crossed_grid["longitude_1"])
    )
    calc_cross = list(product(grid["latitude"], grid["longitude"]))

    assert crossed_coords == calc_cross

    grid2 = grid.transpose()
    crossed_grid2 = cross_coords(grid2)
    assert crossed_grid2["latitude_1"][0] == -87.5
    assert crossed_grid2["longitude_1"][0] == -177.5


def test_cross_grid_class():
    crossed_grid = TEST_GRID._cross_coords()
    new_array = xr.DataArray(coords=crossed_grid)

    assert new_array.shape == (2592, 2592)
    crossed_coords = list(
        zip(crossed_grid["latitude_1"], crossed_grid["longitude_1"])
    )
    calc_cross = list(
        product(TEST_GRID.grid["latitude"], TEST_GRID.grid["longitude"])
    )

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


def test_mask_from():
    arr = load_array(
        os.path.join(os.path.dirname(__file__), "data", "HadSST2_Jan_Clim.nc"),
        "sst",
    )
    res = mask_from_obs_array(arr, 0)
    assert res.shape == (360, 180)

    test = arr[0, :, :].values
    test = np.ma.masked_where(~res, test)

    assert np.isnan(test.compressed()).all()

    obs = pl.from_pandas(arr.to_dataframe().reset_index()).rename(
        {"time": "datetime"}
    )
    obs2 = obs.drop_nulls("sst").sample(fraction=0.85)
    grid = grid_from_resolution(
        1, [(-89.5, 90), (-179.5, 180)], ["latitude", "longitude"]
    )

    mask1 = mask_from_obs_frame(
        obs, ["latitude", "longitude"], "sst", "datetime"
    )
    assert mask1.height == 360 * 180
    assert mask1.get_column("mask").any()
    assert not mask1.get_column("mask").all()
    assert mask1.get_column("mask").sum() == np.sum(np.isnan(test.compressed()))

    mask1 = mask1.filter(pl.col("mask")).join(
        obs,
        left_on=["latitude", "longitude"],
        right_on=["latitude", "longitude"],
        how="left",
    )

    assert mask1.get_column("sst").is_null().all()

    mask2 = mask_from_obs_frame(
        obs2,
        ["latitude", "longitude"],
        "sst",
        "datetime",
        grid,
        ["latitude", "longitude"],
    )

    assert mask2.height == 360 * 180
    assert mask2.get_column("mask").any()
    assert not mask2.get_column("mask").all()
    assert mask2.get_column("mask").sum() == np.sum(np.isnan(test.compressed()))

    mask2 = mask2.filter(pl.col("mask")).join(
        obs,
        left_on=["latitude", "longitude"],
        right_on=["latitude", "longitude"],
        how="left",
    )

    assert mask2.get_column("sst").is_null().all()

    return None
