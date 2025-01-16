"""Functions for applying masks to grids and DataFrames"""

from typing import Any
from warnings import warn
import numpy as np
import polars as pl
import xarray as xr

from glomar_gridding.utils import check_cols

from .grid import align_to_grid


def mask_observations(
    obs: pl.DataFrame,
    mask: xr.DataArray,
    varnames: str | list[str],
    mask_varname: str = "mask",
    masked_value: Any = np.nan,
    mask_value: Any = True,
    mask_coords: list[str] = ["latitude", "longitude"],
    obs_coords: list[str] = ["lat", "lon"],
    align_to_mask: bool = False,
    drop: bool = False,
    mask_grid_prefix: str = "_mask_grid_",
) -> pl.DataFrame:
    """Mask observations in a DataFrame subject to a mask DataArray"""
    varnames = [varnames] if isinstance(varnames, str) else varnames
    check_cols(obs, varnames)

    grid_idx_name = mask_grid_prefix + "idx"
    if grid_idx_name in obs.columns:
        warn(
            f"Mask grid idx column '{grid_idx_name}' already in observational "
            + "DataFrame, values will be overwritten"
        )
    obs = align_to_grid(
        obs=obs,
        grid=mask,
        grid_coords=mask_coords,
        obs_coords=obs_coords,
        grid_prefix=mask_grid_prefix,
        sort=False,
        add_grid_pts=align_to_mask,
    )
    obs[mask_varname] = [
        mask[mask_varname].values[i] for i in obs[grid_idx_name]
    ]
    mask_map: dict = {mask_value: masked_value}
    obs = obs.with_columns(
        [
            pl.col(mask_varname)
            .replace_strict(mask_map, default=pl.col(var))
            .alias(var)
            for var in varnames
        ]
    )
    if drop:
        return obs.filter(pl.col(mask_varname).eq(mask_value))
    return obs.drop([grid_idx_name], strict=True)


def mask_array(
    grid: xr.DataArray,
    mask: xr.DataArray,
    varname: str,
    mask_varname: str = "mask",
    mask_value: Any = True,
    masked_value: Any = np.nan,
) -> xr.DataArray:
    """Apply a mask to a DataArray"""
    # Check that the grid and mask are aligned
    xr.align(grid, mask, join="exact")

    masked_idx = mask[mask_varname] == mask_value
    grid[varname][masked_idx] = masked_value

    return grid


def mask_dataset(
    dataset: xr.Dataset,
    mask: xr.DataArray,
    varnames: str | list[str],
    mask_varname: str = "mask",
    mask_value: Any = True,
    masked_value: Any = np.nan,
) -> xr.Dataset:
    """Apply a mask to a DataSet"""
    # Check that the grid and mask are aligned
    xr.align(dataset, mask, join="exact")

    varnames = [varnames] if isinstance(varnames, str) else varnames
    masked_idx = mask[mask_varname] == mask_value
    for var in varnames:
        dataset[var][masked_idx] = masked_value

    return dataset


def mask_from_obs(
    obs: pl.DataFrame,
    coords: str | list[str],
    datetime_col: str,
    value_col: str,
) -> pl.DataFrame:
    """Compute a mask from observations"""
    if isinstance(coords, str):
        coords = [coords]
    x = obs.select([*coords, datetime_col, value_col]).pivot(
        on=datetime_col, index=coords, values=value_col
    )
    return x.select(
        [
            *coords,
            pl.all_horizontal(pl.exclude(*coords).is_null()).alias("mask"),
        ]
    )


def mask_from_obs_array(
    obs: np.ndarray,
    datetime_idx: int,
) -> np.ndarray:
    A = np.isnan(obs)
    mask = A.all(axis=datetime_idx)
    return mask
