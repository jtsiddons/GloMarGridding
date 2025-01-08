"""Functions for applying masks to grids and DataFrames"""

from typing import Any
from warnings import warn
import pandas as pd
import numpy as np
import xarray as xr
from .grid import align_to_grid


def mask_observations(
    obs: pd.DataFrame,
    mask: xr.DataArray,
    varnames: str | list[str],
    mask_varname: str = "mask",
    mask_coords: list[str] = ["latitude", "longitude"],
    obs_coords: list[str] = ["lat", "lon"],
    mask_value: Any = True,
    drop: bool = False,
    masked_value: Any = np.nan,
    mask_grid_prefix: str = "_mask_grid_",
) -> pd.DataFrame:
    """Mask observations in a DataFrame subject to a mask DataArray"""
    varnames = [varnames] if isinstance(varnames, str) else varnames
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
        add_grid_pts=False,
    )
    obs[mask_varname] = [
        mask[mask_varname].values[i] for i in obs[grid_idx_name]
    ]
    obs.drop(columns=[grid_idx_name], inplace=True)
    if drop:
        return obs.loc[obs[mask_varname] == mask_value]
    for var in varnames:
        obs[var][obs[mask_varname] == mask_value] = masked_value
    return obs


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
