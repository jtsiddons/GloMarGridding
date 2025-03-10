"""
Masking
-------

Functions for applying masks to grids and DataFrames
"""

from typing import Any
from warnings import warn
import numpy as np
import polars as pl
import xarray as xr

from glomar_gridding.grid import map_to_grid
from glomar_gridding.utils import check_cols


def mask_observations(
    obs: pl.DataFrame,
    mask: xr.DataArray,
    varnames: str | list[str],
    mask_varname: str = "mask",
    masked_value: Any = np.nan,
    mask_value: Any = True,
    obs_coords: list[str] = ["lat", "lon"],
    mask_coords: list[str] = ["latitude", "longitude"],
    align_to_mask: bool = False,
    drop: bool = False,
    mask_grid_prefix: str = "_mask_grid_",
) -> pl.DataFrame:
    """
    Mask observations in a DataFrame subject to a mask DataArray.

    Parameters
    ----------
    obs : polars.DataFrame
        Observational DataFrame to be masked by positions in the mask
        DataArray.
    mask : xarray.DataArray
        Array containing values used to mask the observational DataFrame.
    varnames : str | list[str]
        Columns in the observational DataFrame to apply the mask to.
    mask_varname : str
        Name of the mask variable in the mask DataArray.
    masked_value : Any
        Value indicating masked values in the DataArray.
    mask_value : Any
        Value to set masked values to in the observational DataFrame.
    obs_coords : list[str]
        A list of coordinate names in the observational DataFrame. Used to map
        the mask DataArray to the observational DataFrame. The order must align
        with the coordinates of the mask DataArray.
    mask_coords : list[str]
        A list of coordinate names in the mask DataArray. These coordinates are
        mapped onto the observational DataFrame in order to apply the mask. The
        ordering of the coordinate names in this list must match those in the
        obs_coords list.
    align_to_mask : bool
        Optionally align the observational DataFrame to the mask DataArray.
        This essentially sets the mask's grid as the output grid for
        interpolation.
    drop : bool
        Drop masked values in the observational DataFrame.
    mask_grid_prefix : str
        Prefix to use for the mask gridbox index column in the observational
        DataFrame.

    Returns
    -------
    obs : polars.DataFrame
        Input polars.DataFrame containing additional column named by the
        mask_varname argument, indicating records that are masked. Masked values
        are dropped if the drop argument is set to True.
    """
    varnames = [varnames] if isinstance(varnames, str) else varnames
    check_cols(obs, varnames)

    grid_idx_name = mask_grid_prefix + "idx"
    if grid_idx_name in obs.columns:
        warn(
            f"Mask grid idx column '{grid_idx_name}' already in observational "
            + "DataFrame, values will be overwritten"
        )
    obs = map_to_grid(
        obs=obs,
        grid=mask,
        obs_coords=obs_coords,
        grid_coords=mask_coords,
        grid_prefix=mask_grid_prefix,
        sort=False,
        add_grid_pts=align_to_mask,
    )
    obs = obs.with_columns(
        pl.Series(
            "mask", [mask[mask_varname].values[i] for i in obs[grid_idx_name]]
        )
    )
    mask_map: dict = {mask_value: masked_value}
    obs = obs.with_columns(
        [
            pl.col("mask")
            .replace_strict(mask_map, default=pl.col(var))
            .alias(var)
            for var in varnames
        ]
    )
    if drop:
        return obs.filter(pl.col("mask").eq(mask_value))
    return obs.drop([grid_idx_name], strict=True)


def mask_array(
    grid: xr.DataArray,
    mask: xr.DataArray,
    varname: str,
    masked_value: Any = np.nan,
    mask_value: Any = True,
) -> xr.DataArray:
    """
    Apply a mask to a DataArray.

    The grid and mask must already align for this function to work. An error
    will be raised if the coordinate systems cannot be aligned.

    Parameters
    ----------
    grid : xarray.DataArray
        Observational DataArray to be masked by positions in the mask
        DataArray.
    mask : xarray.DataArray
        Array containing values used to mask the observational DataFrame.
    varname : str
        Name of the variable in the observational DataArray to apply the mask
        to.
    masked_value : Any
        Value indicating masked values in the DataArray.
    mask_value : Any
        Value to set masked values to in the observational DataFrame.

    Returns
    -------
    grid : xarray.DataArray
        Input xarray.DataArray with the variable masked by the mask DataArray.
    """
    # Check that the grid and mask are aligned
    xr.align(grid, mask, join="exact")

    masked_idx = np.unravel_index(get_mask_idx(mask, mask_value), mask.shape)
    grid[varname][masked_idx] = masked_value

    return grid


def mask_dataset(
    dataset: xr.Dataset,
    mask: xr.DataArray,
    varnames: str | list[str],
    masked_value: Any = np.nan,
    mask_value: Any = True,
) -> xr.Dataset:
    """
    Apply a mask to a DataSet.

    The grid and mask must already align for this function to work. An error
    will be raised if the coordinate systems cannot be aligned.

    Parameters
    ----------
    dataset : xarray.Dataset
        Observational Dataset to be masked by positions in the mask
        DataArray.
    mask : xarray.DataArray
        Array containing values used to mask the observational DataFrame.
    varnames : str | list[str]
        A list containing the names of  variables in the observational Dataser
        to apply the mask to.
    masked_value : Any
        Value indicating masked values in the DataArray.
    mask_value : Any
        Value to set masked values to in the observational DataFrame.

    Returns
    -------
    grid : xarray.Dataset
        Input xarray.Dataset with the variables masked by the mask DataArray.
    """
    # Check that the grid and mask are aligned
    xr.align(dataset, mask, join="exact")

    varnames = [varnames] if isinstance(varnames, str) else varnames
    masked_idx = np.unravel_index(get_mask_idx(mask, mask_value), mask.shape)
    for var in varnames:
        dataset[var][masked_idx] = masked_value

    return dataset


def mask_from_obs_frame(
    obs: pl.DataFrame,
    coords: str | list[str],
    datetime_col: str,
    value_col: str,
) -> pl.DataFrame:
    """
    Compute a mask from observations.

    Positions defined by the "coords" values that do not have any observations,
    at any datetime value in the "datetime_col", for the "value_col" field are
    masked.

    An example use-case would be to identify land positions from sst records.

    Parameters
    ----------
    obs : polars.DataFrame
        DataFrame containing observations over space and time. The values in
        the "value_col" field will be used to define the mask.
    coords : str | list[str]
        A list of columns containing the coordinates used to define the mask.
        For example ["lat", "lon"].
    datetime_col : str
        Name of the datetime column. Any positions that contain no records at
        any datetime value are masked.
    value_col : str
        Name of the column containing values from which the mask will be
        defined.

    Returns
    -------
    polars.DataFrame containing coordinate columns and a Boolean "mask" column
    indicating positions that contain no observations and would be a mask value.
    """
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
    """
    Infer a mask from an input array. Mask values are those where all values
    are NaN along the time dimension.

    An example use-case would be to infer land-points from a SST data array.

    Parameters
    ----------
    obs : numpy.ndarray
        Array containing the observation values. Records that are numpy.nan
        will count towards the mask, if all values in the datetime dimension
        are numpy.nan.
    datetime_idx : int
        The index of the datetime, or grouping, dimension. If all records at
        a point along this dimension are NaN then this point will be masked.

    Returns
    -------
    mask : numpy.ndarray
        A boolean array with dimension reduced along the datetime dimension.
        A True value indicates that all values along the datetime dimension
        for this index are numpy.nan and are masked.
    """
    A = np.isnan(obs)
    mask = A.all(axis=datetime_idx)
    return mask


def get_mask_idx(
    mask: xr.DataArray,
    mask_val: Any = np.nan,
    masked: bool = True,
) -> np.ndarray:
    """
    Get the 1d indices of masked values from a mask array.

    Parameters
    ----------
    mask : xarray.DataArray
        The mask array, containing values indicated a masked value.
    mask_val : Any
        The value that indicates the position should be masked.
    masked : bool
        Return indices where values in the mask array equal this value. If set
        to False it will return indices where values are not equal to the mask
        value. Can be used to get unmasked indices if this value is set to
        False.

    Returns
    -------
    An array of integers indicating the indices which are masked.
    """
    if masked:
        return np.argwhere((mask.values).flatten(order="C") == mask_val)
    else:
        return np.argwhere((mask.values).flatten(order="C") != mask_val)
