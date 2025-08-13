# Copyright 2025 National Oceanography Centre
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions for applying masks to grids and DataFrames"""

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

    mask_frame = (
        pl.from_numpy(mask.values.flatten())
        .rename({"column_0": "mask"})
        .with_row_index(grid_idx_name)
    )
    obs = obs.join(mask_frame, on=grid_idx_name, how="left")

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
    masked_value : Any
        Value indicating masked values in the DataArray.
    mask_value : Any
        Value to set masked values to in the observational DataFrame.

    Returns
    -------
    grid : xarray.DataArray
        Input xarray.DataArray with the variable masked by the mask DataArray.
    """
    if not isinstance(grid, xr.DataArray):
        raise TypeError("Input 'grid' must be a xarray.DataArray")
    # Check that the grid and mask are aligned
    xr.align(grid, mask, join="exact")

    masked_idx = np.unravel_index(get_mask_idx(mask, mask_value), mask.shape)
    grid.values[masked_idx] = masked_value

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
    if not isinstance(dataset, xr.Dataset):
        raise TypeError("Input 'dataset' must be a xarray.Dataset")
    # Check that the grid and mask are aligned
    xr.align(dataset, mask, join="exact")

    varnames = [varnames] if isinstance(varnames, str) else varnames
    masked_idx = np.unravel_index(get_mask_idx(mask, mask_value), mask.shape)
    print(f"{masked_idx = }")
    for var in varnames:
        dataset[var].values[masked_idx] = masked_value

    return dataset


def mask_from_obs_frame(
    obs: pl.DataFrame,
    coords: str | list[str],
    value_col: str,
    datetime_col: str | None = None,
    grid: xr.DataArray | None = None,
    grid_coords: str | list[str] | None = None,
) -> pl.DataFrame:
    """
    Compute a mask from observations and an optional output grid..

    Positions defined by the "coords" values that do not have any observations,
    at any datetime value in the "datetime_col", for the "value_col" field are
    masked.

    An example use-case would be to identify land positions from sst records.

    If a grid is supplied, the observations are mapped to the grid and any
    positions from the grid that do not contain observations.

    If no grid is supplied, then it is assumed that the observation frame
    represents the full grid, and any positions without observations are
    included with null values in the value_col.

    Parameters
    ----------
    obs : polars.DataFrame
        DataFrame containing observations over space and time. The values in
        the "value_col" field will be used to define the mask.
    coords : str | list[str]
        A list of columns containing the coordinates used to define the mask.
        For example ["lat", "lon"].
    value_col : str
        Name of the column containing values from which the mask will be
        defined.
    datetime_col : str | None
        Name of the datetime column. Any positions that contain no records at
        any datetime value are masked.
    grid : xarray.DataArray | None
        Optional grid, used to map observations so that empty positions can be
        identified. If not supplied, it is assumed that the observations frame
        contains the full grid, and includes nulls in empty positions.
    grid_coords : str | list[str] | None
        Optional grid coordinate names. Must be set if grid is set.

    Returns
    -------
    polars.DataFrame containing coordinate columns and a Boolean "mask" column
    indicating positions that contain no observations and would be a mask value.
    """
    if isinstance(coords, str):
        coords = [coords]
    if isinstance(grid_coords, str):
        grid_coords = [grid_coords]

    if grid is not None:
        if grid_coords is None:
            raise ValueError("grid_coords must be set if grid is set.")
        # Map and Join the Grid such that we have nulls where we don't have
        # observations
        obs = map_to_grid(obs, grid, obs_coords=coords, grid_coords=grid_coords)
        grid_box_coords = [f"grid_{c}" for c in coords]
        grid_df = pl.from_pandas(
            grid.to_dataframe(name="grid").reset_index()
        ).select(grid_coords)

        obs = grid_df.join(
            obs, left_on=grid_coords, right_on=grid_box_coords, how="left"
        )

    datetime_col = datetime_col or "datetime"
    if datetime_col not in obs.columns:
        obs = obs.with_columns(pl.lit(1).alias(datetime_col))

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
    obs: np.ndarray | xr.DataArray,
    datetime_idx: int,
) -> np.ndarray | xr.DataArray:
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
    mask : numpy.ndarray | xarray.DataArray
        A boolean array with dimension reduced along the datetime dimension.
        A True value indicates that all values along the datetime dimension
        for this index are numpy.nan and are masked.
    """
    A = np.isnan(obs)
    mask = A.all(axis=datetime_idx)
    return mask  # type: ignore (array / numpy bool)


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

    Examples
    --------
    >>> data = np.random.rand(4, 4)
    >>> data[data > 0.65] = np.nan
    >>> mask = xr.DataArray(data)
    >>> get_mask_idx(mask)
    array([[1],
           [3],
           [4],
           [5],
           [8]])
    """
    if mask_val is np.nan:
        condition = np.isnan(mask.values)
    else:
        condition = mask.values == mask_val
    if masked:
        return np.argwhere((condition).flatten(order="C"))
    else:
        return np.argwhere((~condition).flatten(order="C"))
