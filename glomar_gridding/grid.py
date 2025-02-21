"""
Grid
----

Functions for creating grids and mapping observations to a grid
"""

from collections.abc import Callable, Iterable
from typing import Any
import numpy as np
import polars as pl
import xarray as xr

from .utils import filter_bounds, find_nearest, select_bounds
from .distances import calculate_distance_matrix, haversine_distance


def map_to_grid(
    obs: pl.DataFrame,
    grid: xr.DataArray,
    obs_coords: list[str] = ["lat", "lon"],
    grid_coords: list[str] = ["latitude", "longitude"],
    sort: bool = True,
    bounds: list[tuple[float, float]] | None = None,
    add_grid_pts: bool = True,
    grid_prefix: str = "grid_",
) -> pl.DataFrame:
    """
    Align an observation dataframe to a grid defined by an xarray DataArray.

    Maps observations to the nearest grid-point, and sorts the data by the
    1d index of the DataArray in a row-major format.

    The grid defined by the latitude and longitude coordinates of the input
    DataArray is then used as the output grid of the Gridding process.

    Parameters
    ----------
    obs : polars.DataFrame
        The observational DataFrame containing positional data with latitude,
        longitude values within the `obs_latname` and `obs_lonname` columns
        respectively. Observations are mapped to the nearest grid-point in the
        grid.
    grid : xarray.DataArray
        Contains the grid coordinates to map observations to.
    obs_coords : list[str]
        Names of the column containing positional values in the input
        observational DataFrame.
    grid_coords : list[str]
        Names of the coordinates in the input grid DataArray used to define the
        grid.
    sort : bool
        Sort the observational DataFrame by the grid index
    bounds : list[tuple[float, float]] | None
        Optionally filter the grid and DataFrame to fall within spatial bounds.
        This list must have the same size and ordering as `obs_coords` and
        `grid_coords` arguments.
    add_grid_pts : bool
        Add the grid positional information to the observational DataFrame.
    grid_prefix : str
        Prefix to use for the new grid columns in the observational DataFrame.

    Returns
    -------
    obs : pandas.DataFrame
        Containing additional `grid_*`, and `grid_idx` values
        indicating the positions and grid index of the observation
        respectively. The DataFrame is also sorted (ascendingly) by the
        `grid_idx` columns for consistency with the gridding functions.
    """
    if bounds is not None:
        grid = select_bounds(grid, bounds, grid_coords)
        obs = filter_bounds(obs, bounds, obs_coords)

    grid_size = grid.shape

    grid_idx: list[list[int]] = []
    obs_to_grid_pos: list[np.ndarray] = []
    for grid_coord, obs_coord in zip(grid_coords, obs_coords):
        grid_pos = grid.coords[grid_coord].values
        _grid_idx, _obs_to_grid_pos = find_nearest(grid_pos, obs[obs_coord])
        grid_idx.append(_grid_idx)
        obs_to_grid_pos.append(_obs_to_grid_pos)
        del _grid_idx, _obs_to_grid_pos

    flattened_idx = np.ravel_multi_index(
        grid_idx,
        grid_size,
        order="C",  # row-major
    )

    obs = obs.with_columns(pl.Series(grid_prefix + "idx", flattened_idx))
    if add_grid_pts:
        obs = obs.with_columns(
            [
                pl.Series(grid_prefix + obs_coord, grid_pos)
                for grid_pos, obs_coord in zip(obs_to_grid_pos, obs_coords)
            ]
        )

    if sort:
        obs = obs.sort("grid_idx", descending=False)

    return obs


def grid_from_resolution(
    resolution: float | list[float],
    bounds: list[tuple[float, float]],
    coord_names: list[str],
) -> xr.DataArray:
    """
    Generate a grid from a resolution value, or a list of resolutions for
    given boundaries and coordinate names.

    Note that all list inputs must have the same length, the ordering of values
    in the lists is assumed align.

    Parameters
    ----------
    resolution : float | list[float]
        Resolution of the grid. Can be a single resolution value that will be
        applied to all coordinates, or a list of values mapping a resolution
        value to each of the coordinates.
    bounds : list[tuple[float, float]]
        A list of bounds of the form `(lower_bound, upper_bound)` indicating
        the bounding box of the returned grid
    coord_names : list[str]
        List of coordinate names

    Returns
    -------
    grid : xarray.DataArray:
        The grid defined by the resolution and bounding box.
    """
    if not isinstance(resolution, Iterable):
        resolution = [resolution for _ in range(len(bounds))]
    if len(resolution) != len(coord_names) or len(bounds) != len(coord_names):
        raise ValueError("Input lists must have the same length")
    coords = {
        c_name: np.arange(lbound, ubound, res)
        for c_name, (lbound, ubound), res in zip(
            coord_names, bounds, resolution
        )
    }
    grid = xr.DataArray(coords=xr.Coordinates(coords))
    return grid


def assign_to_grid(
    values: np.ndarray,
    grid_idx: np.ndarray,
    grid: xr.DataArray,
    mask_grid: bool = False,
    mask_value: Any = np.nan,
) -> xr.DataArray:
    """
    Assign a vector of values to a grid, using a list of grid index values. The
    default value for grid values is 0.0.

    Optionally, if the grid is a mask, apply the mask to the output grid.

    Parameters
    ----------
    values : pl.Series
        The values to map onto the output grid.
    grid_idx : pl.Series
        The 1d index of the grid (assuming "C" style ravelling) for each value.
    grid : xarray.DataArray
        The grid used to define the output grid.
    mask_grid : bool
        Optionally use values in the grid to mask the output grid.
    mask_value : Any
        The value in the grid to use for masking the output grid.

    Returns
    -------
    out_grid : xarray.DataArray
        A new grid containing the values mapped onto the grid.
    """
    out_grid = xr.DataArray(
        data=np.zeros(grid.shape, dtype="float"),
        coords=grid.coords,
    )
    coords_to_assign = np.unravel_index(grid_idx, out_grid.shape, "C")
    out_grid.values[coords_to_assign] = values

    if mask_grid:
        out_grid.values = np.where(grid == mask_value, np.nan, out_grid.values)

    return out_grid


def grid_to_distance_matrix(
    grid: xr.DataArray,
    dist_func: Callable = haversine_distance,
    lat_coord: str = "lat",
    lon_coord: str = "lon",
) -> xr.DataArray:
    """
    Calculate a distance matrix between all positions in a grid. Orientation of
    latitude and longitude will be maintained in the returned distance matrix.

    Parameters
    ----------
    grid : xarray.DataArray
        A 2-d grid containing latitude and longitude indexes specified in
        decimal degrees.
    dist_func : Callable
        Distance function to use to compute pairwise distances. See
        glomar_gridding.distances.calculate_distance_matrix for more
        information.
    lat_coord : str
        Name of the latitude coordinate in the input grid.
    lon_coord : str
        Name of the longitude coordinate in the input grid.

    Returns
    -------
    dist : xarray.DataArray
        A DataArray containing the distance matrix with coordinate system
        defined with grid cell index ("index_1" and "index_2"). The coordinates
        of the original grid are also kept as coordinates related to each
        index (the coordinate names are suffixed with "_1" or "_2" respectively.
    """
    coords = grid.coords
    if len(coords) != 2:
        raise ValueError(
            "Input grid must have 2 indexes - "
            + "specifying latitude and longitude, in decimal degree."
        )
    if lat_coord not in coords:
        raise KeyError(
            f"Cannot find latitude coordinate {lat_coord} in the grid."
        )
    if lon_coord not in coords:
        raise KeyError(
            f"Cannot find longitude coordinate {lon_coord} in the grid."
        )

    coord_df = pl.from_records(
        list(coords.to_index()),
        schema=list(coords.keys()),
        orient="row",
    )

    dist: np.ndarray = calculate_distance_matrix(
        coord_df,
        dist_func=dist_func,
        lat_col=lat_coord,
        lon_col=lon_coord,
    )

    n = coord_df.height
    out_coords: dict[str, Any] = {"index_1": range(n), "index_2": range(n)}
    for i in range(1, 3):
        out_coords.update(
            {f"{c}_{i}": (f"index_{i}", coord_df[c]) for c in coord_df.columns}
        )

    return xr.DataArray(
        dist,
        coords=xr.Coordinates(out_coords),
        name="dist",
    )
