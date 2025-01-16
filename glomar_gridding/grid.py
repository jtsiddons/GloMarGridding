"""Functions for creating grids and mapping observations to a grid"""

from collections.abc import Iterable
import numpy as np
import polars as pl
import xarray as xr

from .utils import filter_bounds, find_nearest, select_bounds


def align_to_grid(
    obs: pl.DataFrame,
    grid: xr.DataArray,
    grid_coords: list[str] = ["latitude", "longitude"],
    obs_coords: list[str] = ["lat", "lon"],
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
    grid_coords : list[str]
        Names of the coordinates in the input grid DataArray used to define the
        grid.
    obs_coords : list[str]
        Names of the column containing positional values in the input
        observational DataFrame.
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

    obs[grid_prefix + "idx"] = flattened_idx
    if add_grid_pts:
        for grid_pos, obs_coord in zip(obs_to_grid_pos, obs_coords):
            obs[grid_prefix + obs_coord] = grid_pos

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
    grid = xr.DataArray(coords=coords)
    return grid
